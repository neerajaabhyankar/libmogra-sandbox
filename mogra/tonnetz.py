from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum
from fractions import Fraction
from typing import List, Dict, Tuple
import itertools

import plotly.graph_objects as go
import numpy as np
from mogra.datatypes import normalize_frequency, ratio_to_swar, Swar

OCCUR_FREQ_THRESHOLD = 0.04  # a normalized probability below this => ignore this note


"""
An N-dimensional bounded tonnetz net can be initialized with N prime numbers and their maximum allowable powers,
i.e. an Euler-Fokker Genus https://en.wikipedia.org/wiki/Euler%E2%80%93Fokker_genus
"""

class EFGenus:
    """
    An N-dimensional bounded tonnetz net can be initialized with
    N prime numbers and their maximum allowable powers,
    i.e. an Euler-Fokker Genus https://en.wikipedia.org/wiki/Euler%E2%80%93Fokker_genus
    """

    def __init__(self, primes=[3, 5, 7], powers=[0, 0, 0]) -> None:
        assert len(primes) == len(
            powers
        ), "the number of primes should match the number of corresponding specified powers"
        self.primes = primes
        self.powers = powers

    @classmethod
    def from_list(cls, genus_list: List):
        """Initializes the genus from a non-decreasing list of prime numbers.
        The number of occurences of a prime number in this list = the max allowable power of that prime.
        """
        primes = []
        powers = []
        for new_prime in genus_list:
            if len(primes) > 0:
                assert new_prime >= primes[-1]
                if new_prime == primes[-1]:
                    powers[-1] += 1
                else:
                    primes.append(new_prime)
                    powers.append(1)
            else:
                primes.append(new_prime)
                powers.append(1)

        return cls(primes, powers)


class Tonnetz:
    def __init__(self, genus) -> None:
        if len(genus.primes) > 3:
            print("cannot handle more than 3 dimensions")
            return

        self.primes: List = genus.primes
        self.powers: List = genus.powers

        ranges = []
        for prime, power in zip(genus.primes, genus.powers):
            ranges.append(range(-power, power + 1))
        self.node_coordinates: List[Tuple] = list(itertools.product(*ranges))

        self.assign_notes()

    def coord_to_ratio(self, coords) -> Fraction:
        """Given a coordinate in the tonnetz net, find
        the octave-normalized relative frequency ratio that it represents.
        """
        ff = Fraction(1)
        for ii, cc in enumerate(coords):
            if cc >= 0:
                ff *= self.primes[ii] ** cc
            else:
                ff /= self.primes[ii] ** (-cc)
        return normalize_frequency(ff)

    def assign_notes(self):
        self.node_ratios: List[Fraction] = [
            self.coord_to_ratio(nc) for nc in self.node_coordinates
        ]
        self.node_names: List[str] = [ratio_to_swar(nf) for nf in self.node_ratios]

    def get_swar_options(self, swar) -> List[Tuple]:
        """Given a Swar, return a list of coordinates
        where the Swar appears in this Tonnetz net
        """
        swar_node_indices = [nn == swar for nn in self.node_names]
        swar_node_coordinates = np.array(self.node_coordinates)[swar_node_indices]
        return [tuple(nc) for nc in swar_node_coordinates.tolist()]

    def get_neighbors(self, node: List) -> Tuple[List, List[Tuple]]:
        """Indices in the self.node_coordinates list
        and coordinates in the net
        of neighbors of a given node
        """
        neighbor_indices = []
        for ii, nc in enumerate(self.node_coordinates):
            if sum(abs(np.array(nc) - np.array(node))) == 1:
                neighbor_indices.append(ii)
        return neighbor_indices, [self.node_coordinates[ii] for ii in neighbor_indices]

    def adjacency_matrix(self):
        """
        len(nodes) x len(nodes) matrix; represents geometric lattice
        """
        mat = np.zeros(
            (len(self.node_coordinates), len(self.node_coordinates)), dtype=int
        )
        for ii, nc in enumerate(self.node_coordinates):
            nb_indices, _ = self.get_neighbors(nc)
            for jj in nb_indices:
                mat[ii, jj] = 1
        return mat

    def equivalence_matrix(tn):
        """
        len(nodes) x 12 matrix; for each swar column, nodes (swar options) for that swar are 1
        """
        mat = np.zeros((len(tn.node_coordinates), 12), dtype=int)
        for ss in range(12):
            swar = Swar(ss).name
            swar_node_indices = [nn == swar for nn in tn.node_names]
            for jj in np.where(swar_node_indices)[0]:
                mat[jj, ss] = 1
        return mat

class TonnetzAlgo1:
    def __init__(self, net: Tonnetz) -> None:
        self.net = net
        # hyperparameters
        # TODO(neeraja): replace placeholder penalties
        self.prime_penalties = [np.exp(pp)/np.exp(5) for ii, pp in enumerate(self.net.primes)]
    
    def compute_prime_complexity(self, node):
        # TODO(neeraja): replace placeholder formula
        return sum([abs(node[ii])*self.prime_penalties[ii] for ii in range(len(node))])
        
    def set_pc12(self, pc12_distribution):
        """ assign initial weights to all the nodes
        """
        assert len(pc12_distribution) == 12
        pc12_distribution = pc12_distribution/np.sum(pc12_distribution)
        self.pc12_distribution = pc12_distribution
        self.node_distribution = [
            pc12_distribution[Swar[nn].value]
            for nn in self.net.node_names
        ]
    
    def plot_swar_hist(self):
        fig = go.Figure(data=[go.Scatter3d(
            x=self.net.coords3d[0],
            y=self.net.coords3d[1],
            z=self.net.coords3d[2],
            mode="text+markers",
            text=self.net.node_names,
            textposition="middle center",
            textfont=dict(
                # family="Overpass",
                size=[30 * mm if mm > OCCUR_FREQ_THRESHOLD else 10 for mm in self.node_distribution],
                color="dimgray"
            ),
        )])
        
        fig = self.net.prep_plot(fig)
        fig.show()

    def consolidate_sa(self):
        sa_options, primes = self.net.get_swar_options("S")
        for sa_option in sa_options:
            if (sa_option == np.zeros(len(primes))).all():
                continue
            self.node_distribution[self.net.node_coordinates.index(sa_option)] = 0
    
    def zero_out_below_threshold(self):
        for ii, nn in enumerate(self.net.node_names):
            if self.node_distribution[ii] < OCCUR_FREQ_THRESHOLD:
                self.node_distribution[ii] = 0

    def consolidate_swar(self, swar):
        # get options
        swar_options, primes = self.net.get_swar_options(swar)
        # keep track of scores
        swar_option_scores = {}
        for swar_option in swar_options:
            # get all the neighbors
            _, nbd = self.net.get_neighbors(swar_option)
            nbd_score = np.sum([self.node_distribution[self.net.node_coordinates.index(nbd_node)] for nbd_node in nbd])
            # compute prime complexity
            prime_complexity = self.compute_prime_complexity(swar_option)
            # TODO(neeraja): replace placeholder formula
            total_score = nbd_score + 1/prime_complexity
            swar_option_scores[swar_option] = total_score
        winning_option = max(swar_option_scores, key=swar_option_scores.get)
        # zero out the rest
        for swar_option in swar_options:
            if swar_option == winning_option:
                continue
            self.node_distribution[self.net.node_coordinates.index(swar_option)] = 0
        
    def execute(self, plot=True):
        if plot:
            print("initial plot")
            self.plot_swar_hist()

        self.consolidate_sa()
        def sort_nonsa_swars(pc12_distribution):
            thresholded_set = np.where(pc12_distribution > OCCUR_FREQ_THRESHOLD)[0]
            nonsa_set = "".join([Swar(ii).name for ii in thresholded_set if ii != 0])
            nonsa_occur = [pc12_distribution[Swar[swar].value] for swar in nonsa_set]
            decreasing = np.argsort(nonsa_occur)[::-1]
            sorted_nonsa_set = [nonsa_set[i] for i in decreasing]
            return sorted_nonsa_set

        self.zero_out_below_threshold()
        for ss in sort_nonsa_swars(self.pc12_distribution):
            self.consolidate_swar(ss)

        if plot:
            print("final plot")
            self.plot_swar_hist()

        result = {}
        for nd in self.net.node_coordinates:
            if self.node_distribution[self.net.node_coordinates.index(nd)] > 0:
                result[ratio_to_swar(normalize_frequency(self.net.frequency_from_coord(nd)))] = normalize_frequency(self.net.frequency_from_coord(nd))
        result = OrderedDict(sorted(result.items(), key=lambda x: x[1]))
        return result


""" Unit Tests """

def unit_tests():
    g1 = EFGenus.from_list([3,3,5])
    assert len(g1.primes) == 2
    assert g1.powers == [2, 1]
    
    tn = Tonnetz(g1)
    assert len(set(tn.node_names)) == 12


if __name__ == "__main__":
    g1 = EFGenus.from_list([3,3,3,5])
    tn = Tonnetz(g1)
    
    swar_set = "Sgn"
    tn.plot_swar_set(swar_set)
