from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum
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
    def __init__(self, primes=[3, 5, 7], powers=[0, 0, 0]) -> None:
        self.primes = primes
        self.powers = powers
    
    @classmethod
    def from_list(cls, genus_list: List):
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

        self.primes = genus.primes
        self.powers = genus.powers
        
        ranges = []
        for prime, power in zip(genus.primes, genus.powers):
            ranges.append(range(-power, power+1))
        self.node_coordinates = list(itertools.product(*ranges))
        
        self.assign_coords3d()
        self.assign_notes()
    
    def prep_plot(self, figure):
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=-1.25, z=1.25)
        )
        figure.update_layout(scene_aspectmode="data", scene_camera=camera)
        figure.update_layout(
            scene=dict(
                xaxis_title = "powers of " + str(self.primes[0]) if len(self.primes) > 0 else "null",
                yaxis_title = "powers of " + str(self.primes[1]) if len(self.primes) > 1 else "null",
                zaxis_title = "powers of " + str(self.primes[2]) if len(self.primes) > 2 else "null",
            ),
            # make axis labels integers only
            scene_xaxis_tickvals=list(range(-self.powers[0], self.powers[0]+1)) if len(self.primes) > 0 else [],
            scene_yaxis_tickvals=list(range(-self.powers[1], self.powers[1]+1)) if len(self.primes) > 1 else [],
            scene_zaxis_tickvals=list(range(-self.powers[2], self.powers[2]+1)) if len(self.primes) > 2 else [],
        )
        return figure
    
    def frequency_from_coord(self, coords):
        ff = 1
        for ii, cc in enumerate(coords):
            ff *= self.primes[ii]**cc
        return ff
    
    def assign_coords3d(self):
        coords = list(zip(*self.node_coordinates))
        # Coordinates for Plotly Scatter3d
        self.coords3d = {i: [0] * len(self.node_coordinates) for i in range(3)}
        for i, coords in enumerate(coords):
            if i < len(coords):
                self.coords3d[i] = coords
    
    def assign_notes(self):
        self.node_frequencies = [
            normalize_frequency(self.frequency_from_coord(nc))
            for nc in self.node_coordinates
        ]
        self.node_names = [
            ratio_to_swar(nf)
            for nf in self.node_frequencies
        ]
    
    def plot(self):
        # Create the 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=self.coords3d[0],
            y=self.coords3d[1],
            z=self.coords3d[2],
            mode="text+markers",
            marker=dict(size=12, symbol="circle"),
            marker_color=["midnightblue" for mm in self.node_names],
            text=self.node_names,
            textposition="middle center",
            textfont=dict(family="Overpass", size=10, color="white"),
        )])
        
        fig = self.prep_plot(fig)
        fig.show()

    def plot_swar_set(self, swar_set):
        fig = go.Figure(data=[go.Scatter3d(
            x=self.coords3d[0],
            y=self.coords3d[1],
            z=self.coords3d[2],
            mode="text+markers",
            marker=dict(
                size=12,
                symbol="circle",
                color=["gold" if mm in swar_set else "midnightblue" for mm in self.node_names]
            ),
            text=self.node_names,
            textposition="middle center",
            textfont=dict(family="Overpass", size=10, color="white"),
        )])
        
        fig = self.prep_plot(fig)
        fig.show()
    
    def plot_swar_hist(self, swar_set, swar_occur):
        fig = go.Figure(data=[go.Scatter3d(
            x=self.coords3d[0],
            y=self.coords3d[1],
            z=self.coords3d[2],
            mode="text+markers",
            marker=dict(
                size=[5 if mm not in swar_set else 100 * swar_occur[swar_set.index(mm)] for mm in self.node_names],
                symbol="circle",
                color=["gold" if mm in swar_set else "midnightblue" for mm in self.node_names]
            ),
            text=self.node_names,
            textposition="middle center",
            textfont=dict(
                # family="Overpass",
                size=[10 if mm not in swar_set else 30 * swar_occur[swar_set.index(mm)] for mm in self.node_names],
                color="dimgray"
            ),
        )])

    def plot_cone(self):
        """
        tonnetz + folded frequency heights
        """
        assert len(self.primes) == 2
        # seq = np.argsort(self.node_frequencies)
        # breakpoint()
        fig = go.Figure(data=[go.Scatter3d(
            x=self.coords3d[0],
            y=self.coords3d[1],
            z=self.node_frequencies,
            mode="text+markers",
            marker=dict(size=12, symbol="circle"),
            marker_color=["midnightblue" for mm in self.node_names],
            text=self.node_names,
            textposition="middle center",
            textfont=dict(family="Overpass", size=10, color="white"),
        )])
        fig = self.prep_plot(fig)
        # fig.update_zaxes(title_text="frequency ratio", type="log")
        fig.update_layout(
            scene=dict(
                xaxis_title = self.primes[0] if len(self.primes) > 0 else "null",
                yaxis_title = self.primes[1] if len(self.primes) > 1 else "null",
                zaxis_title = self.primes[2] if len(self.primes) > 2 else "frequency",
                zaxis_type = "log"
            ),
        )
        fig.show()
        
    def plot1d(self):
        """
        post octave-folding
        """
        seq = np.argsort(self.node_frequencies)
        fig = go.Figure(data=go.Scatter(
            x=[
                sum([np.log(self.primes[ii])*pows[ii] for ii in range(len(self.primes))])
                for pows in np.array(self.node_coordinates)[seq]
            ],  # hints at the power complexity
            y=np.array(self.node_frequencies)[seq],  # just the sorted frequencies
            mode="markers+text",
            marker=dict(size=14, symbol="circle"),
            marker_color=["midnightblue" for mm in np.array(self.node_names)[seq]],
            text=np.array(self.node_names)[seq],
            textposition="middle center",
            textfont=dict(family="Overpass", size=12, color="white"),
        ))
        fig.update_yaxes(title_text="frequency ratio", type="log")
        fig.update_layout(autosize=False, width=700, height=700)
        fig.layout.yaxis.scaleanchor="x"
        fig.show()
    
    def get_swar_options(self, swar):
        swar_node_indices = [nn == swar for nn in self.node_names]
        swar_node_coordinates = np.array(self.node_coordinates)[swar_node_indices]
        return [tuple(nc) for nc in swar_node_coordinates.tolist()], self.primes
    
    def get_neighbors(self, node: List):
        neighbor_indices = []
        for ii, nc in enumerate(self.node_coordinates):
            if sum(abs(np.array(nc)-np.array(node))) == 1:
                neighbor_indices.append(ii)
        return neighbor_indices, [self.node_coordinates[ii] for ii in neighbor_indices]

    def adjacency_matrix(self):
        """
        len(nodes) x len(nodes) matrix; represents geometric lattice
        """
        mat = np.zeros((len(self.node_coordinates), len(self.node_coordinates)), dtype=int)
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
