from dataclasses import dataclass
from collections import OrderedDict
from enum import Enum
from typing import List, Dict, Tuple
import itertools

import plotly.graph_objects as go
import numpy as np
from mogra.datatypes import normalize_frequency, ratio_to_swar


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
                xaxis_title = self.primes[0] if len(self.primes) > 0 else "null",
                yaxis_title = self.primes[1] if len(self.primes) > 1 else "null",
                zaxis_title = self.primes[2] if len(self.primes) > 2 else "null",
            ),
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
            marker=dict(size=12, symbol="circle"),
            marker_color=["gold" if mm in swar_set else "midnightblue" for mm in self.node_names],
            text=self.node_names,
            textposition="middle center",
            textfont=dict(family="Overpass", size=10, color="white"),
        )])
        
        fig = self.prep_plot(fig)
        fig.show()


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
        return swar_node_coordinates.tolist(), self.primes


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
