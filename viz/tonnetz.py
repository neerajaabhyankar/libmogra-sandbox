# import manim
from manim import *

import os,sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(""))
sys.path.append(SCRIPT_DIR)

from mogra import tonnetz

# define the net
gs = tonnetz.EFGenus.from_list([3,3,3,3,5,5])
tn = tonnetz.Tonnetz(gs)

class IllustrateTonnetz(ThreeDScene):
    def init(self):
        self.node_locs = [
            [tn.coords3d[0][ii], tn.coords3d[1][ii], 0]
            for ii in range(len(tn.node_names))
        ]
        self.label_locs = [
            [tn.coords3d[0][ii], tn.coords3d[1][ii], 0.1]
            for ii in range(len(tn.node_names))
        ]
    
    def construct2dtop(self):
        # Set up the axes
        self.axes_2dtop = Axes(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            x_length=16,
            y_length=16,
            axis_config={"color": GRAY_A}
        )
        # Add axis labels
        self.x_label_2dtop = self.axes_2dtop.get_x_axis_label(tn.primes[0])
        self.y_label_2dtop = self.axes_2dtop.get_y_axis_label(tn.primes[1])
        self.y_label_2dtop.rotate(-90 * DEGREES)

        # Add points and labels
        points = [
            Dot(
                point=self.axes_2dtop.c2p(*(self.node_locs[ii][:2])),
                radius=0.2, color=DARK_BLUE
            )
            for ii in range(len(tn.node_names))
        ]
        labels = [
            Text(tn.node_names[ii], font_size=12, color=BLACK).move_to(
                self.axes_2dtop.c2p(*(self.label_locs[ii][:2]))
            )
            for ii in range(len(tn.node_names))
        ]
        
        return points, labels
    
    def show2dtop(self, points, labels):
        # Add everything to the scene
        self.set_camera_orientation(
            zoom=0.8
        )
        self.add(self.axes_2dtop, self.x_label_2dtop, self.y_label_2dtop)
        self.add(*points, *labels)
        self.wait(3)
        
    def construct3d(self):
        # Set up the axes
        self.axes_3d = ThreeDAxes(
            x_range=[-6, 6, 1],
            y_range=[-6, 6, 1],
            z_range=[0, 12, 1],
            x_length=16,
            y_length=16,
            z_length=16,
            axis_config={"color": GRAY_A}
        )
        # Add axis labels
        self.x_label_3d = self.axes_3d.get_x_axis_label(tn.primes[0])
        self.y_label_3d = self.axes_3d.get_y_axis_label(tn.primes[1])
        self.y_label_3d.rotate(-90 * DEGREES)
        self.z_label_3d = self.axes_3d.get_z_axis_label("Octave-Normalized Frequency")

        # Add points and labels
        points = [
            Dot3D(
                point=self.axes_3d.c2p(*self.node_locs[ii]),
                radius=0.2, color=DARK_BLUE, fill_opacity=1, resolution=16
            )
            for ii in range(len(tn.node_names))
        ]
        labels = [
            Text(tn.node_names[ii], font_size=12, color=BLACK).move_to(
                self.axes_3d.c2p(*self.label_locs[ii])
            )
            for ii in range(len(tn.node_names))
        ]
        for label in labels:
            self.add_fixed_in_frame_mobjects(label)
        
        return points, labels
    
    def show3d(self, points, labels):
        # Add everything to the scene
        self.set_camera_orientation(
            phi=0*DEGREES, theta=0*DEGREES, gamma=90*DEGREES,
            frame_center=[0,0,0], zoom=0.8
        )
        self.add(self.axes_3d, self.x_label_3d, self.y_label_3d, self.z_label_3d)
        self.add(*points, *labels)
        self.wait(3)
        self.move_camera(
            phi=90*DEGREES, theta=-90*DEGREES, gamma=0*DEGREES,
            frame_center=[0, 0, 4], zoom=0.8
        )
        self.wait(3)
    
    def construct(self):
        self.init()
        # p2t, l2t = self.construct2dtop()
        p3, l3 = self.construct3d()
        
        # self.show2dtop(p2t, l2t)
        # self.play(
        #     ReplacementTransform(self.axes_2dtop, self.axes_3d),
        #     ReplacementTransform(self.y_label_2dtop, self.axes_3d.get_y_axis_label("5").rotate(90 * DEGREES)),
        #     FadeIn(self.z_label_3d)
        # )
        self.show3d(p3, l3)
        
        self.wait(5)
