from src.utils import geometry as geom
import numpy as np


class LinearSegment:
    def __init__(self, start_pos, end_pos):
        self.type = 'linear'
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.start_theta = geom.get_angle_between_points(start_pos, end_pos)
        self.end_theta = self.start_theta
        self.d_theta = 0
        self.distance = geom.get_distance_between_points(start_pos, end_pos)
        # self.radius = 0


class CircularSegment:
    def __init__(self, start_pos, end_pos, center_pos):
        self.type = 'circular'
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.center = np.array(center_pos)
        self.radius = geom.get_distance_between_points(start_pos, center_pos)
        # self.start_theta = geom.get_angle_between_points(
        #     start_pos, center_pos) + np.pi / 2
        # self.end_theta = geom.get_angle_between_points(
        #     end_pos, center_pos) + np.pi / 2
        self.start_theta, self.end_theta, self.d_theta, self.clockwise = geom.get_arc_angles(
            start_pos, end_pos, center_pos
        )
        self.direction = -1 if self.clockwise else 1
        self.distance = np.abs(self.d_theta * self.radius)


class Path:
    def __init__(self, segments: list[LinearSegment or CircularSegment]):
        self.segments = segments
        self.length = sum([segment.distance for segment in self.segments])
        self.start_pos = self.segments[0].start_pos
        self.end_pos = self.segments[-1].end_pos
        self.start_theta = self.segments[0].start_theta
        self.end_theta = self.segments[-1].end_theta
