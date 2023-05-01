from .utils.path import Path
from .utils import math as m
import numpy as np


class FixedStandardWheel:
    def __init__(self, radius: float, max_rpm: float, name: str = 'wheel'):
        self.type = 'fixed_standard'
        self.name = name
        self.radius = radius
        self.max_rpm = max_rpm
        self.max_angular_velocity = max_rpm * np.pi / 30

    def set_angular_velocity(self, angular_velocity: float):
        rounding_dp = 3
        min_ang_vel = m.round(-self.max_angular_velocity,
                              rounding_dp, round_down=True)
        max_ang_vel = m.round(self.max_angular_velocity,
                              rounding_dp, round_down=False)
        if min_ang_vel <= angular_velocity <= max_ang_vel:
            self.angular_velocity = angular_velocity
        else:
            print("Angular velocity out of range")


class MobileRobot:
    def __init__(self, name: str, width: float, wheel_offset: float, wheel_radius: float, wheel_specs: list[dict[str, float]], chassis_height: float = 0.0):
        self.name = name
        self.width = width
        self.chassis_height = chassis_height
        self.wheel_offset = wheel_offset
        self.wheel_radius = wheel_radius
        self.global_position = np.array([0, 0, 0], dtype=float)
        self.global_orientation = 0
        self.x_velocity = 0
        self.y_velocity = 0
        self.angular_velocity = 0
        self.wheels = {wheel_spec['name']: self.__create_wheel_from_spec(
            wheel_spec) for wheel_spec in wheel_specs}

    def __create_wheel_from_spec(self, wheel_spec: dict[str, float]):
        wheel_type = wheel_spec.pop('type')
        if wheel_type == 'fixed_standard':
            return FixedStandardWheel(**wheel_spec)
        else:
            raise NotImplementedError

    @property
    def rot_matrix(self):
        theta = self.global_orientation
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0],
                         [0, 0, 1]])

    @property
    def transformation_matrix(self):
        R = self.rot_matrix
        T = self.global_position
        return np.array([[R[0, 0], R[0, 1],  R[0, 2], T[0]],
                         [R[1, 0], R[1, 1],  R[1, 2], T[1]],
                         [R[2, 0], R[2, 1],  R[2, 2], T[2]],
                         [0, 0, 0, 1]])

    @property
    def inv_transformation_matrix(self):
        R = self.rot_matrix.T
        T = -np.matmul(R, self.global_position)
        return np.array([[R[0, 0], R[0, 1],  R[0, 2], T[0]],
                         [R[1, 0], R[1, 1],  R[1, 2], T[1]],
                         [R[2, 0], R[2, 1],  R[2, 2], T[2]],
                         [0, 0, 0, 1]])

    def set_global_position(self, global_position: list = [0, 0, 0]):
        self.global_position = np.array(global_position, dtype=float)

    def set_global_orientation(self, global_orientation: float):
        self.global_orientation = global_orientation

    def set_path(self, path: Path):
        self.path = path
        self.set_global_position([*path.start_pos, 0])
        self.set_global_orientation(path.start_theta)

    def transform_to_robot_frame(self, target):
        global_target = np.array([*target, 1], dtype=float)
        local_target = np.matmul(
            self.inv_transformation_matrix, global_target.T)
        return local_target[: 3]

    def transform_to_global_frame(self, target):
        local_target = np.array([*target, 1], dtype=float)
        global_target = np.matmul(self.transformation_matrix, local_target)
        return global_target[: 3]


class DiffDriveRobot(MobileRobot):
    def __init__(self, name: str, width: float, chassis_height: float, wheel_offset: float, wheel_radius: float, max_rpm: float):
        wheel_specs = [
            {
                'name': 'left',
                'type': 'fixed_standard',
                'radius': wheel_radius,
                'max_rpm': max_rpm
            },
            {
                'name': 'right',
                'type': 'fixed_standard',
                'radius': wheel_radius,
                'max_rpm': max_rpm
            }
        ]
        super().__init__(name=name,
                         width=width,
                         chassis_height=chassis_height,
                         wheel_offset=wheel_offset,
                         wheel_radius=wheel_radius,
                         wheel_specs=wheel_specs)
        self.type = 'diff_drive'
        self.max_wheel_rpm = max_rpm
        self.max_wheel_angular_velocity = max_rpm * np.pi / 30
        self.max_speed = self.max_wheel_angular_velocity * self.wheel_radius
        [wheel.set_angular_velocity(0) for wheel in self.wheels.values()]

    def set_wheel_speeds(self,
                         left_wheel_speed: float = None,
                         right_wheel_speed: float = None):
        if left_wheel_speed:
            self.wheels['left'].set_angular_velocity(left_wheel_speed)
        if right_wheel_speed:
            self.wheels['right'].set_angular_velocity(right_wheel_speed)
        self.x_velocity, self.y_velocity, self.angular_velocity = self.__solve_forward_kin()

    def __solve_forward_kin(self):
        left_wheel_speed = self.wheels['left'].angular_velocity
        right_wheel_speed = self.wheels['right'].angular_velocity
        A = np.array([0.5 * self.wheel_radius * (left_wheel_speed + right_wheel_speed),
                      0,
                      0.5 * self.wheel_radius * (right_wheel_speed - left_wheel_speed) / self.wheel_offset])
        P = np.matmul(self.rot_matrix, A)
        x_velocity, y_velocity, angular_velocity = P
        return x_velocity, y_velocity, angular_velocity

    def __solve_inv_kin(self, target_x_vel: float, target_y_vel: float, target_angular_vel: float):
        A_inv = np.array([[1, 0, self.wheel_offset],
                          [1, 0, -self.wheel_offset]]) / self.wheel_radius
        P = np.array([target_x_vel, target_y_vel, target_angular_vel]).T
        wheel_speeds = np.matmul(A_inv, np.matmul(self.rot_matrix.T, P))
        right_wheel_speed, left_wheel_speed = wheel_speeds
        return left_wheel_speed, right_wheel_speed

    def __update_position(self, left_wheel_speed, right_wheel_speed, dt):
        self.set_wheel_speeds(left_wheel_speed, right_wheel_speed)
        new_position = self.global_position + np.array(
            [self.x_velocity, self.y_velocity, 0]) * dt
        new_orientation = self.global_orientation + self.angular_velocity * dt
        self.set_global_position(new_position)
        self.set_global_orientation(new_orientation)

    def forward_kin_sim(self,
                        left_wheel_speed: float,
                        right_wheel_speed: float,
                        dt: float,
                        time: float):
        t = np.arange(0, time, dt)
        global_positions = np.zeros((len(t), 3))
        global_orientations = np.zeros(len(t))
        for i in range(len(t)):
            self.__update_position(left_wheel_speed, right_wheel_speed, dt)
            global_positions[i] = self.global_position
            global_orientations[i] = self.global_orientation
        return t, global_positions, global_orientations

    def inverse_kin_sim(self, path: Path, dt: float):
        self.set_path(path)
        all_seg_t = []
        all_seg_global_positions = []
        all_seg_global_orientations = []
        all_seg_left_wheel_speeds = []
        all_seg_right_wheel_speeds = []
        start_time = 0

        for i, segment in enumerate(self.path.segments):
            self.current_segment = i
            seg_time, seg_t, seg_global_positions, seg_global_orientations, seg_left_wheel_speeds, seg_right_wheel_speeds = self.__process_segment_inv_kin(
                segment, dt)
            all_seg_t.append(seg_t + start_time)
            all_seg_global_positions.append(seg_global_positions)
            all_seg_global_orientations.append(seg_global_orientations)
            all_seg_left_wheel_speeds.append(seg_left_wheel_speeds)
            all_seg_right_wheel_speeds.append(seg_right_wheel_speeds)
            start_time += seg_time

        return all_seg_t, all_seg_global_positions, all_seg_global_orientations, all_seg_left_wheel_speeds, all_seg_right_wheel_speeds

    def __process_segment_inv_kin(self, segment, dt):
        if segment.type == 'linear':
            seg_time = segment.distance / self.max_speed
            target_x_vel = self.max_speed * np.cos(segment.start_theta)
            target_y_vel = self.max_speed * np.sin(segment.start_theta)
            target_angular_vel = 0
        elif segment.type == 'circular':
            max_angular_vel = self.max_speed / \
                (segment.radius + self.wheel_offset)
            seg_time = np.abs(segment.d_theta / max_angular_vel)
            theta_0 = self.global_orientation
            alpha_0 = theta_0 - np.pi / 2 * (-1 if segment.clockwise else 1)
            target_angular_vel = max_angular_vel * \
                (-1 if segment.clockwise else 1)

        seg_t = np.arange(0, seg_time, dt)
        seg_global_positions = np.zeros((len(seg_t), 3))
        seg_global_orientations = np.zeros(len(seg_t))
        seg_left_wheel_speeds = np.zeros(len(seg_t))
        seg_right_wheel_speeds = np.zeros(len(seg_t))

        for j, t_j in enumerate(seg_t):
            if segment.type == 'circular':
                alpha = target_angular_vel * t_j + alpha_0
                target_x_vel = -segment.radius * \
                    target_angular_vel * np.sin(alpha)
                target_y_vel = segment.radius * \
                    target_angular_vel * np.cos(alpha)

            left_wheel_speed, right_wheel_speed = self.__solve_inv_kin(
                target_x_vel=target_x_vel,
                target_y_vel=target_y_vel,
                target_angular_vel=target_angular_vel)
            self.__update_position(
                left_wheel_speed, right_wheel_speed, dt)
            seg_global_positions[j] = self.global_position
            seg_global_orientations[j] = self.global_orientation
            seg_left_wheel_speeds[j] = left_wheel_speed
            seg_right_wheel_speeds[j] = right_wheel_speed

        return seg_time, seg_t, seg_global_positions, seg_global_orientations, seg_left_wheel_speeds, seg_right_wheel_speeds


class OmniDriveRobot:
    ...

# class MobileRobot(DiffDriveRobot, OmniDriveRobot):

# class MobileRobot:
#     def __init__(self, name:str, path: Path, )
