import unittest
from src.robots import HexaArm, QuadArm
from src.utils import geometry as geom
import numpy as np


class TestQuadBot(unittest.TestCase):
    def setUp(self):
        self.robot = QuadArm(theta_1_init=np.pi,
                             theta_2_init=-np.pi/4,
                             theta_4_init=-np.pi/5,
                             d_3_init=50,
                             L_1=60,
                             L_2=40,
                             L_4=10,
                             L_E=30,
                             theta_1_range=[-np.pi/2, np.pi/2],
                             theta_2_range=[-np.pi/2, np.pi/2],
                             theta_4_range=[-np.pi/2, np.pi/2],
                             d_3_range=[0, 100],
                             FLOOR_Z=0)

    def test_forward_kin(self):
        rounding_dp = 5
        index_e = self.robot.joint_link_orders['end_effector']
        end_eff = self.robot.joint_links[index_e]
        end_eff_pos = end_eff.global_position
        end_eff_pos_rounded = np.round(end_eff_pos, rounding_dp)
        end_eff_forward_kin_pos = self.robot.get_end_effector_position()
        end_eff_forward_kin_pos_rounded = np.round(
            end_eff_forward_kin_pos, rounding_dp)
        self.assertTrue(np.array_equal(
            end_eff_pos_rounded, end_eff_forward_kin_pos_rounded))

    def test_inverse_kin(self):
        joint_positions = [np.pi, -np.pi/4, -np.pi/5, 50]
        target_pos = self.robot.forward_kin_pos(joint_positions)
        target_orientation = self.robot.forward_kin_orientation(
            joint_positions)
        target = geom.create_t_matrix(target_orientation, target_pos)
        t1, t2, d3, t4 = self.robot.add_target(target)
        calc_end_eff_pos = self.robot.forward_kin_pos([t1, t2, d3, t4])
        calc_end_eff_pos_rounded = np.fix(calc_end_eff_pos)
        target_pos_rounded = np.fix(target_pos)
        self.assertTrue(np.array_equal(
            calc_end_eff_pos_rounded, target_pos_rounded))


if __name__ == '__main__':
    unittest.main()
