import numpy as np
from utils.coordinates_transformation import utm_to_bev
from utils.math import rotate_vec_2d

class ToBEV(object):

    def __init__(self):
        super().__init__()

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """ Transform (x, y, heading) of ego and objects from UTM to BEV.
            Transform (v_x, v_y) of objects from UTM to BEV.

        Args:
            observation (np.ndarray): Raw observation
            ([x, y, heading, v_x, v_y, a_x, a_y, valid])

        Returns:
            np.ndarray: Transformed observation
        """

        observation_raw = observation.copy()
        
        # for objects (x, y, heading) 
        observation[1:, 0:3] = utm_to_bev(observation_raw[1:, 0:3], *observation_raw[0, 0:3])
        # for objects (v_x, v_y): rotate axis is equivalent to negative rotate vector
        observation[1:, 3:5] = np.array([
            rotate_vec_2d(obj_v, -observation_raw[0][2]) for obj_v in observation_raw[1:, 3:5]
        ])
        # for objects (a_x, a_y): rotate axis is equivalent to negative rotate vector
        observation[1:, 5:7] = np.array([
            rotate_vec_2d(obj_a, -observation_raw[0][2]) for obj_a in observation_raw[1:, 5:7]
        ])
        # for ego (x, y. heading)
        observation[0, 0:3] = np.zeros(3)
        # for ego (v_x, v_y): already w.r.t. BEV
        # for ego (a_x, a_y): already w.r.t. BEV

        return observation
