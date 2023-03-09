import numpy as np
from utils import utm_to_bev, rotate_vec_2d, calculate_distance

class ToBEV(object):

    def __init__(self) -> None:
        pass

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """ Transform (x, y, heading) of ego and objects from UTM to BEV.
            Transform (v_x, v_y) of objects from UTM to BEV.

        Args:
            observation (np.ndarray): Original observation
            ([x, y, heading, v_x, v_y, a_x, a_y, valid])

        Returns:
            np.ndarray: Transformed observation
        """

        assert len(observation) != 0, 'Empty input recived!'
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

class FilterObjectsByRadius(object):

    def __init__(self,
        roi_radius: float,
        num_objects: int,
        is_shuffle_objects: bool = True
    ) -> None:

        self.roi_radius = roi_radius
        self.num_objects = num_objects
        self.is_shuffle_objects = is_shuffle_objects

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """ Remove virtual objects and objects out of ROI
            Reshape to (num_objects(including ego), features)
            Shuffle objects (Optional, true by default)

        Args:
            observation (np.ndarray): Original observation

        Returns:
            np.ndarray: Filtered observation
        """
        assert len(observation) != 0, 'Empty input recived!'

        # filter virtual objects
        observation = self.filter_virtual_objects(observation)

        # remove far objects
        observation = self.filter_far_objects(observation)

        # slice or pad
        observation = self.resize_observation(observation)

        # shuffle
        if self.is_shuffle_objects:
            self.shuffle_objects(observation)
        
        return observation
    
    def filter_virtual_objects(self, observation: np.ndarray) -> np.ndarray:   
        observation_raw = observation.copy()
        real_selections = observation_raw[:, -1].astype(bool)
        observation = observation_raw[real_selections, :]
        return observation
    
    def filter_far_objects(self, observation: np.ndarray) -> np.ndarray:
        observation_raw = observation.copy()
        ego_loc = observation_raw[0][0:2]
        distance_array = np.array(
            [calculate_distance(obj_loc, ego_loc) for obj_loc in observation_raw[:, 0:2]]
        )
        near_selections = distance_array <= self.roi_radius
        observation = observation_raw[near_selections, :]
        return observation
    
    def resize_observation(self, observation: np.ndarray) -> np.ndarray:
        observation_raw = observation.copy()
        num_real_objects = len(observation_raw)
        num_padding = self.num_objects - num_real_objects

        if num_padding <= 0:
            observation = observation_raw[0:self.num_objects, :]
        else:
            observation_virtual = np.zeros((num_padding, observation_raw.shape[1]))
            observation = np.concatenate((observation_raw, observation_virtual), axis=0)

        return observation
    
    def shuffle_objects(self, observation: np.ndarray) -> None:
        np.random.shuffle(observation[1:, :])
        pass