import numpy as np

def rotate_vec_2d(
    vec: np.ndarray,
    radian: float,
) -> np.ndarray:
    """ Rotate a 2D vector clockwise.

    Args:
        vec (np.ndarray): Query 2d vector.
        radian (float): Radian to rotate, in rad.

    Returns:
        np.ndarray: Result 2d vector.
    """

    rotation_matrix = np.array([
        [np.cos(radian), -np.sin(radian)],
        [np.sin(radian), np.cos(radian)]
    ])

    return np.matmul(rotation_matrix, vec)

def calculate_distance(
    query_point: np.ndarray,
    reference_point: np.ndarray
) -> float:
    """ Calculate distance between two points of any dimension.

    Args:
        query_point (np.ndarray): The query point.
        reference_point (np.ndarray): The reference point.

    Returns:
        float: distance
    """
    return np.linalg.norm(query_point-reference_point)