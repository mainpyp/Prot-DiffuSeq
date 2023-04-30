import numpy as np


def increase_ndarray(input_arr: np.ndarray) -> np.ndarray:
    """This function takes an ndarray and returns an ndarray
    with values between 0 and 1 that are gradually increasing."""

    return_arr = np.linspace(.01, 0.99, input_arr.shape[0])
    return return_arr

    
print(increase_ndarray(np.array([1,2,3,4,5,6,7,8,9,10])))