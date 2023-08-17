"""
Utility functions for math operations.
"""
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt


def log_std(var: npt.NDArray[Union[np.int_, np.float_]]) -> npt.NDArray[np.float_]:
    """
    Get log-transformed standardized values of a list of variables.

    :param var: An np.array of variables to be transformed.
    :return: An np.array of the transformed variables.
    """
    log_var = np.log(var)

    # If all values are the same, return an array of zeros.
    if np.all(log_var == log_var[0]):
        return np.zeros_like(log_var)

    log_std_var: npt.NDArray[np.float_] = (log_var - np.mean(log_var)) / np.std(log_var)
    return log_std_var


def get_std_devs_and_means(
    *arrays: List[npt.NDArray[Union[np.int_, np.float_]]],
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Get the standard deviations and means of multiple numpy arrays. The values
    are log-transformed first.

    :param arrays: A list of numpy arrays.
    :return: An np.array of the resulting standard deviations, and an np.array
        of the resulting means.
    """
    std_devs_combined = np.array(np.std(np.log(array)) for array in arrays)
    means_combined = np.array(np.mean(np.log(array)) for array in arrays)
    return std_devs_combined, means_combined


def log_std_all(
    variables: npt.NDArray[np.float_],
    sds: npt.NDArray[np.float_],
    means: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    Get the log-transformed standardized values for all variables.

    :param variables: An np.array of variables to be transformed.
    :param sds: An np.array of standard deviations.
    :param means: An np.array of means.
    :return: An np.array of transformed variables.
    """

    for variable in range(variables.shape[1]):
        log_var: npt.NDArray[np.float_] = np.log(variables[:, variable])
        variables[:, variable] = (log_var - means[variable]) / sds[variable]

    return variables.astype(np.float_)
