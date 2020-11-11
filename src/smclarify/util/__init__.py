import numpy as np
from collections import defaultdict
from functional import seq
from typing import List


def collapse_to_binary(values, pivot=0.0):
    # Collapsing to binary categorical and continuous attributes
    # values = attribute values (e.g. labels or sensitive attribute)
    # pivot = if single float number -> continuous case;
    # otherwise categorical case with pivot as list of positive categories
    if np.isscalar(pivot):  # continuous case: 0 if the attribute is < pivot value, otherwise 1
        nvalues = [1 if el >= pivot else 0 for el in values]
    else:  # categorical case
        nvalues = [1 if el in pivot else 0 for el in values]
    return np.array(nvalues)


def GaussianFilter(input_array: np.array, sigma: int = 1) -> np.array:
    """
    :param input_array: array which Gaussian Filter is applied to
    :param sigma: integer which indicates standard deviation of the desired Gaussian distribution
    :return: smoothed array
    """

    if len(input_array) == 0:
        raise ValueError("input array is empty")

    def GaussianKernel(x: float, sigma: int) -> float:

        return np.exp(-((x ** 2) / (2 * (sigma ** 2)))) * 1 / (np.sqrt(2 * np.pi) * sigma)

    x = np.linspace(1, len(input_array), len(input_array))
    centered_x = x - np.mean(x)

    gauss_filter = GaussianKernel(centered_x, sigma)
    return np.convolve(input_array, gauss_filter, "same")


def pdf(xs) -> dict:
    """
    Probability distribution function
    :param xs: input sequence
    :return: sequence of tuples as (value, frequency)
    """
    counts = seq(xs).map(lambda x: (x, 1)).reduce_by_key(lambda x, y: x + y)
    total = counts.map(lambda x: x[1]).sum()
    result_pdf = counts.map(lambda x: (x[0], x[1] / total)).sorted().list()
    return result_pdf


def pdfs_aligned_nonzero(*args) -> List[np.array]:
    """
    Convert a list of discrete pdfs / freq counts to aligned numpy arrays of the same size for common non-zero elements
    :return: pair of numpy arrays of the same size with the aligned pdfs
    """
    num_pdfs = len(args)
    pdfs = []
    for x in args:
        pdfs.append(pdf(x))

    def keys(_xs):
        return seq(_xs).map(lambda x: x[0])

    # Extract union of keys
    all_keys = seq(pdfs).flat_map(keys).distinct().sorted()

    # Index all pdfs by value
    dict_pdfs = seq(pdfs).map(dict).list()

    # result aligned lists
    aligned_lists: List[List[np.array]] = [[] for x in range(num_pdfs)]

    # fill keys present in all pdfs
    for i, key in enumerate(all_keys):
        for j, d in enumerate(dict_pdfs):
            if d.get(key, 0) == 0:
                break
        else:
            # All keys exist and are != 0
            for j, d in enumerate(dict_pdfs):
                aligned_lists[j].append(d[key])
    np_arrays = seq(aligned_lists).map(np.array).list()
    return np_arrays
