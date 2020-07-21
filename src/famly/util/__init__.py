import numpy as np
from collections import defaultdict
from functional import seq


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


def pdfs_aligned_nonzero(xs, ys) -> (np.array, np.array):
    """
    Convert a pair of discrete pdfs / freq counts to aligned numpy arrays of the same size for common non-zero elements
    :return: pair of numpy arrays of the same size with the aligned pdfs
    """
    # Calculate pdf
    xs_f = pdf(xs)
    ys_f = pdf(ys)

    def keys(_xs):
        return seq(_xs).map(lambda x: x[0])

    # Extract union of keys
    all_keys = keys(xs_f).union(keys(ys_f)).sorted()

    # Fill numpy arrays for nonzero elements
    xs_f_d = dict(xs_f)
    ys_f_d = dict(ys_f)
    xs_f_lst = []
    ys_f_lst = []
    for i, key in enumerate(all_keys):
        if key in xs_f_d and key in ys_f_d and xs_f_d[key] and ys_f_d[key]:
            xs_f_lst.append(xs_f_d[key])
            ys_f_lst.append(ys_f_d[key])
    return (np.array(xs_f_lst), np.array(ys_f_lst))
