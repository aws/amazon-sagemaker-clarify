import numpy as np


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


def PDF(x: np.array) -> np.array:
    """
    #Calculates the estimated probability distribution based on input sample
    :param x: input array
    :return: probability distribution of the input array
    """
    y = np.unique(x)

    p, bins_edges = np.histogram(x, range=(0, 1))
    filtered_zeros = p[p > 0]
    normalized = filtered_zeros / np.sum(filtered_zeros)

    return normalized
