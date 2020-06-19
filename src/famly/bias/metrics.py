import logging
# from typing import dict

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

log = logging.getLogger(__name__)

#Global Variables
pretraining_metrics = ['class_imbalance', 'diff_positive_labels', 'kl_divergence', 'JS', 'LPnorm', 'TVD', 'KS', 'CDD']
posttraining_metrics = ['DPPL', 'DI', 'DCO', 'RD', 'DLR', 'PD', 'AD', 'TE']

#Helper Functions
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

def GaussianFilter(input_array: np.array, sigma: int) -> np.array:
    #
    # Returns a 'smoothed' version of input array, using a Gaussian Filter
    #

    if len(input_array) == 0:
        raise ValueError("input array is empty")

    def GaussianKernel(x: float, sigma: int) -> float:

        return np.exp(-((x ** 2) / (2 * (sigma ** 2)))) * 1 / (np.sqrt(2 * np.pi) * sigma)

    x = np.linspace(1, len(input_array), len(input_array))
    centered_x = x - np.mean(x)

    gauss_filter = GaussianKernel(centered_x, sigma)
    return np.convolve(input_array, gauss_filter, 'same')

def prob_pdf(x: np.array) -> (np.array, np.array):
    y = np.unique(x)

    p, bin_edges = np.histogram(x)

    p = p[p > 0]
    p = p / np.sum(p)
    return p, y

def calculateSmoothedDistributions(a: pd.Series, b: pd.Series, sigma: int) -> (pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series):
    ab = np.append(a, b)  # merge the two samples
    print('past')
    nbins = max(2, int(max(len(a), len(b)) / 10))  # set the number of bins to 10% of sample size

    bins = np.linspace(np.min(ab), np.max(ab), nbins)  # bin values for the histogram

    ahist, bhist, abhist = (
        np.histogram(a, bins=bins)[0],
        np.histogram(b, bins=bins)[0],
        np.histogram(ab, bins=bins)[0])  # generate histograms counts

    print('ahist', a)
    print('bhist', b)
    print('abhist', ab)
    print('bins', bins)

    if sigma == 1:
        ahist, bhist, abhist = (GaussianFilter(ahist, sigma), GaussianFilter(bhist, sigma),
                                GaussianFilter(abhist, sigma))  # smoothing of the histogram

    ahist = np.where(ahist == 0, 0.0001, ahist)  # deal with zero values
    bhist = np.where(bhist == 0, 0.0001, bhist)
    abhist = np.where(abhist == 0, 0.0001, abhist)
    prob_a = ahist / np.sum(ahist)  # probabilities
    prob_b = bhist / np.sum(bhist)
    prob_ab = abhist / np.sum(abhist)

    return ahist, bhist, prob_a, prob_b, prob_ab, bins

def metric_one_vs_all(metric, x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series=None, predicted_labels: pd.Series=None, labels: pd.Series=None, strata: pd.Series=None) -> dict:
    """
    Calculate any metric for a categorical series doing 1 vs all
    :param metric: a function defined in this file which computes a metric
    :param x: pandas series
    :param facet_index: facet containing multicategory values for each element in x
    :param positive_label_index: series of boolean values indicating positive target labels
    :param labels: series of true labels (optional)
    :return:
    """

    categories = facet_index.unique()
    res = dict()
    for cat in categories:
        if len(np.unique(labels)) <= 2:
            if metric.__name__ in pretraining_metrics:
                if metric != CDD:
                    res[cat] = metric(x, facet_index == cat, positive_label_index)[1]
                else:
                    res[cat] = metric(x, facet_index == cat, positive_label_index, strata)[1]
            else:
                if metric in [DPPL, DI]:
                    res[cat] = metric(x, facet_index == cat, predicted_labels)
                else:
                    res[cat] = metric(x, facet_index == cat, labels, predicted_labels)
        else:
            res[cat] = label_one_vs_all(metric, x, facet_index, category=cat, predicted_labels=predicted_labels, labels=labels, strata=strata)

    return res

def label_one_vs_all(metric, x: pd.Series, facet_index: pd.Series, category, predicted_labels: pd.Series=None, labels: pd.Series=None, strata: pd.Series=None) -> dict:
    values = {}
    label_unique = np.unique(labels)

    for label in label_unique:
        if metric.__name__ in pretraining_metrics:
            if metric != CDD:
                values[label] = metric(x, facet_index == category, labels == label)[1]
            else:
                values[label] = metric(x, facet_index == category, labels == label, strata)[1]
        else:
            if metric in [DPPL, DI]:
                values[label] = metric(x, facet_index == category, predicted_labels == label)[1]
            else:
                values[label] = metric(x, facet_index == category, labels == label, predicted_labels == label)[1]

    return values

def class_imbalance(x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series) -> dict:
    """
    Class imbalance (CI)
    :param x: pandas series
    :param facet_index: boolean index series selecting protected instances
    :param positive_label_index: series of boolean values indicating positive target labels
    :return: a float in the interval [-1, +1] indicating an under-representation or over-representation
    of the protected class.

    Bias is often generated from an under-representation of
    the protected class in the dataset, especially if the desired “golden truth”
    is equality across classes. Imbalance carries over into model predictions.
    We will report all measures in differences and normalized differences. Since
    the measures are often probabilities or proportions, the differences will lie in
    We define CI = (np − p)/(np + p). Where np is the number of instances in the not protected group
    and p is number of instances in the protected group.
    """
    res = dict()

    pos = np.sum(facet_index)
    q = len(facet_index)
    neg = q - pos


    if neg == 0:
        raise ValueError("class_imbalance: negated facet set is empty. Check that x[~facet_index] has non-zero length.")
    if pos == 0:
        raise ValueError("class_imbalance: facet set is empty. Check that x[facet_index] has non-zero length.")

    assert q != 0

    ci = float(neg - pos) / q
    res[1] = ci

    return res


def diff_positive_labels(x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series) -> dict:
    """
    Difference in positive proportions in predicted labels
    :param x: pandas series of the target column
    :param facet_index: boolean series indicating protected class
    :param label: pandas series of labels (binary, multicategory, or continuous)
    :param positive_label_index: consider this label value as the positive value, default is 1.
    :return: a float in the interval [-1, +1] indicating bias in the labels.
    """

    positive_label_index_neg_facet = (positive_label_index) & ~facet_index
    positive_label_index_facet = (positive_label_index) & facet_index

    np = len(x[~facet_index])
    p = len(x[facet_index])

    n_pos_label_neg_facet = sum(positive_label_index_neg_facet)
    n_pos_label_facet = sum(positive_label_index_facet)

    if np == 0:
        raise ValueError("diff_positive_labels: negative facet set is empty.")
    if p == 0:
        raise ValueError("diff_positive_labels: facet set is empty.")
    q_neg = n_pos_label_neg_facet / np
    q_pos = n_pos_label_facet / p
    if (q_neg + q_pos) == 0:
        raise ValueError("diff_positive_labels: label facet is empty.")
    res = (q_neg - q_pos) / (q_neg + q_pos)

    return {1: res}

# def KL_binary(x: np.array, y: np.array) -> dict:
#     # KL Divergence  - binary label case
#     x_a = y[x == 0]
#     x_d = y[x == 1]
#     Pa, ya = prob_pdf(x_a)  # x: raw values of the variable (column of data)
#     Pd, yd = prob_pdf(x_d)
#
#     if len(Pa) == len(Pd):
#         kl_divergence = np.sum(Pa * np.log(Pa / Pd))  # note log is base e, measured in nats
#     else:
#         raise ValueError("a and d are not absolutely continuous")
#     return {1 : kl_divergence}
#
# def KL_multicategory(x: np.array, y:np.array) -> dict:
#     # KL Divergence  - multicategory label case
#     y_categories = np.unique(y)
#     res = dict()
#
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         res[y_true] = KL_binary(x, y_tmp)[1]
#     return res
#
# For continuous label case
# #####################################################################################################################
# def KL_from_rawdata(a: np.array, b: np.array, sigma=1) -> float:  # Set sigma=1 if we want smoothed histogram, else =0
#     prob_a, prob_b = calculateSmoothedDistributions(a, b, sigma)[2:4]
#
#     return np.sum(prob_b * np.log(prob_b / prob_a))
#
#
# def KL_continuous(facet_index: pd.Series, y: np.array) -> dict:
#     # KL Divergence  - continuous case
#     y1 = y[facet_index == 1]  # instead of x_a, x_d, using y1, y2, to signify continuous case
#     y2 = y[facet_index == 0]
#     print('y1', y1)
#     print('y2', y2)
#
#     kl = KL_from_rawdata(np.array(y1),np.array(y2))
#     return {1 : kl}
# #####################################################################################################################

def kl_divergence(x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series) -> dict:
    # KL Divergence
    x_a = positive_label_index[facet_index == 0]
    x_d = positive_label_index[facet_index == 1]
    Pa, ya = prob_pdf(x_a)  # x: raw values of the variable (column of data)
    Pd, yd = prob_pdf(x_d)

    if len(Pa) == len(Pd):
        kl = np.sum(Pa * np.log(Pa / Pd))  # note log is base e, measured in nats
    else:
        kl = "a and d are not absolutely continuous"
    return {1 : kl}

# def JS_binary(Xr: np.array, y: np.array) -> dict:
#     # JS Divergence - binary label case
#     x_a = y[Xr == 0]
#     x_d = y[Xr == 1]
#     Pa, ya = prob_pdf(x_a)  # x: raw values of the variable (column of data)
#     Pd, yd = prob_pdf(x_d)
#     if len(Pa) == len(Pd):
#         P, yy = prob_pdf(y)
#         js_divergence = 0.5 * (np.sum(Pa * np.log(Pa / P)) + np.sum(Pd * np.log(Pd / P)))  # note log is base e, measured in nats
#     else:
#         js_divergence = "Failed. a and d are not absolutely continuous."
#     return {1: js_divergence}
#
#
# def JS_multicategory(Xr: np.array , y: np.array) -> dict:
#     # JS Divergence  - multicategory label case
#     y_categories = np.unique(y)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         res[y_true] = KL_binary(Xr, y_tmp)[1]
#     return res
#
#
# def JS_continuous(Xr: np.array, y: np.array) -> dict:
#     # JS Divergence  - continuous label case
#     y1 = y[Xr == 1]
#     y2 = y[Xr == 0]
#     js = 0.5 * (c(y1, y)[0] + KL_from_rawdata(y2, y)[0])
#     return {'y': js}


def JS(x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series) -> dict:
    # JS Divergence
    x_a = positive_label_index[facet_index == 0]
    x_d = positive_label_index[facet_index == 1]

    Pa, ya = prob_pdf(x_a)  # x: raw values of the variable (column of data)
    Pd, yd = prob_pdf(x_d)

    if len(Pa) == len(Pd):
        P, yy = prob_pdf(positive_label_index)
        js_divergence = 0.5 * (np.sum(Pa * np.log(Pa / P)) + np.sum(Pd * np.log(Pd / P)))  # note log is base e, measured in nats
    else:
        js_divergence = "Failed. a and d are not absolutely continuous."

    return {1: js_divergence}

# def LPnorm_binary(Xr: np.array, y: np.array, q=2) -> dict:  # Chose q=2
#     # Lp-norm - binary label case
#     x_a = y[Xr == 0]
#     x_d = y[Xr == 1]
#     Pa, ya = prob_pdf(x_a)  # x: raw values of the variable (column of data)
#     Pd, yd = prob_pdf(x_d)
#     if len(Pa) == len(Pd):
#         lp_norm = np.linalg.norm(Pa - Pd, q)
#     else:
#         lp_norm = "Failed. a and d are not absolutely continuous."
#     return {1: lp_norm}
#
#
# def LPnorm_multicategory(Xr: np.array, y: np.array, q=2) -> dict:
#     # LP-norm  - multicategory label case
#     y_categories = np.unique(y)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         res[y_true] = LPnorm_binary(Xr, y_tmp, q=2)[1]
#     return res
#
# #####################################################################################################################
# def LPnorm_continuous(Xr: np.array, y: np.array, q=2) -> dict:
#     # LP-norm  - continuous label case
#     y1 = y[Xr == 1]
#     y2 = y[Xr == 0]
#     p_a, p_b = KL_from_rawdata(y1, y2)[3:5]
#     lp_norm = np.linalg.norm(p_a - p_b, q)
#     return {'y': lp_norm}
# #####################################################################################################################


def LPnorm(x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series, q: int=2) -> dict:
    # LP-norm
    x_a = positive_label_index[facet_index == 0]
    x_d = positive_label_index[facet_index == 1]

    Pa, ya = prob_pdf(x_a)  # x: raw values of the variable (column of data)
    Pd, yd = prob_pdf(x_d)

    if len(Pa) == len(Pd):
        lp_norm = np.linalg.norm(Pa - Pd, q)
    else:
        lp_norm = "Failed. a and d are not absolutely continuous."

    return {1: lp_norm}


def TVD(x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series) -> dict:
    # Total Variation Distance
    Lp_res = LPnorm(x, facet_index, positive_label_index, q=1)


    if type(Lp_res[1]) != str:
        Lp_res[1] = 0.5 * np.array(Lp_res[1])

    return Lp_res

# def KS_binary(Xr: np.array, y: np.array) -> dict:
#     # Kolmogorov - Smirnov distance  - binary label case
#     x_a = positive_label_index[facet_index == 0]
#     x_d = positive_label_index[facet_index == 1]
#
#     Pa, ya = prob_pdf(x_a)  # x: raw values of the variable (column of data)
#     Pd, yd = prob_pdf(x_d)
#     max_distance = np.max(np.abs(Pa - Pd))
#     return {1: max_distance}
#
#
# def KS_multicategory(Xr: np.array, y: np.array, q=2) -> dict:
#     # Kolmogorov - Smirnov distance  - multicategory label case
#     y_categories = np.unique(y)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         res[y_true] = KS_binary(Xr, y_tmp)[1]
#     return res
#
# #####################################################################################################################
# def KS_continuous(Xr: np.array, y: np.array, q=2) -> dict:
#     # Kolmogorov - Smirnov distance  - continuous label case
#     y1 = y[Xr == 0]
#     y2 = y[Xr == 1]
#     p_a, p_b = KL_from_rawdata(y1, y2)[3:5]  # raw values of the variable (column of data)
#     max_distance = np.max([0, 0] if len(np.abs(p_a - p_b)) < 1 else np.abs(p_a - p_b))
#     return {'y': max_distance}
# #####################################################################################################################


def KS(x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series) -> dict:
    # Kolmogorov - Smirnov distance
    x_a = positive_label_index[facet_index == 0]
    x_d = positive_label_index[facet_index == 1]

    Pa, ya = prob_pdf(x_a)  # x: raw values of the variable (column of data)
    Pd, yd = prob_pdf(x_d)
    max_distance = np.max(np.abs(Pa - Pd))

    return {1: max_distance}

def CDD(x: pd.Series, facet_index: pd.Series, positive_label_index: pd.Series, strata: pd.Series) -> dict:
    all_strata = np.unique(strata)

    # Global demographic disparity (DD)
    numA = len(positive_label_index[(positive_label_index == 1) & (facet_index == 1)])
    denomA = len(facet_index[positive_label_index == 1])
    A = numA / denomA
    numD = len(positive_label_index[(positive_label_index == 0) & (facet_index == 1)])
    denomD = len(positive_label_index[positive_label_index == 0])
    D = numD / denomD
    DD = D - A
    # Conditional demographic disparity (CDD)
    CDD = []
    counts = []
    for subgroup in strata:
        counts = np.append(counts, len(strata[strata == subgroup]))
        numA = len(positive_label_index[(positive_label_index == 1) & (facet_index == 1) & (strata == subgroup)])
        denomA = len(facet_index[(positive_label_index == 1) & (strata == subgroup)])
        A = numA / denomA if denomA != 0 else 0
        numD = len(positive_label_index[(positive_label_index == 0) & (facet_index == 1) & (strata == subgroup)])
        denomD = len(facet_index[(positive_label_index == 0) & (strata == subgroup)])
        D = numD / denomD if denomD != 0 else 0
        CDD = np.append(CDD, D - A)
    wtd_mean_CDD = np.sum(counts * CDD) / np.sum(counts)
    #return {1: [DD, CDD, counts, wtd_mean_CDD]}
    return {1 : wtd_mean_CDD}

############################################ Post Training Metrics ###############################################

# def DPPL_binary(Xr: np.array, yhat: np.array) -> dict:
#     # Difference in proportions of predicted labels (DPPL) - binary case
#     na1hat = len(yhat[(yhat == 1) & (Xr == 0)])
#     na = len(Xr[Xr == 0])
#     qa = na1hat / na
#     nd1hat = len(yhat[(yhat == 1) & (Xr == 1)])
#     nd = len(Xr[Xr == 1])
#     qd = nd1hat / nd
#     # return (qa - qd), (qa - qd) / (qa + qd)
#     return {1: [(qa - qd), (qa - qd) / (qa + qd)]}
#
#
# def DPPL_multicategory(Xr: np.array, yhat: np.array) -> dict:
#     # Difference in proportions of predicted labels (DPPL) - multicategory case
#     y_categories = np.unique(yhat)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(yhat, [y_true])
#         res[y_true] = DPPL_binary(Xr, y_tmp)[1]
#     return res
#
#
# def DPPL_continuous(Xr: np.array, yhat: np.array) -> dict:
#     # Difference in proportions of predicted labels (DPPL) - continuous case
#     meana = np.mean(yhat[Xr == 0])
#     meand = np.mean(yhat[Xr == 1])
#     # return (meana - meand), (meana - meand) / (meana + meand)
#     return {1: (meana - meand) / (meana + meand)}


def DPPL(x: pd.Series, facet_index: pd.Series, predicted_labels: pd.Series) -> dict:
    # Difference in proportions of predicted labels (DPPL)
    na1hat = len(predicted_labels[(predicted_labels == 1) & (facet_index == 0)])
    na = len(facet_index[facet_index == 0])
    qa = na1hat / na
    nd1hat = len(predicted_labels[(predicted_labels == 1) & (facet_index == 1)])
    nd = len(facet_index[facet_index == 1])
    qd = nd1hat / nd
    # return (qa - qd), (qa - qd) / (qa + qd)
    return {1: qa - qd}

# def DI_binary(Xr: np.array, yhat: np.array) -> dict:
#     # Disparate impact - binary case
#     na1hat = len(yhat[(yhat == 1) & (Xr == 0)])
#     na = len(Xr[Xr == 0])
#     qa = na1hat / na
#     nd1hat = len(yhat[(yhat == 1) & (Xr == 1)])
#     nd = len(Xr[Xr == 1])
#     qd = nd1hat / nd
#
#     if qa != 0:
#         return {1: qd / qa}
#     return {1 : 1e10} # TODO : handle the infinity value
#
#
# def DI_multicategory(Xr: np.array, yhat: np.array) -> dict:
#     # Disparate impact - multicategory case
#     y_categories = np.unique(yhat)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(yhat, [y_true])
#         res[y_true] = DI_binary(Xr, y_tmp)[1]
#     return res
#
#
# def DI_continuous(Xr: np.array, yhat: np.array) -> dict:
#     # Disparate impact - continuous case
#     meana = np.mean(yhat[Xr == 0])
#     meand = np.mean(yhat[Xr == 1])
#     return {1: meand / meana}


def DI(x: pd.Series, facet_index: pd.Series, predicted_labels: pd.Series) -> dict:
    # Disparate impact

    na1hat = len(predicted_labels[(predicted_labels == 1) & (facet_index == 0)])
    na = len(facet_index[facet_index == 0])
    qa = na1hat / na
    nd1hat = len(predicted_labels[(predicted_labels == 1) & (facet_index == 1)])
    nd = len(facet_index[facet_index == 1])
    qd = nd1hat / nd

    if qa != 0:
        return {1: qd / qa}
    return {1 : 1e10} # TODO : handle the infinity value


# def DCO_multicategory(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Difference in Conditional Outcomes (DCO) - multicategory case
#     y_categories = np.unique(yhat)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         y_hat_tmp = collapse_to_binary(yhat, [y_true])
#         res[y_true] = DCO_binary(Xr, y_tmp, y_hat_tmp)[1]
#     return res
#
#
# def DCO_continuous(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Disparate impact - continuous case
#     meana = np.mean(y[Xr == 0]) / np.mean(yhat[Xr == 0])
#     meand = np.mean(y[Xr == 1]) / np.mean(yhat[Xr == 1])
#     return {1: meana - meand}


def DCO(x: pd.Series, facet_index: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> dict:
    # Difference in Conditional Outcomes (DCO)
    na1 = len(labels[(labels == 1) & (facet_index == 0)])
    na1hat = len(predicted_labels[(predicted_labels == 1) & (facet_index == 0)])
    nd1 = len(labels[(labels == 1) & (facet_index == 1)])
    nd1hat = len(predicted_labels[(predicted_labels == 1) & (facet_index == 1)])

    if na1hat != 0:
        ca = na1 / na1hat
    else:
        ca = 1e10  # TODO : handle the infinity value

    if nd1hat != 0:
        cd = nd1 / nd1hat
    else:
        cd = 1e10 # TODO : handle the infinity value

    # DCR
    TN_a = len(labels[(labels == 0) & (predicted_labels == 0) & (facet_index == 0)])
    na0hat = len(predicted_labels[(predicted_labels == 0) & (facet_index == 0)])
    TN_d = len(labels[(labels == 0) & (predicted_labels == 0) & (facet_index == 1)])
    nd0hat = len(predicted_labels[(predicted_labels == 0) & (facet_index == 1)])

    if na0hat != 0:
        rr_a = TN_a / na0hat
    else:
        rr_a = 1e10 # TODO : handle the infinity value

    if nd0hat != 0:
        rr_d = TN_d / nd0hat
    else:
        rr_d = 1e10 # TODO : handle the infinity value

    return {1: [ca - cd, (rr_a - rr_d), (ca - cd) / (ca + cd), (rr_a - rr_d) / (rr_a + rr_d)]}

    # if pred_type[0] == 0:
    #     return DCO_binary(Xr, y, yhat)
    # elif pred_type[0] == 1:
    #     return DCO_multicategory(Xr, y, yhat)
    # else:  # pred_type[0] == 2:
    #     return DCO_continuous(Xr, y, yhat)


# def RD_binary(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Recall difference - binary case
#     TP_a = len(y[(y == 1) & (yhat == 1) & (Xr == 0)])
#     FN_a = len(y[(y == 1) & (yhat == 0) & (Xr == 0)])
#     rec_a = TP_a / (TP_a + FN_a)
#     TP_d = len(y[(y == 1) & (yhat == 1) & (Xr == 1)])
#     FN_d = len(y[(y == 1) & (yhat == 0) & (Xr == 1)])
#     rec_d = TP_d / (TP_d + FN_d)
#     # return (rec_a - rec_d), (rec_a - rec_d) / (rec_a + rec_d)
#     return {1: (rec_a - rec_d) / (rec_a + rec_d)}
#
#
# def RD_multicategory(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Recall difference - multicategory case
#     y_categories = np.unique(yhat)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         y_hat_tmp = collapse_to_binary(yhat, [y_true])
#         res[y_true] = RD_binary(Xr, y_tmp, y_hat_tmp)[1]
#     return res
#
#
# def RD_continuous(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Disparate impact - continuous case
#     # Not def
#     return {'y': None}


def RD(x: pd.Series, facet_index: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> dict:
    # Recall difference

    TP_a = len(labels[(labels == 1) & (predicted_labels == 1) & (facet_index == 0)])
    FN_a = len(labels[(labels == 1) & (predicted_labels == 0) & (facet_index == 0)])
    rec_a = TP_a / (TP_a + FN_a) if TP_a + FN_a != 0 else 0
    TP_d = len(labels[(labels == 1) & (predicted_labels == 1) & (facet_index == 1)])
    FN_d = len(labels[(labels == 1) & (predicted_labels == 0) & (facet_index == 1)])
    rec_d = TP_d / (TP_d + FN_d) if TP_d + FN_d != 0 else 0
    return {1 : [(rec_a - rec_d), (rec_a - rec_d) / (rec_a + rec_d)]}

    # return {1: (rec_a - rec_d) / (rec_a + rec_d)}
    #
    #
    # if pred_type[0] == 0:
    #     return RD_binary(Xr, y, yhat)
    # elif pred_type[0] == 1:
    #     return RD_multicategory(Xr, y, yhat)
    # else:  # pred_type[0] == 2:
    #     return RD_continuous(Xr, y, yhat)

# def DLR_binary(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Difference in rates - binary case
#
#     # DAR
#     TP_a = len(y[(y == 1) & (yhat == 1) & (Xr == 0)])
#     na1hat = len(yhat[(yhat == 1) & (Xr == 0)])
#     TP_d = len(y[(y == 1) & (yhat == 1) & (Xr == 1)])
#     nd1hat = len(yhat[(yhat == 1) & (Xr == 1)])
#
#     if na1hat != 0:
#         ar_a = TP_a / na1hat
#     else:
#         ar_a = 1e10 # TODO : handle the infinity value
#
#     if nd1hat != 0:
#         ar_d = TP_d / nd1hat
#     else:
#         ar_d = 1e10 # TODO : handle the infinity value
#
#     # DRR
#     TN_a = len(y[(y == 0) & (yhat == 0) & (Xr == 0)])
#     na0hat = len(yhat[(yhat == 0) & (Xr == 0)])
#     TN_d = len(y[(y == 0) & (yhat == 0) & (Xr == 1)])
#     nd0hat = len(yhat[(yhat == 0) & (Xr == 1)])
#
#     if na0hat != 0:
#         rr_a = TN_a / na0hat
#     else:
#         rr_a = 1e10 # TODO : handle the infinity value
#
#     if nd0hat != 0:
#         rr_d = TN_d / nd0hat
#     else:
#         rr_d = 1e10 # TODO : handle the infinity value
#
#     # return (ar_a - ar_d), (ar_a - ar_d) / (ar_a + ar_d), (rr_a - rr_d), (rr_a - rr_d) / (rr_a + rr_d)
#     return {1: [(ar_a - ar_d), (rr_a - rr_d), (ar_a - ar_d) / (ar_a + ar_d), (rr_a - rr_d) / (rr_a + rr_d)]}
#
#
# def DLR_multicategory(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Difference in rates - multicategory case
#     y_categories = np.unique(yhat)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         y_hat_tmp = collapse_to_binary(yhat, [y_true])
#         res[y_true] = DLR_binary(Xr, y_tmp, y_hat_tmp)[1]
#     return res
#
#
# def DLR_continuous(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Difference in rates - continuous case
#     meana = np.mean(yhat[Xr == 0]) / np.mean(y[Xr == 0])
#     meand = np.mean(yhat[Xr == 1]) / np.mean(y[Xr == 1])
#     return {1: meana - meand}


def DLR(x: pd.Series, facet_index: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> dict:
    # Difference in rates

    TP_a = len(labels[(labels == 1) & (predicted_labels == 1) & (facet_index == 0)])
    na1hat = len(predicted_labels[(predicted_labels == 1) & (facet_index == 0)])
    TP_d = len(labels[(labels == 1) & (predicted_labels == 1) & (facet_index == 1)])
    nd1hat = len(predicted_labels[(predicted_labels == 1) & (facet_index == 1)])

    if na1hat != 0:
        ar_a = TP_a / na1hat
    else:
        ar_a = 1e10 # TODO : handle the infinity value

    if nd1hat != 0:
        ar_d = TP_d / nd1hat
    else:
        ar_d = 1e10 # TODO : handle the infinity value

    # DRR
    TN_a = len(labels[(labels == 0) & (predicted_labels == 0) & (facet_index == 0)])
    na0hat = len(predicted_labels[(predicted_labels == 0) & (facet_index == 0)])
    TN_d = len(labels[(labels == 0) & (predicted_labels == 0) & (facet_index == 1)])
    nd0hat = len(predicted_labels[(predicted_labels == 0) & (facet_index == 1)])

    if na0hat != 0:
        rr_a = TN_a / na0hat
    else:
        rr_a = 1e10 # TODO : handle the infinity value

    if nd0hat != 0:
        rr_d = TN_d / nd0hat
    else:
        rr_d = 1e10 # TODO : handle the infinity value

    # return (ar_a - ar_d), (ar_a - ar_d) / (ar_a + ar_d), (rr_a - rr_d), (rr_a - rr_d) / (rr_a + rr_d)
    return {1: [(ar_a - ar_d), (rr_a - rr_d), (ar_a - ar_d) / (ar_a + ar_d), (rr_a - rr_d) / (rr_a + rr_d)]}


# def PD_binary(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     tmp = DLR_binary(Xr, y, yhat)[1]
#     return {1: [tmp[0], tmp[2]]}
#
#
# def PD_multicategory(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # PD - multicategory case
#     y_categories = np.unique(yhat)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         y_hat_tmp = collapse_to_binary(yhat, [y_true])
#         res[y_true] = PD_binary(Xr, y_tmp, y_hat_tmp)[1]
#     return res
#
#
# def PD_continuous(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # PD - continuous case
#     # Not def
#     return {'y': None}


def PD(x: pd.Series, facet_index: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> dict:
    # PD

    tmp = DLR(x, facet_index, labels, predicted_labels)[1]
    return {1: [tmp[0], tmp[2]]}



# def AD_binary(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Accuracy Difference (AD) - binary case
#     TP_a = len(y[(y == 1) & (yhat == 1) & (Xr == 0)])
#     FP_a = len(y[(y == 0) & (yhat == 1) & (Xr == 0)])
#     FN_a = len(y[(y == 1) & (yhat == 0) & (Xr == 0)])
#     TN_a = len(y[(y == 0) & (yhat == 0) & (Xr == 0)])
#     acc_a = (TP_a + TN_a) / (TP_a + TN_a + FP_a + FN_a)
#     TP_d = len(y[(y == 1) & (yhat == 1) & (Xr == 1)])
#     FP_d = len(y[(y == 0) & (yhat == 1) & (Xr == 1)])
#     FN_d = len(y[(y == 1) & (yhat == 0) & (Xr == 1)])
#     TN_d = len(y[(y == 0) & (yhat == 0) & (Xr == 1)])
#     acc_d = (TP_d + TN_d) / (TP_d + TN_d + FP_d + FN_d)
#     # return (acc_a - acc_d), (acc_a - acc_d) / (acc_a + acc_d)
#     return {1: (acc_a - acc_d) / (acc_a + acc_d)}
#
#
# def AD_multicategory(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Accuracy Difference (AD) - multicategory case
#     y_categories = np.unique(yhat)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         y_hat_tmp = collapse_to_binary(yhat, [y_true])
#         res[y_true] = AD_binary(Xr, y_tmp, y_hat_tmp)[1]
#     return res
#
#
# def AD_continuous(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Accuracy Difference (AD)  - continuous case
#     perm = np.argsort(y)
#     y = y[perm]
#     yhat = yhat[perm]
#     Xr = Xr[perm]
#     ad = np.linalg.norm(y[Xr == 1] - yhat[Xr == 1]) - np.linalg.norm(y[Xr == 0] - yhat[Xr == 0])
#     return {'y': ad}


def AD(x: pd.Series, facet_index: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> dict:
    # Difference in rates

    TP_a = len(labels[(labels == 1) & (predicted_labels == 1) & (facet_index == 0)])
    FP_a = len(labels[(labels == 0) & (predicted_labels == 1) & (facet_index == 0)])
    FN_a = len(labels[(labels == 1) & (predicted_labels == 0) & (facet_index == 0)])
    TN_a = len(labels[(labels == 0) & (predicted_labels == 0) & (facet_index == 0)])
    acc_a = (TP_a + TN_a) / (TP_a + TN_a + FP_a + FN_a)
    TP_d = len(labels[(labels == 1) & (predicted_labels == 1) & (facet_index == 1)])
    FP_d = len(labels[(labels == 0) & (predicted_labels == 1) & (facet_index == 1)])
    FN_d = len(labels[(labels == 1) & (predicted_labels == 0) & (facet_index == 1)])
    TN_d = len(labels[(labels == 0) & (predicted_labels == 0) & (facet_index == 1)])
    acc_d = (TP_d + TN_d) / (TP_d + TN_d + FP_d + FN_d)
    # return [(acc_a - acc_d), (acc_a - acc_d) / (acc_a + acc_d)]
    return {1: [(acc_a - acc_d), (acc_a - acc_d) / (acc_a + acc_d)]}


    # if pred_type[0] == 0:
    #     return AD_binary(Xr, y, yhat)
    # elif pred_type[0] == 1:
    #     return AD_multicategory(Xr, y, yhat)
    # else:  # pred_type[0] == 2:
    #     return AD_continuous(Xr, y, yhat)


# def TE_binary(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Treatment Equality - binary case
#     FP_a = len(y[(y == 0) & (yhat == 1) & (Xr == 0)])
#     FN_a = len(y[(y == 1) & (yhat == 0) & (Xr == 0)])
#     FP_d = len(y[(y == 0) & (yhat == 1) & (Xr == 1)])
#     FN_d = len(y[(y == 1) & (yhat == 0) & (Xr == 1)])
#     if FP_a != 0:
#         tau_a = FN_a / FP_a
#         if FP_d != 0:
#             tau_d = FN_d / FP_d
#         else:
#             return {1: -1.0}
#     else:
#         if FP_d == 0:
#             return {1: 0.0}
#         else:
#             return {1: 1.0}
#     # return (tau_d - tau_a), (tau_d - tau_a) / (tau_a + tau_d)
#     return {1: (tau_d - tau_a) / (tau_a + tau_d)}
#
#
# def TE_multicategory(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Treatment Equality - multicategory case
#     y_categories = np.unique(yhat)
#     res = dict()
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         y_hat_tmp = collapse_to_binary(yhat, [y_true])
#         res[y_true] = TE_binary(Xr, y_tmp, y_hat_tmp)[1]
#     return res
#
#
# def TE_continuous(Xr: np.array, y: np.array, yhat: np.array) -> dict:
#     # Treatment Equality   - continuous case
#     # Not def
#     return {'y': None}


def TE(x: pd.Series, facet_index: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> dict:
    # Treatment Equality
    FP_a = len(labels[(labels == 0) & (predicted_labels == 1) & (facet_index == 0)])
    FN_a = len(labels[(labels == 1) & (predicted_labels == 0) & (facet_index == 0)])
    FP_d = len(labels[(labels == 0) & (predicted_labels == 1) & (facet_index == 1)])
    FN_d = len(labels[(labels == 1) & (predicted_labels == 0) & (facet_index == 1)])
    if FP_a != 0:
        tau_a = FN_a / FP_a
        if FP_d != 0:
            tau_d = FN_d / FP_d
        else:
            return {1: -1.0}
    else:
        if FP_d == 0:
            return {1: 0.0}
        else:
            return {1: 1.0}
    # return [(tau_d - tau_a), (tau_d - tau_a) / (tau_a + tau_d)]
    return {1: [(tau_d - tau_a), (tau_d - tau_a) / (tau_a + tau_d)]}


def FlipSet_pos(X: np.array, y: np.array, yG: np.array) -> np.array:
    return np.array([X[i] for i in range(len(X)) if y[i] > yG[i]])


def FlipSet_neg(X: np.array, y: np.array, yG: np.array) -> np.array:
    return np.array([X[i] for i in range(len(X)) if y[i] < yG[i]])


def FlipSet(X: np.array, y: np.array, yG: np.array) -> np.array:
    return np.array([X[i] for i in range(len(X)) if y[i] != yG[i]])


def FT_binary(X: np.ndarray, Xr: np.array, y: np.array, yhat: np.array, verbose=0) -> dict:
    # FlipTest - binary case
    # a = adv facet, d = disadv facet
    data_a = ([el for idx, el in enumerate(X) if Xr[idx] == 0],
              [el for idx, el in enumerate(yhat) if Xr[idx] == 0],
              [el for idx, el in enumerate(Xr) if Xr[idx] == 0])
    data_d = ([el for idx, el in enumerate(X) if Xr[idx] == 1],
              [el for idx, el in enumerate(yhat) if Xr[idx] == 1],
              [el for idx, el in enumerate(Xr) if Xr[idx] == 1])

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                               metric='minkowski', metric_params=None, n_jobs=None)

    # kNN method over a with Labels from the model
    knn.fit(np.array(data_a[0]), np.array(data_a[1]))
    # kNN prediction over d
    d_y_if_a = knn.predict(data_d[0])
    # Model predictions over the same test d
    d_y_model = data_d[1]

    FS_pos = FlipSet_pos(X=data_d[1], y=d_y_model, yG=d_y_if_a)
    FS_neg = FlipSet_neg(X=data_d[1], y=d_y_model, yG=d_y_if_a)
    FS = FlipSet(X=data_d[1], y=d_y_model, yG=d_y_if_a)

    if verbose > 0:
        print('Data with', len(X), 'examples -- ', len(data_d[0]), 'female examples')
        print('Length of FlipSet positive (i.e. positive bias to females w.r.t. males):', len(FS_pos), '(',
              100 * len(FS_pos) / len(data_d[0]), '%)')
        print('Length of FlipSet negative (i.e. negative bias to females w.r.t. males):', len(FS_neg), '(',
              100 * len(FS_neg) / len(data_d[0]), '%)')
        print('Length of FlipSet:', len(FS), '(', 100 * len(FS) / len(data_d[0]), '%)')

    FTd = (len(FS_pos) - len(FS_neg)) / len(data_d[0])
    FTs = len(FS) / len(data_d[0])

    return {1: [FTd, FTs]}


def FT_multicategory(X: np.ndarray, Xr: np.array, y: np.array, yhat: np.array) -> dict:
    # FlipTest - multicategory case
    y_categories = set(yhat)
    res = {}
    for y_true in y_categories:
        y_tmp = collapse_to_binary(y, [y_true])
        y_hat_tmp = collapse_to_binary(yhat, [y_true])
        res[y_true] = FT_binary(X, Xr, y_tmp, y_hat_tmp)[1]
    return res


def FT_continuous(X: np.ndarray, Xr: np.array, y: np.array, yhat: np.array) -> dict:
    # FlipTest - continuous case
    # FlipTest - binary case
    # a = adv facet, d = disadv facet
    data_a = ([el for idx, el in enumerate(X) if Xr[idx] == 0],
              [el for idx, el in enumerate(yhat) if Xr[idx] == 0],
              [el for idx, el in enumerate(Xr) if Xr[idx] == 0])
    data_d = ([el for idx, el in enumerate(X) if Xr[idx] == 1],
              [el for idx, el in enumerate(yhat) if Xr[idx] == 1],
              [el for idx, el in enumerate(Xr) if Xr[idx] == 1])

    knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                              metric='minkowski', metric_params=None, n_jobs=None)

    # kNN method over a with Labels from the model
    knn.fit(np.array(data_a[0]), np.array(data_a[1]))
    # kNN prediction over d, first finding the index of the k neighbors for each example over d
    kneigh = knn.kneighbors(data_d[0])[1]
    # print(kneigh)
    # Finding the average value of the label of the k neighbors for each example
    data1_a_array = np.array(data_a[1])
    data1_a_y_model_average = np.mean(data1_a_array[kneigh], axis=1)
    # print(tmp)
    d_y_if_a = data1_a_y_model_average
    # Model predictions over the same test d
    d_y_model = data_d[1]
    # print(data_d[1], d_y_model, d_y_if_a)

    FS_pos = FlipSet_pos(X=data_d[1], y=d_y_model, yG=d_y_if_a)
    FS_neg = FlipSet_neg(X=data_d[1], y=d_y_model, yG=d_y_if_a)
    FS = FlipSet(X=data_d[1], y=d_y_model, yG=d_y_if_a)

    FTd = (len(FS_pos) - len(FS_neg)) / len(data_d[0])
    FTs = len(FS) / len(data_d[0])

    return {'y': [FTd, FTs]}


def FT(X: np.ndarray, Xr: np.array, y: np.array, yhat: np.array, pred_type: tuple) -> dict:
    # FlipTest
    if pred_type[0] == 0:
        return FT_binary(X, Xr, y, yhat)
    elif pred_type[0] == 1:
        return FT_multicategory(X, Xr, y, yhat)
    else:
        return FT_continuous(X, Xr, y, yhat)

if __name__ == "__main__":
    for metric in [class_imbalance, diff_positive_labels, kl_divergence, JS, LPnorm, TVD, KS, CDD]:
        print('\n\n' + metric.__name__)
        metric_name = metric.__name__
        print('Binary facet, Binary Label')
        x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F'])

        facet_index = x == 'F'

        positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])

        strata = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2])

        if metric_name != 'CDD':
            print(metric(x, facet_index, positive_label_index))
        else:
            print(metric(x, facet_index, positive_label_index, strata))

        print('Continuous facet, Binary Label')
        x = pd.Series(np.random.uniform(0, 2, 12))

        facet_index = x > 1.0

        positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])

        strata = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2])

        if metric_name != 'CDD':
            print(metric(x, facet_index, positive_label_index))
        else:
            print(metric(x, facet_index, positive_label_index, strata))

        print('Multicategory facet, Binary Label')
        x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'O', 'O', 'O', 'O'])

        positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1])

        strata = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2, 0, 1, 1, 2])

        if metric_name != 'CDD':
            print(metric_one_vs_all(metric, x, x, positive_label_index=positive_label_index))
        else:
            print(metric_one_vs_all(metric, x, x, positive_label_index=positive_label_index, strata=strata))

        print('Binary facet, Multicategory Label')

        x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F'])

        facet_index = x == 'F'

        labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1])

        strata = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2])

        if metric_name != 'CDD':

            print(metric_one_vs_all(metric, x, x, labels=labels))

        else:

            print(metric_one_vs_all(metric, x, x, labels=labels, strata=strata))

        print('Continuous facet, Multicategory Label')

        x = pd.Series(np.random.uniform(0, 2, 12))

        facet_index = x > 1.0

        labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1])

        strata = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2])

        if metric_name != 'CDD':
            print(metric_one_vs_all(metric, x, x, labels=labels))
        else:
            print(metric_one_vs_all(metric, x, x, labels=labels, strata=strata))

    for metric in [DPPL, DI, DCO, RD, DLR, PD, AD, TE]:
        print('\n\n' + metric.__name__)
        metric_name = metric.__name__
        print('Binary facet, Binary Label')
        x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F'])

        facet_index = x == 'F'

        labels = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
        predicted_labels = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])

        if metric in [DPPL, DI]:
            print(metric(x, facet_index, predicted_labels))
        else:
            print(metric(x, facet_index, labels, predicted_labels))

        print('Continuous facet, Binary Label')
        x = pd.Series(np.random.uniform(0, 2, 12))

        facet_index = x > 1.0

        labels = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
        predicted_labels = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])


        if metric in [DPPL, DI]:
            print(metric(x, facet_index, predicted_labels))
        else:
            print(metric(x, facet_index, labels, predicted_labels))

        print('Multicategory facet, Binary Label')
        x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'O', 'O', 'O', 'O'])

        # facet_index = x == 'F'

        labels = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        predicted_labels = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1])


        #TODO: Need to change to metric_one_vs_all
        if metric in [DPPL, DI]:
            print(metric_one_vs_all(metric, x, facet_index, predicted_labels = predicted_labels))
        else:
            print(metric_one_vs_all(metric, x, facet_index, labels = labels, predicted_labels=predicted_labels))

        print('Binary facet, Multicategory Label')

        x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F'])

        facet_index = x == 'F'

        labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1])

        predicted_labels = pd.Series([0, 2, 1, 2, 1, 0, 1, 1, 2, 2, 1, 0])

        if metric in [DPPL, DI]:
            print(metric_one_vs_all(metric, x, facet_index, predicted_labels = predicted_labels))
        else:
            print(metric_one_vs_all(metric, x, facet_index, labels = labels, predicted_labels=predicted_labels))

        print('Continuous facet, Multicategory Label')

        x = pd.Series(np.random.uniform(0, 2, 12))

        facet_index = x > 1.0

        labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1])

        predicted_labels = pd.Series([0, 2, 1, 2, 1, 0, 1, 1, 2, 2, 1, 0])

        if metric in [DPPL, DI]:
            print(metric_one_vs_all(metric, x, facet_index, predicted_labels = predicted_labels))
        else:
            print(metric_one_vs_all(metric, x, facet_index, labels = labels, predicted_labels=predicted_labels))


    X = np.array([[0, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],
                  [1, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1]])

    # print('\n\nFT - cont.')
    # print(FT(X, Xr, y, yhat, pred_type=prediction_type[2]))
