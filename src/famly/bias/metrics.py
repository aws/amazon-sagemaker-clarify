import logging
from typing import Dict

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

log = logging.getLogger(__name__)

pretraining_metrics = ['CI', 'DPL', 'KL', 'JS', 'LPnorm', 'TVD', 'KS', 'CDD']
posttraining_metrics = ['DPPL', 'DI', 'DCA', 'DCR', 'RD', 'DRR', 'PD', 'AD', 'TE']

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

def GaussianFilter(input_array: np.array, sigma: int=1) -> np.array:
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
    return np.convolve(input_array, gauss_filter, 'same')

def PDF(x: np.array) -> (np.array, np.array):
    """
    :param x: input array
    :return: probability distribution of the input array
    """
    y = np.unique(x)

    p, bins_edges = np.histogram(x, range=(0, 1))

    p = p / np.sum(p)
    return p

#Methods to handle multicategory cases

def metric_one_vs_all(metric, x: pd.Series, facet: pd.Series, positive_label_index: pd.Series=None, predicted_labels: pd.Series=None, labels: pd.Series=None, group_variable: pd.Series=None, dataset: pd.DataFrame=None) -> Dict:
    """
    Calculate any metric for a categorical facet and/or label using 1 vs all
    :param metric: a function defined in this file which computes a metric
    :param x: pandas series containing categorical values
    :param facet: facet containing multicategory values for each element in x
    :param positive_label_index: series of boolean values indicating positive target labels (optional)
    :param predicted_labels: series of model predictions of target column (optional)
    :param labels: series of true labels (optional)
    :param group_variable: series indicating strata each point belongs to (used for CDD metric) (optional)
    :param dataset: full dataset (used only for FlipTest metric) (optional)
    :return: A dictionary in which each key is one of the sensitive attributes in the facet column, and each value is its corresponding metric according to the requested bias measure
    """
    #Ensure correct parameter types
    x = pd.Series(x)
    facet = pd.Series(facet)
    if not positive_label_index is None:
        positive_label_index = pd.Series(positive_label_index)
    if not predicted_labels is None:
        predicted_labels = pd.Series(predicted_labels)
    if not labels is None:
        labels = pd.Series(labels)
    if not group_variable is None:
        group_variable = pd.Series(group_variable)
    if dataset:
        dataset = pd.DataFrame(dataset)

    categories = facet.unique()
    res = {}
    for cat in categories:
        if labels is None or len(np.unique(labels)) <= 2:
            if metric.__name__ in pretraining_metrics:
                if metric != CDD:
                    res[cat] = metric(x, facet == cat, positive_label_index)
                else:
                    res[cat] = metric(x, facet == cat, positive_label_index, group_variable)
            else:
                if metric == FT:
                    res[cat] = metric(dataset, facet == cat, labels, predicted_labels)
                else:
                    res[cat] = metric(x, facet == cat, labels, predicted_labels)
        else:
            res[cat] = label_one_vs_all(metric, x, facet == cat, predicted_labels=predicted_labels, labels=labels, group_variable=group_variable)

    return res

def label_one_vs_all(metric, x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series=None, group_variable: pd.Series=None) -> Dict:
    """
    :param metric: one of the bias measures defined in this file
    :param x: data from the feature of interest
    :param facet: boolean column with true values indicate a sensitive value
    :param predicted_labels: predictions for labels made by model
    :param labels: True values of the target column
    :param group_variable: column of values indicating the subgroup each data point belongs to (used for calculating CDD metric only)
    :return:
    """

    values = {}
    label_unique = np.unique(labels)

    for label in label_unique:
        if metric.__name__ in pretraining_metrics:
            if metric != CDD:
                values[label] = metric(x, facet, labels == label)
            else:
                values[label] = metric(x, facet, labels == label, group_variable)
        else:
            values[label] = metric(x, facet, labels == label, predicted_labels == label)

    return values

######################################### Pre-training Bias Measures ###################################################

def CI(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    Class imbalance (CI)
    :param x: pandas series
    :param facet: boolean index series selecting protected instances
    :param positive_label_index: series of boolean values indicating positive target labels
    :return: a float in the interval [-1, +1] indicating an under-representation or over-representation
    of the protected class.

    Bias is often generated from an under-representation of
    the protected class in the dataset, especially if the desired “golden truth”
    is equality across classes. Imbalance carries over into model predictions.
    We will report all measures in differences and normalized differences. Since
    the measures are often probabilities or proportions, the differences will lie in
    We define CI = (np − p)/(np + p). Where np is the number of instances in the not protected group_variable
    and p is number of instances in the protected group_variable.
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)


    pos = len(x[facet])
    neg = len(x[~facet])
    q = pos + neg

    if neg == 0:
        raise ValueError("CI: negated facet set is empty. Check that x[~facet] has non-zero length.")
    if pos == 0:
        raise ValueError("CI: facet set is empty. Check that x[facet] has non-zero length.")

    assert q != 0

    ci = float(neg - pos) / q
    return ci


def DPL(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    Difference in positive proportions in predicted labels
    :param x: pandas series of the target column
    :param facet: boolean series indicating protected class
    :param label: pandas series of labels (binary, multicategory, or continuous)
    :param positive_label_index: consider this label value as the positive value, default is 1.
    :return: a float in the interval [-1, +1] indicating bias in the labels.
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)


    positive_label_index_neg_facet = (positive_label_index) & ~facet
    positive_label_index_facet = (positive_label_index) & facet

    np = len(x[~facet])
    p = len(x[facet])

    n_pos_label_neg_facet = len(x[positive_label_index_neg_facet])
    n_pos_label_facet = len(x[positive_label_index_facet])


    if np == 0:
        raise ValueError("DPL: negative facet set is empty.")
    if p == 0:
        raise ValueError("DPL: facet set is empty.")
    q_neg = n_pos_label_neg_facet / np
    q_pos = n_pos_label_facet / p
    if (q_neg + q_pos) == 0:
        raise ValueError("DPL: label facet is empty.")
    dpl = (q_neg - q_pos) / (q_neg + q_pos)

    return dpl

def KL(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param positive_label_index: boolean column indicating positive labels
    :return: Kullback and Leibler (KL) divergence metric
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    facet = np.array(facet)
    x_a = positive_label_index[~facet]
    x_d = positive_label_index[facet]
    Pa = PDF(x_a)  # x: raw values of the variable (column of data)
    Pd = PDF(x_d)

    if len(Pa) == len(Pd):
        kl = np.sum(Pa * np.log(Pa / Pd))  # note log is base e, measured in nats
    else:
        kl = -1.0
    return kl


def JS(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param positive_label_index: boolean column indicating positive labels
    :return: Jenson-Shannon (JS) divergence metric
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    x_a = positive_label_index[~facet]
    x_d = positive_label_index[facet]

    Pa = PDF(x_a)  # x: raw values of the variable (column of data)
    Pd = PDF(x_d)

    if len(Pa) == len(Pd):
        P = PDF(positive_label_index)
        js_divergence = 0.5 * (np.sum(Pa * np.log(Pa / P)) + np.sum(Pd * np.log(Pd / P)))
    else:
        js_divergence = -1.0

    return js_divergence

def LPnorm(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series, p: int=2) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param positive_label_index: boolean column indicating positive labels
    :param q: the order of norm desired
    :return: Lp-norm metric
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    x_a = positive_label_index[~facet]
    x_d = positive_label_index[facet]

    Pa = PDF(x_a)
    Pd = PDF(x_d)

    if len(Pa) == len(Pd):
        lp_norm = np.linalg.norm(Pa - Pd, p)
    else:
        lp_norm = -1.0

    return lp_norm


def TVD(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
   :param x: input feature
   :param facet: boolean column indicating sensitive values
   :param positive_label_index: boolean column indicating positive labels
   :return: 1/2 * L-1 norm
   """

    Lp_res = LPnorm(x, facet, positive_label_index, p=1)

    if Lp_res == -1.0:
        return -1.0

    tvd = 0.5 * Lp_res

    return tvd

def KS(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param positive_label_index: boolean column indicating positive labels
    :return: Kolmogorov-Smirnov metric
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    x_a = positive_label_index[~facet]
    x_d = positive_label_index[facet]

    Pa = PDF(x_a)  # x: raw values of the variable (column of data)
    Pd = PDF(x_d)
    max_distance = np.max(np.abs(Pa - Pd))

    return max_distance

def CDD(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series, group_variable: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param positive_label_index: boolean column indicating positive labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    all_group_variable = np.unique(group_variable)
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    # Global demographic disparity (DD)
    numA = len(positive_label_index[(positive_label_index) & (facet)])
    denomA = len(facet[positive_label_index])
    A = numA / denomA if denomA != 0 else 0
    numD = len(positive_label_index[(~positive_label_index) & (facet)])
    denomD = len(positive_label_index[~positive_label_index])
    D = numD / denomD if denomD != 0 else 0
    DD = D - A

    # Conditional demographic disparity (CDD)
    CDD = []
    counts = []
    for subgroup_variable in group_variable:
        counts = np.append(counts, len(group_variable[group_variable == subgroup_variable]))
        numA = len(positive_label_index[(positive_label_index) & (facet) & (group_variable == subgroup_variable)])
        denomA = len(facet[(positive_label_index) & (group_variable == subgroup_variable)])
        A = numA / denomA if denomA != 0 else 0
        numD = len(positive_label_index[(~positive_label_index) & (facet) & (group_variable == subgroup_variable)])
        denomD = len(facet[(~positive_label_index) & (group_variable == subgroup_variable)])
        D = numD / denomD if denomD != 0 else 0
        CDD = np.append(CDD, D - A)
    wtd_mean_CDD = np.sum(counts * CDD) / np.sum(counts)

    return wtd_mean_CDD

############################################ Post Training Metrics ###############################################


def DPPL(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param labels: boolean column indicating true values of target column
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Returns Difference in Positive Proportions, based on predictions rather than true labels
    """
    na1hat = len(predicted_labels[(predicted_labels) & (~facet)])
    na = len(facet[~facet])
    qa = na1hat / na if na != 0 else 0
    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])
    nd = len(facet[facet])
    qd = nd1hat / nd if nd != 0 else 0

    return qa - qd


def DI(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    # Disparate impact
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param labels: boolean column indicating true values of target column
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Returns disparate impact, the ratio between positive proportions, based on predicted labels
    """

    na1hat = len(predicted_labels[(predicted_labels) & (~facet)])
    na = len(facet[~facet])
    qa = na1hat / na if na != 0 else 0
    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])
    nd = len(facet[facet])
    qd = nd1hat / nd if nd != 0 else 0

    if qa != 0:
        return qd / qa
    return 1e10 # TODO : handle the infinity value

def DCA(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Difference in Conditional Acceptance between advantaged and disadvantaged classes
    """
    na1 = len(labels[(labels) & (~facet)])
    na1hat = len(predicted_labels[(predicted_labels) & (~facet)])
    nd1 = len(labels[(labels) & (facet)])
    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])

    if na1hat != 0:
        ca = na1 / na1hat
    else:
        ca = 1e10  # TODO : handle the infinity value

    if nd1hat != 0:
        cd = nd1 / nd1hat
    else:
        cd = 1e10 # TODO : handle the infinity value

    return (ca - cd)

def DCR(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Difference in Conditional Rejection between advantaged and disadvantaged classes
    """

    TN_a = len(labels[(~labels) & (~predicted_labels) & (~facet)])
    na0hat = len(predicted_labels[(~predicted_labels) & (~facet)])
    TN_d = len(labels[(~labels) & (~predicted_labels) & (facet)])
    nd0hat = len(predicted_labels[(~predicted_labels) & (facet)])

    if na0hat != 0:
        rr_a = TN_a / na0hat
    else:
        rr_a = 1e10 # TODO : handle the infinity value

    if nd0hat != 0:
        rr_d = TN_d / nd0hat
    else:
        rr_d = 1e10 # TODO : handle the infinity value

    return rr_a - rr_d

def RD(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Recall Difference between advantaged and disadvantaged classes
    """

    TP_a = len(labels[(labels) & (predicted_labels) & (~facet)])
    FN_a = len(labels[(labels) & (~predicted_labels) & (~facet)])
    rec_a = TP_a / (TP_a + FN_a) if TP_a + FN_a != 0 else 0

    TP_d = len(labels[(labels) & (predicted_labels) & (facet)])
    FN_d = len(labels[(labels) & (~predicted_labels) & (facet)])

    rec_d = TP_d / (TP_d + FN_d) if TP_d + FN_d != 0 else 0

    return rec_a - rec_d


def DRR(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Difference in Rejection Rates between advantaged and disadvantaged classes
    """

    TN_a = len(labels[(~labels) & (~predicted_labels) & (~facet)])
    na0hat = len(predicted_labels[(~predicted_labels) & (~facet)])
    TN_d = len(labels[(~labels) & (~predicted_labels) & (facet)])
    nd0hat = len(predicted_labels[(~predicted_labels) & (facet)])

    if na0hat != 0:
        rr_a = TN_a / na0hat
    else:
        rr_a = 1e10 # TODO : handle the infinity value

    if nd0hat != 0:
        rr_d = TN_d / nd0hat
    else:
        rr_d = 1e10 # TODO : handle the infinity value

    return (rr_a - rr_d)

def PD(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> dict:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Precision Difference (aka Difference in Acceptance Rates)
    """

    TP_a = len(labels[(labels) & (predicted_labels) & (~facet)])
    na1hat = len(predicted_labels[(predicted_labels) & (~facet)])
    TP_d = len(labels[(labels) & (predicted_labels) & (facet)])
    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])

    if na1hat != 0:
        ar_a = TP_a / na1hat
    else:
        ar_a = 1e10 # TODO : handle the infinity value

    if nd1hat != 0:
        ar_d = TP_d / nd1hat
    else:
        ar_d = 1e10 # TODO : handle the infinity value

    return ar_a - ar_d


def AD(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Accuracy Difference between advantaged and disadvantaged classes
    """

    TP_a = len(labels[(labels) & (predicted_labels) & (~facet)])
    FP_a = len(labels[(~labels) & (predicted_labels) & (~facet)])
    FN_a = len(labels[(labels) & (~predicted_labels) & (~facet)])
    TN_a = len(labels[(~labels) & (~predicted_labels) & (~facet)])
    acc_a = (TP_a + TN_a) / (TP_a + TN_a + FP_a + FN_a) if (TP_a + TN_a + FP_a + FN_a) != 0 else 0
    TP_d = len(labels[(labels) & (predicted_labels) & (facet)])
    FP_d = len(labels[(~labels) & (predicted_labels) & (facet)])
    FN_d = len(labels[(labels) & (~predicted_labels) & (facet)])
    TN_d = len(labels[(~labels) & (~predicted_labels) & (facet)])
    acc_d = (TP_d + TN_d) / (TP_d + TN_d + FP_d + FN_d) if (TP_d + TN_d + FP_d + FN_d) != 0 else 0

    return acc_a - acc_d

def TE(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive values
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Returns the difference in ratios between false negatives and false positives for the advantaged
    and disadvantaged classes
    """

    FP_a = len(labels[(~labels) & (predicted_labels) & (~facet)])
    FN_a = len(labels[(labels) & (~predicted_labels) & (~facet)])
    FP_d = len(labels[(~labels) & (predicted_labels) & (facet)])
    FN_d = len(labels[(labels) & (~predicted_labels) & (facet)])

    if FP_a != 0:
        tau_a = FN_a / FP_a
        if FP_d != 0:
            tau_d = FN_d / FP_d
        else:
            return -1.0
    else:
        if FP_d == 0:
            return 0.0
        else:
            return 1.0

    return tau_d - tau_a


def FlipSet_pos(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] > predicted_labels[i]])


def FlipSet_neg(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] < predicted_labels[i]])


def FlipSet(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] != predicted_labels[i]])


def FT(dataset: np.ndarray, facet: np.array, labels: np.array, predicted_labels: np.array, verbose=0) -> float:
    """
    :param dataset: array of data points
    :param facet: boolean column indicating sensitive vales
    :param labels: boolean column of positive values for target column
    :param predicted_labels: boolean column of predicted positive values for target column
    :param verbose: optional boolean value
    :return: FT difference metric
    """

    # FlipTest - binary case
    # a = adv facet, d = disadv facet

    data_a = ([el for idx, el in enumerate(dataset) if ~facet[idx]],
              [el for idx, el in enumerate(predicted_labels) if ~facet[idx]],
              [el for idx, el in enumerate(facet) if ~facet[idx]])
    data_d = ([el for idx, el in enumerate(dataset) if facet[idx]],
              [el for idx, el in enumerate(predicted_labels) if facet[idx]],
              [el for idx, el in enumerate(facet) if facet[idx]])

    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                               metric='minkowski', metric_params=None, n_jobs=None)

    # kNN method over a with Labels from the model
    knn.fit(np.array(data_a[0]), np.array(data_a[1]))
    # kNN prediction over d
    d_y_if_a = knn.predict(data_d[0])
    # Model predictions over the same test d
    d_y_model = data_d[1]

    FS_pos = FlipSet_pos(dataset=data_d[1], labels=d_y_model, predicted_labels=d_y_if_a)
    FS_neg = FlipSet_neg(dataset=data_d[1], labels=d_y_model, predicted_labels=d_y_if_a)
    FS = FlipSet(dataset=data_d[1], labels=d_y_model, predicted_labels=d_y_if_a)

    if verbose > 0:
        print('Data with', len(dataset), 'examples -- ', len(data_d[0]), 'female examples')
        print('Length of FlipSet positive (i.e. positive bias to females w.r.t. males):', len(FS_pos), '(',
              100 * len(FS_pos) / len(data_d[0]), '%)')
        print('Length of FlipSet negative (i.e. negative bias to females w.r.t. males):', len(FS_neg), '(',
              100 * len(FS_neg) / len(data_d[0]), '%)')
        print('Length of FlipSet:', len(FS), '(', 100 * len(FS) / len(data_d[0]), '%)')

    FTd = (len(FS_pos) - len(FS_neg)) / len(data_d[0])
    FTs = len(FS) / len(data_d[0])

    return FTd
