import logging
from typing import Dict, Callable
from ...famly.util.util import PDF
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

log = logging.getLogger(__name__)

INFINITE = float('inf') #Default return value for all metrics to avoid division by zero errors

pretraining_metrics = ['CI', 'DPL', 'KL', 'JS', 'LP', 'TVD', 'KS', 'CDD']
posttraining_metrics = ['DPPL', 'DI', 'DCO', 'RD', 'DLR', 'AD', 'TE', 'FT']

#Methods to handle multicategory cases

def metric_one_vs_all(metric: Callable[..., float], x: pd.Series, facet: pd.Series, positive_label_index: pd.Series=None, predicted_labels: pd.Series=None, labels: pd.Series=None, group_variable: pd.Series=None, dataset: pd.DataFrame=None) -> Dict:
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
    :return: A dictionary in which each key is one of the sensitive attributes in the facet column, and each value is
            its corresponding metric according to the requested bias measure
    """
    #Ensure correct parameter types
    x = pd.Series(x)
    facet = pd.Series(facet)
    if metric.__name__ not in pretraining_metrics and metric.__name__ not in posttraining_metrics:
        raise ValueError("Metric passed in is invalid - not an implemented bias metric")
    if positive_label_index is not None:
        positive_label_index = pd.Series(positive_label_index)
    if predicted_labels is not None:
        predicted_labels = pd.Series(predicted_labels)
    if labels is not None:
        labels = pd.Series(labels)
    if group_variable is not None:
        group_variable = pd.Series(group_variable)
    if dataset is not None:
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

def label_one_vs_all(metric: Callable[..., float], x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series=None, group_variable: pd.Series=None) -> Dict:
    """
    :param metric: one of the bias measures defined in this file
    :param x: input feature
    :param facet: boolean column with true values indicating sensitive value
    :param predicted_labels: predictions for labels made by model
    :param labels: True values of the target column
    :param group_variable: column of values indicating the subgroup each data point belongs to (used for calculating CDD metric only)
    :return: value returned by the specified bias measure
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
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: series of boolean values indicating positive target labels
    :return: a float in the interval [-1, +1] indicating an under-representation or over-representation
    of the protected class.

    Bias is often generated from an under-representation of
    the protected class in the dataset, especially if the desired “golden truth”
    is equality across classes. Imbalance carries over into model predictions.
    We will report all measures in differences and normalized differences. Since
    the measures are often probabilities or proportions, the differences will lie in
    We define CI = (np − p)/(np + p). Where np is the number of instances in the not protected group
    and p is number of instances in the sensitive group.
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
    :param x: input feature
    :param facet: boolean column indicating sensitive group
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
    :param facet: boolean column indicating sensitive group
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
        raise ValueError('KL: Either facet set or negated facet set is empty')
    return kl


def JS(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
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
        raise ValueError('JS: Either facet set or negated facet set is empty')

    return js_divergence

def LP(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series, p: int=2) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
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
        raise ValueError('LP: Either facet set or negated facet set is empty')

    return lp_norm


def TVD(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
   :param x: input feature
   :param facet: boolean column indicating sensitive group
   :param positive_label_index: boolean column indicating positive labels
   :return: 1/2 * L-1 norm
   """

    Lp_res = LP(x, facet, positive_label_index, p=1)

    tvd = 0.5 * Lp_res

    return tvd

def KS(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :return: Kolmogorov-Smirnov metric
    """
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    x_a = positive_label_index[~facet]
    x_d = positive_label_index[facet]

    Pa = PDF(x_a)  # x: raw values of the variable (column of data)
    Pd = PDF(x_d)

    if len(Pa) == len(Pd):
        max_distance = np.max(np.abs(Pa - Pd))
    else:
        raise ValueError('KS: Either facet set or negated facet set is empty')

    return max_distance

def CDD(x: pd.Series, facet: pd.Series, positive_label_index: pd.Series, group_variable: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    unique_groups = np.unique(group_variable)
    positive_label_index = positive_label_index.astype(bool)
    facet = facet.astype(bool)

    # Global demographic disparity (DD)
    numA = len(positive_label_index[(positive_label_index) & (facet)])
    denomA = len(facet[positive_label_index])

    if denomA == 0:
        raise ValueError('CDD: No positive labels in set')

    A = numA / denomA
    numD = len(positive_label_index[(~positive_label_index) & (facet)])
    denomD = len(facet[~positive_label_index])

    if denomD == 0:
        raise ValueError('CDD: No negative labels in set')

    D = numD / denomD
    DD = D - A

    # Conditional demographic disparity (CDD)
    CDD = []
    counts = []
    for subgroup_variable in unique_groups:
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
    :param facet: boolean column indicating sensitive group
    :param labels: boolean column indicating true values of target column
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Returns Difference in Positive Proportions, based on predictions rather than true labels
    """
    predicted_labels = predicted_labels.astype(bool)
    labels = labels.astype(bool)
    facet = facet.astype(bool)

    na1hat = len(predicted_labels[(predicted_labels) & (~facet)])
    na = len(facet[~facet])

    if na == 0:
        raise ValueError('DPPL: Negated facet set is empty')

    qa = na1hat / na
    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])
    nd = len(facet[facet])

    if nd == 0:
        raise ValueError('DPPL: facet set is empty')

    qd = nd1hat / nd

    return qa - qd


def DI(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    # Disparate impact
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param labels: boolean column indicating true values of target column
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Returns disparate impact, the ratio between positive proportions, based on predicted labels
    """
    predicted_labels = predicted_labels.astype(bool)
    labels = labels.astype(bool)
    facet = facet.astype(bool)

    na1hat = len(predicted_labels[(predicted_labels) & (~facet)])
    na = len(facet[~facet])

    qa = na1hat / na

    if na == 0:
        raise ValueError('DI: Negated facet set is empty')

    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])
    nd = len(facet[facet])

    if nd == 0:
        raise ValueError('DI: Facet set is empty')

    qd = nd1hat / nd

    return qd / qa


def DCO(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> (float, float):
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Difference in Conditional Outcomes (Acceptance and Rejection) between advantaged and disadvantaged classes
    """
    predicted_labels = predicted_labels.astype(bool)
    labels = labels.astype(bool)
    facet = facet.astype(bool)

    if len(facet[facet]) == 0:
        raise ValueError('DCO: Facet set is empty')
    if len(facet[~facet]) == 0:
        raise ValueError('DCO: Negated Facet set is empty')

    TN_a = len(labels[(~labels) & (~predicted_labels) & (~facet)])
    na0hat = len(predicted_labels[(~predicted_labels) & (~facet)])
    TN_d = len(labels[(~labels) & (~predicted_labels) & (facet)])
    nd0hat = len(predicted_labels[(~predicted_labels) & (facet)])

    na1 = len(labels[(labels) & (~facet)])
    na1hat = len(predicted_labels[(predicted_labels) & (~facet)])
    nd1 = len(labels[(labels) & (facet)])
    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])

    if na0hat != 0:
        rr_a = TN_a / na0hat
    else:
        rr_a = INFINITE

    if nd0hat != 0:
        rr_d = TN_d / nd0hat
    else:
        rr_d = INFINITE

    if na1hat != 0:
        ca = na1 / na1hat
    else:
        ca = INFINITE

    if nd1hat != 0:
        cd = nd1 / nd1hat
    else:
        cd = INFINITE

    dca = ca - cd
    dcr = rr_a - rr_d

    if ca == cd and ca == float('inf'):
        dca = 0
    if rr_a == rr_d and rr_a == float('inf'):
        dcr = 0

    return dca, dcr

def RD(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Recall Difference between advantaged and disadvantaged classes
    """
    predicted_labels = predicted_labels.astype(bool)
    labels = labels.astype(bool)
    facet = facet.astype(bool)

    if len(facet[facet]) == 0:
        raise ValueError('RD: Facet set is empty')
    if len(facet[~facet]) == 0:
        raise ValueError('RD: Negated Facet set is empty')

    TP_a = len(labels[(labels) & (predicted_labels) & (~facet)])
    FN_a = len(labels[(labels) & (~predicted_labels) & (~facet)])

    rec_a = TP_a / (TP_a + FN_a) if TP_a + FN_a != 0 else INFINITE

    TP_d = len(labels[(labels) & (predicted_labels) & (facet)])
    FN_d = len(labels[(labels) & (~predicted_labels) & (facet)])

    rec_d = TP_d / (TP_d + FN_d) if TP_d + FN_d != 0 else INFINITE

    rd = rec_a - rec_d

    if rec_a == rec_d and rec_a == float('inf'):
        rd = 0
    return rd


def DLR(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> (float, float):
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Difference in Label Rates (aka Difference in Acceptance Rates AND Difference in Rejected Rates)
    """
    predicted_labels = predicted_labels.astype(bool)
    labels = labels.astype(bool)
    facet = facet.astype(bool)

    if len(facet[facet]) == 0:
        raise ValueError('DLR: Facet set is empty')
    if len(facet[~facet]) == 0:
        raise ValueError('DLR: Negated Facet set is empty')

    TP_a = len(labels[(labels) & (predicted_labels) & (~facet)])
    na1hat = len(predicted_labels[(predicted_labels) & (~facet)])
    TP_d = len(labels[(labels) & (predicted_labels) & (facet)])
    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])

    TN_a = len(labels[(~labels) & (~predicted_labels) & (~facet)])
    na0hat = len(predicted_labels[(~predicted_labels) & (~facet)])
    TN_d = len(labels[(~labels) & (~predicted_labels) & (facet)])
    nd0hat = len(predicted_labels[(~predicted_labels) & (facet)])

    if na1hat != 0:
        ar_a = TP_a / na1hat
    else:
        ar_a = INFINITE

    if nd1hat != 0:
        ar_d = TP_d / nd1hat
    else:
        ar_d = INFINITE

    if na0hat != 0:
        rr_a = TN_a / na0hat
    else:
        rr_a = INFINITE

    if nd0hat != 0:
        rr_d = TN_d / nd0hat
    else:
        rr_d = INFINITE

    dar = ar_a - ar_d
    drr = rr_a - rr_d

    if ar_a == ar_d and ar_a == float('inf'):
        dar = 0
    if rr_a == rr_d and rr_a == float('inf'):
        drr = 0

    return dar, drr


def AD(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Accuracy Difference between advantaged and disadvantaged classes
    """
    predicted_labels = predicted_labels.astype(bool)
    labels = labels.astype(bool)
    facet = facet.astype(bool)

    if len(facet[facet]) == 0:
        raise ValueError('AD: Facet set is empty')
    if len(facet[~facet]) == 0:
        raise ValueError('AD: Negated Facet set is empty')

    TP_a = len(labels[(labels) & (predicted_labels) & (~facet)])
    FP_a = len(labels[(~labels) & (predicted_labels) & (~facet)])
    FN_a = len(labels[(labels) & (~predicted_labels) & (~facet)])
    TN_a = len(labels[(~labels) & (~predicted_labels) & (~facet)])


    acc_a = (TP_a + TN_a) / (TP_a + TN_a + FP_a + FN_a) if (TP_a + TN_a + FP_a + FN_a) != 0 else INFINITE

    TP_d = len(labels[(labels) & (predicted_labels) & (facet)])
    FP_d = len(labels[(~labels) & (predicted_labels) & (facet)])
    FN_d = len(labels[(labels) & (~predicted_labels) & (facet)])
    TN_d = len(labels[(~labels) & (~predicted_labels) & (facet)])



    acc_d = (TP_d + TN_d) / (TP_d + TN_d + FP_d + FN_d) if (TP_d + TN_d + FP_d + FN_d) != 0 else INFINITE

    ad = acc_a - acc_d

    if acc_a == acc_d and acc_a == float('inf'):
        ad = 0

    return ad

def TE(x: pd.Series, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
    """
    :param x: input feature
    :param facet: boolean column indicating sensitive group
    :param labels: true values of the target column for the data
    :param predicted_labels: boolean column indicating predictions made by model
    :return: Returns the difference in ratios between false negatives and false positives for the advantaged
    and disadvantaged classes
    """
    predicted_labels = predicted_labels.astype(bool)
    labels = labels.astype(bool)
    facet = facet.astype(bool)

    if len(facet[facet]) == 0:
        raise ValueError('TE: Facet set is empty')
    if len(facet[~facet]) == 0:
        raise ValueError('TE: Negated Facet set is empty')

    FP_a = len(labels[(~labels) & (predicted_labels) & (~facet)])
    FN_a = len(labels[(labels) & (~predicted_labels) & (~facet)])
    FP_d = len(labels[(~labels) & (predicted_labels) & (facet)])
    FN_d = len(labels[(labels) & (~predicted_labels) & (facet)])

    tau_a = FN_a / FP_a if FP_a != 0 else INFINITE
    tau_d = FN_d / FP_d if FP_d != 0 else INFINITE

    te = tau_d - tau_a

    if tau_a == tau_d and tau_a == float('inf'):
        te = 0

    return te


def FlipSet_pos(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] > predicted_labels[i]])


def FlipSet_neg(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] < predicted_labels[i]])


def FlipSet(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] != predicted_labels[i]])


def FT(dataset: pd.DataFrame, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series, verbose=0) -> float:
    """
    :param dataset: array of data points
    :param facet: boolean column indicating sensitive group
    :param labels: boolean column of positive values for target column
    :param predicted_labels: boolean column of predicted positive values for target column
    :param verbose: optional boolean value
    :return: FT difference metric
    """
    # FlipTest - binary case
    # a = adv facet, d = disadv facet
    predicted_labels = predicted_labels.astype(bool)
    labels = labels.astype(bool)
    facet = facet.astype(bool)

    if len(facet[facet]) == 0:
        raise ValueError('FT: Facet set is empty')
    if len(facet[~facet]) == 0:
        raise ValueError('FT: Negated Facet set is empty')

    dataset = np.array(dataset)

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
