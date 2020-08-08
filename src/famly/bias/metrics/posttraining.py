"""
Post training metrics
"""
import logging
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from famly.bias.metrics.constants import INFINITY
from . import registry, common

log = logging.getLogger(__name__)


@registry.posttraining
def DPPL(
    feature: pd.Series, facet: pd.Series, predicted_label: pd.Series, positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Difference in positive proportions in predicted labels.

    Indication if initial bias resident in the dataset increases or decreases after training.

    .. math::
        DPPL = \frac{\hat{n_a}^{(1)}}{n_a}-\frac{\hat{n_d}^{(1)}}{n_d}

    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param predicted_label: boolean column indicating predictions made by model
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Returns Difference in Positive Proportions, based on predictions rather than labels
    """
    return common.DPL(feature, facet, predicted_label, positive_predicted_label_index)


@registry.posttraining
def DI(
    feature: pd.Series, facet: pd.Series, predicted_label: pd.Series, positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Disparate Impact

    Measures adverse effects by the model predictions with respect to labels on different groups selected by
    the facet.

    .. math::
        DI = \frac{\frac{\hat{n_a}^{(1)}}{n_a}}{\frac{\hat{n_d}^{(1)}}{n_d}}

    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param predicted_label: boolean column indicating predictions made by model
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Returns disparate impact, the ratio between positive proportions, based on predicted labels
    """
    facet = facet.astype(bool)
    positive_predicted_label_index = positive_predicted_label_index.astype(bool)

    na1hat = len(predicted_label[positive_predicted_label_index & (~facet)])
    na = len(feature[~facet])
    if na == 0:
        raise ValueError("DI: Negated facet set is empty")
    qa = na1hat / na
    nd1hat = len(predicted_label[positive_predicted_label_index & facet])
    nd = len(feature[facet])
    if nd == 0:
        raise ValueError("DI: Facet set is empty")
    qd = nd1hat / nd
    if qa == 0:
        return INFINITY
    return qd / qa


@registry.posttraining
def DCO(
    feature: pd.Series, facet: pd.Series, positive_label_index: pd.Series, positive_predicted_label_index: pd.Series,
) -> (float, float):
    """
    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Difference in Conditional Outcomes (Acceptance and Rejection) between advantaged and disadvantaged classes
    """
    facet = facet.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    positive_predicted_label_index = positive_predicted_label_index.astype(bool)

    if len(feature[facet]) == 0:
        raise ValueError("DCO: Facet set is empty")
    if len(feature[~facet]) == 0:
        raise ValueError("DCO: Negated Facet set is empty")

    na0 = len(feature[~positive_label_index & ~facet])
    na0hat = len(feature[~positive_predicted_label_index & ~facet])
    nd0 = len(feature[~positive_label_index & facet])
    nd0hat = len(feature[~positive_predicted_label_index & facet])

    na1 = len(feature[positive_label_index & ~facet])
    na1hat = len(feature[positive_predicted_label_index & ~facet])
    nd1 = len(feature[positive_label_index & facet])
    nd1hat = len(feature[positive_predicted_label_index & facet])

    if na0hat != 0:
        rr_a = na0 / na0hat
    else:
        rr_a = INFINITY

    if nd0hat != 0:
        rr_d = nd0 / nd0hat
    else:
        rr_d = INFINITY

    if na1hat != 0:
        ca = na1 / na1hat
    else:
        ca = INFINITY

    if nd1hat != 0:
        cd = nd1 / nd1hat
    else:
        cd = INFINITY

    dca = ca - cd
    dcr = rr_d - rr_a

    if ca == cd and ca == INFINITY:
        dca = 0
    if rr_a == rr_d and rr_a == INFINITY:
        dcr = 0

    return dca, dcr


@registry.posttraining
def RD(
    feature: pd.Series,
    facet: pd.Series,
    label: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    """
    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param label: boolean column indicating labels
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Recall Difference between advantaged and disadvantaged classes
    """
    facet = facet.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    positive_predicted_label_index = positive_predicted_label_index.astype(bool)

    if len(feature[facet]) == 0:
        raise ValueError("RD: Facet set is empty")
    if len(feature[~facet]) == 0:
        raise ValueError("RD: Negated Facet set is empty")

    TP_a = len(label[positive_label_index & positive_predicted_label_index & (~facet)])
    FN_a = len(label[positive_label_index & (~positive_predicted_label_index) & (~facet)])

    rec_a = TP_a / (TP_a + FN_a) if TP_a + FN_a != 0 else INFINITY

    TP_d = len(label[positive_label_index & positive_predicted_label_index & (facet)])
    FN_d = len(label[positive_label_index & (~positive_predicted_label_index) & (facet)])

    rec_d = TP_d / (TP_d + FN_d) if TP_d + FN_d != 0 else INFINITY

    rd = rec_a - rec_d

    if rec_a == rec_d and rec_a == INFINITY:
        rd = 0
    return rd


@registry.posttraining
def DLR(
    feature: pd.Series,
    facet: pd.Series,
    label: pd.Series,
    positive_label_index: pd.Series,
    predicted_label: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> (float, float):
    """
    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param label: boolean column indicating labels
    :param positive_label_index: boolean column indicating positive labels
    :param predicted_label: boolean column indicating predictions made by model
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Difference in Label Rates (aka Difference in Acceptance Rates AND Difference in Rejected Rates)
    """
    facet = facet.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    positive_predicted_label_index = positive_predicted_label_index.astype(bool)

    if len(feature[facet]) == 0:
        raise ValueError("DLR: Facet set is empty")
    if len(feature[~facet]) == 0:
        raise ValueError("DLR: Negated Facet set is empty")

    TP_a = len(label[positive_label_index & positive_predicted_label_index & (~facet)])
    na1hat = len(predicted_label[positive_predicted_label_index & (~facet)])
    TP_d = len(label[positive_label_index & positive_predicted_label_index & facet])
    nd1hat = len(predicted_label[positive_predicted_label_index & facet])

    TN_a = len(label[(~positive_label_index) & (~positive_predicted_label_index) & (~facet)])
    na0hat = len(predicted_label[(~positive_predicted_label_index) & (~facet)])
    TN_d = len(label[(~positive_label_index) & (~positive_predicted_label_index) & facet])
    nd0hat = len(predicted_label[(~positive_predicted_label_index) & facet])

    if na1hat != 0:
        ar_a = TP_a / na1hat
    else:
        ar_a = INFINITY

    if nd1hat != 0:
        ar_d = TP_d / nd1hat
    else:
        ar_d = INFINITY

    if na0hat != 0:
        rr_a = TN_a / na0hat
    else:
        rr_a = INFINITY

    if nd0hat != 0:
        rr_d = TN_d / nd0hat
    else:
        rr_d = INFINITY

    dar = ar_a - ar_d
    drr = rr_a - rr_d

    if ar_a == ar_d and ar_a == INFINITY:
        dar = 0
    if rr_a == rr_d and rr_a == INFINITY:
        drr = 0

    return dar, drr


@registry.posttraining
def AD(
    feature: pd.Series,
    facet: pd.Series,
    label: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    """
    Accuracy difference

    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param label: boolean column indicating labels
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Accuracy Difference between advantaged and disadvantaged classes
    """
    label = label.astype(bool)
    facet = facet.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    positive_predicted_label_index = positive_predicted_label_index.astype(bool)

    if len(feature[facet]) == 0:
        raise ValueError("AD: Facet set is empty")
    if len(feature[~facet]) == 0:
        raise ValueError("AD: Negated Facet set is empty")

    idx_tp_a = positive_label_index & positive_predicted_label_index & ~facet
    TP_a = len(label[idx_tp_a])
    idx_fp_a = ~positive_label_index & positive_predicted_label_index & ~facet
    FP_a = len(label[idx_fp_a])
    idx_fn_a = positive_label_index & ~positive_predicted_label_index & ~facet
    FN_a = len(label[idx_fn_a])
    idx_tn_a = ~positive_label_index & ~positive_predicted_label_index & ~facet
    TN_a = len(label[idx_tn_a])

    total_a = TP_a + TN_a + FP_a + FN_a
    acc_a = (TP_a + TN_a) / total_a if total_a != 0 else INFINITY

    idx_tp_d = positive_label_index & positive_predicted_label_index & facet
    TP_d = len(label[idx_tp_d])
    idx_fp_d = ~positive_label_index & positive_predicted_label_index & facet
    FP_d = len(label[idx_fp_d])
    idx_fn_d = positive_label_index & ~positive_predicted_label_index & facet
    FN_d = len(label[idx_fn_d])
    idx_tn_d = ~positive_label_index & ~positive_predicted_label_index & facet
    TN_d = len(label[idx_tn_d])

    total_d = TP_d + TN_d + FP_d + FN_d
    acc_d = (TP_d + TN_d) / total_d if total_d != 0 else INFINITY

    ad = acc_a - acc_d
    if acc_a == acc_d and acc_a == INFINITY:
        ad = 0.0
    return ad


# FIXME, CDDPL needs to be looked into
# @registry.posttraining
def CDDPL(feature: pd.Series, facet: pd.Series, predicted_labels: pd.Series, group_variable: pd.Series) -> float:
    r"""
    Conditional Demographic Disparity in labels
    .. math::
        CDD = \frac{1}{n}\sum_i n_i * DD_i \\\quad\:where \: DD_i = \frac{Number\:of\:rejected\:applicants\:protected\:facet}{Total\:number\:of\:rejected\:applicants} -
        \frac{Number\:of\:accepted\:applicants\:protected\:facet}{Total\:number\:of\:accepted\:applicants} \\for\:each\:group\:variable\: i

    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param label: boolean column indicating positive labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    return common.CDD(feature, facet, predicted_labels, group_variable)


@registry.posttraining
def TE(
    feature: pd.Series,
    facet: pd.Series,
    label: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    """
    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param label: boolean column indicating labels
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return: Returns the difference in ratios between false negatives and false positives for the advantaged
    and disadvantaged classes
    """
    facet = facet.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    positive_predicted_label_index = positive_predicted_label_index.astype(bool)

    if len(feature[facet]) == 0:
        raise ValueError("TE: Facet set is empty")
    if len(feature[~facet]) == 0:
        raise ValueError("TE: Negated Facet set is empty")

    FP_a = len(label[(~label) & positive_predicted_label_index & (~facet)])
    FN_a = len(label[positive_label_index & (~positive_predicted_label_index) & (~facet)])
    FP_d = len(label[(~label) & positive_predicted_label_index & facet])
    FN_d = len(label[positive_label_index & (~positive_predicted_label_index) & facet])

    tau_a = FN_a / FP_a if FP_a != 0 else INFINITY
    tau_d = FN_d / FP_d if FP_d != 0 else INFINITY

    te = tau_d - tau_a

    if tau_a == tau_d and tau_a == INFINITY:
        te = 0

    return te


def FlipSet_pos(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] > predicted_labels[i]])


def FlipSet_neg(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] < predicted_labels[i]])


def FlipSet(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] != predicted_labels[i]])


# @registry.posttraining
# FIXME: Registering this metric with post training metrics results in failure
def FT(dataset: pd.DataFrame, facet: pd.Series, labels: pd.Series, predicted_labels: pd.Series) -> float:
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
        raise ValueError("FT: Facet set is empty")
    if len(facet[~facet]) == 0:
        raise ValueError("FT: Negated Facet set is empty")

    dataset = np.array(dataset)

    data_a = (
        [el for idx, el in enumerate(dataset) if ~facet[idx]],
        [el for idx, el in enumerate(predicted_labels) if ~facet[idx]],
        [el for idx, el in enumerate(facet) if ~facet[idx]],
    )
    data_d = (
        [el for idx, el in enumerate(dataset) if facet[idx]],
        [el for idx, el in enumerate(predicted_labels) if facet[idx]],
        [el for idx, el in enumerate(facet) if facet[idx]],
    )

    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    )

    # kNN method over a with Labels from the model
    knn.fit(np.array(data_a[0]), np.array(data_a[1]))
    # kNN prediction over d
    d_y_if_a = knn.predict(data_d[0])
    # Model predictions over the same test d
    d_y_model = data_d[1]

    FS_pos = FlipSet_pos(dataset=data_d[1], labels=d_y_model, predicted_labels=d_y_if_a)
    FS_neg = FlipSet_neg(dataset=data_d[1], labels=d_y_model, predicted_labels=d_y_if_a)
    FS = FlipSet(dataset=data_d[1], labels=d_y_model, predicted_labels=d_y_if_a)

    # if verbose > 0:
    #     print('Data with', len(dataset), 'examples -- ', len(data_d[0]), 'female examples')
    #     print('Length of FlipSet positive (i.e. positive bias to females w.r.t. males):', len(FS_pos), '(',
    #           100 * len(FS_pos) / len(data_d[0]), '%)')
    #     print('Length of FlipSet negative (i.e. negative bias to females w.r.t. males):', len(FS_neg), '(',
    #           100 * len(FS_neg) / len(data_d[0]), '%)')
    #     print('Length of FlipSet:', len(FS), '(', 100 * len(FS) / len(data_d[0]), '%)')

    FTd = (len(FS_pos) - len(FS_neg)) / len(data_d[0])
    FTs = len(FS) / len(data_d[0])

    return FTd
