"""
Post training metrics
"""
import logging
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from famly.bias.metrics.constants import INFINITY
from . import registry, common
from famly.bias.metrics.common import require
from .registry import ProblemType

log = logging.getLogger(__name__)


@registry.posttraining(problem_type=ProblemType.BINARY)
def DPPL(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    predicted_label: pd.Series,
) -> float:
    r"""
    "Difference in Positive Proportions in Predicted Labels (DPPL)")

    Indication if initial bias resident in the dataset increases or decreases after training.

    .. math::
        DPPL = \frac{\hat{n_a}^{(1)}}{n_a}-\frac{\hat{n_d}^{(1)}}{n_d}

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param predicted_label: boolean column indicating positive predicted labels
    :return: Returns Difference in Positive Proportions, based on predictions rather than labels
    """
    return common.DPL(feature, sensitive_facet_index, predicted_label)


@registry.posttraining(problem_type=ProblemType.BINARY)
def DI(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    predicted_label: pd.Series,
) -> float:
    r"""
    Disparate Impact (DI)

    Measures adverse effects by the model predictions with respect to labels on different groups selected by
    the facet.

    .. math::
        DI = \frac{\frac{\hat{n_a}^{(1)}}{n_a}}{\frac{\hat{n_d}^{(1)}}{n_d}}

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param predicted_label: boolean column indicating positive predicted labels
    :return: Returns disparate impact, the ratio between positive proportions, based on predicted labels
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must of dtype bool")
    require(predicted_label.dtype == bool, "predicted_label must of dtype bool")

    na1hat = len(feature[predicted_label & (~sensitive_facet_index)])
    na = len(feature[~sensitive_facet_index])
    if na == 0:
        raise ValueError("DI: Negated facet set is empty")
    qa = na1hat / na
    nd1hat = len(feature[predicted_label & sensitive_facet_index])
    nd = len(feature[sensitive_facet_index])
    if nd == 0:
        raise ValueError("DI: Facet set is empty")
    qd = nd1hat / nd
    if qa == 0:
        return INFINITY
    return qd / qa


@registry.posttraining(problem_type=ProblemType.BINARY)
def DCA(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    label: pd.Series,
    predicted_label: pd.Series,
) -> float:
    """
    Difference in Conditional Acceptance (DCA)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param label: boolean column indicating positive labels
    :param predicted_label: boolean column indicating positive predicted labels
    :return: Difference in Conditional Acceptance between advantaged and disadvantaged classes
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must of dtype bool")
    require(predicted_label.dtype == bool, "predicted_label must of dtype bool")
    dca, _ = common.DCO(feature, sensitive_facet_index, label, predicted_label)
    return dca


@registry.posttraining(problem_type=ProblemType.BINARY)
def DCR(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    label: pd.Series,
    predicted_label: pd.Series,
) -> float:
    """
    Difference in Conditional Rejection (DCR)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param label: boolean column indicating positive labels
    :param predicted_label: boolean column indicating positive predicted labels
    :return: Difference in Conditional Rejection between advantaged and disadvantaged classes
    """
    _, dcr = common.DCO(feature, sensitive_facet_index, label, predicted_label)
    return dcr


@registry.posttraining(problem_type=ProblemType.BINARY)
def RD(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    label: pd.Series,
    predicted_label: pd.Series,
) -> float:
    """
    Recall Difference (RD)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param label: boolean column indicating positive labels
    :param predicted_label: boolean column indicating positive predicted labels
    :return: Recall Difference between advantaged and disadvantaged classes
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must of dtype bool")
    require(label.dtype == bool, "label must of dtype bool")
    require(predicted_label.dtype == bool, "predicted_label must of dtype bool")

    if len(feature[sensitive_facet_index]) == 0:
        raise ValueError("RD: Facet set is empty")
    if len(feature[~sensitive_facet_index]) == 0:
        raise ValueError("RD: Negated Facet set is empty")

    TP_a = len(feature[label & predicted_label & (~sensitive_facet_index)])
    FN_a = len(feature[label & (~predicted_label) & (~sensitive_facet_index)])

    rec_a = TP_a / (TP_a + FN_a) if TP_a + FN_a != 0 else INFINITY

    TP_d = len(feature[label & predicted_label & sensitive_facet_index])
    FN_d = len(feature[label & (~predicted_label) & sensitive_facet_index])

    rec_d = TP_d / (TP_d + FN_d) if TP_d + FN_d != 0 else INFINITY

    rd = rec_a - rec_d

    if rec_a == rec_d and rec_a == INFINITY:
        rd = 0
    return rd


@registry.posttraining(problem_type=ProblemType.BINARY)
def DAR(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    label: pd.Series,
    predicted_label: pd.Series,
) -> float:
    """
    Difference in Acceptance Rates (DAR)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param label: boolean column indicating positive labels
    :param predicted_label: boolean column indicating positive predicted labels
    :return: Difference in Acceptance Rates
    """

    dar, _ = common.DLR(feature, sensitive_facet_index, label, predicted_label)
    return dar


@registry.posttraining(problem_type=ProblemType.BINARY)
def DRR(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    label: pd.Series,
    predicted_label: pd.Series,
) -> float:
    """
    Difference in Rejection Rates (DRR)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param label: boolean column indicating positive labels
    :param predicted_label: boolean column indicating positive predicted labels
    :return: Difference in Rejection Rates
    """
    _, drr = common.DLR(feature, sensitive_facet_index, label, predicted_label)
    return drr


@registry.posttraining(problem_type=ProblemType.BINARY)
def AD(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    label: pd.Series,
    predicted_label: pd.Series,
) -> float:
    """
    Accuracy Difference (AD)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param label: boolean column indicating positive labels
    :param predicted_label: boolean column indicating positive predicted labels
    :return: Accuracy Difference between advantaged and disadvantaged classes
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must of dtype bool")
    require(label.dtype == bool, "label must of dtype bool")
    require(predicted_label.dtype == bool, "predicted_label must of dtype bool")

    if len(feature[sensitive_facet_index]) == 0:
        raise ValueError("AD: Facet set is empty")
    if len(feature[~sensitive_facet_index]) == 0:
        raise ValueError("AD: Negated Facet set is empty")

    idx_tp_a = label & predicted_label & ~sensitive_facet_index
    TP_a = len(feature[idx_tp_a])
    idx_fp_a = ~label & predicted_label & ~sensitive_facet_index
    FP_a = len(feature[idx_fp_a])
    idx_fn_a = label & ~predicted_label & ~sensitive_facet_index
    FN_a = len(feature[idx_fn_a])
    idx_tn_a = ~label & ~predicted_label & ~sensitive_facet_index
    TN_a = len(feature[idx_tn_a])

    total_a = TP_a + TN_a + FP_a + FN_a
    acc_a = (TP_a + TN_a) / total_a if total_a != 0 else INFINITY

    idx_tp_d = label & predicted_label & sensitive_facet_index
    TP_d = len(feature[idx_tp_d])
    idx_fp_d = ~label & predicted_label & sensitive_facet_index
    FP_d = len(feature[idx_fp_d])
    idx_fn_d = label & ~predicted_label & sensitive_facet_index
    FN_d = len(feature[idx_fn_d])
    idx_tn_d = ~label & ~predicted_label & sensitive_facet_index
    TN_d = len(feature[idx_tn_d])

    total_d = TP_d + TN_d + FP_d + FN_d
    acc_d = (TP_d + TN_d) / total_d if total_d != 0 else INFINITY

    ad = acc_a - acc_d
    if acc_a == acc_d and acc_a == INFINITY:
        ad = 0.0
    return ad


@registry.posttraining(problem_type=ProblemType.BINARY)
def CDDPL(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    predicted_label: pd.Series,
    group_variable: pd.Series,
) -> float:
    r"""
    Conditional Demographic Disparity in Predicted Labels (CDDPL)

    .. math::
        CDD = \frac{1}{n}\sum_i n_i * DD_i \\\quad\:where \: DD_i = \frac{Number\:of\:rejected\:applicants\:sensitive\:facet}{Total\:number\:of\:rejected\:applicants} -
        \frac{Number\:of\:accepted\:applicants\:sensitive\:facet}{Total\:number\:of\:accepted\:applicants} \\for\:each\:group\:variable\: i

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    return common.CDD(feature, sensitive_facet_index, predicted_label, group_variable)


@registry.posttraining(problem_type=ProblemType.BINARY)
def TE(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    label: pd.Series,
    predicted_label: pd.Series,
) -> float:
    """
    Treatment Equality (TE)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param label: boolean column indicating positive labels
    :param predicted_label: boolean column indicating positive predicted labels
    :return: Returns the difference in ratios between false negatives and false positives for the advantaged
        and disadvantaged classes
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must of dtype bool")
    require(label.dtype == bool, "label must of dtype bool")
    require(predicted_label.dtype == bool, "predicted_label must of dtype bool")

    if len(feature[sensitive_facet_index]) == 0:
        raise ValueError("TE: Facet set is empty")
    if len(feature[~sensitive_facet_index]) == 0:
        raise ValueError("TE: Negated Facet set is empty")

    FP_a = len(feature[(~label) & predicted_label & (~sensitive_facet_index)])
    FN_a = len(feature[label & (~predicted_label) & (~sensitive_facet_index)])
    FP_d = len(feature[(~label) & predicted_label & sensitive_facet_index])
    FN_d = len(feature[label & (~predicted_label) & sensitive_facet_index])

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


@registry.posttraining(problem_type=ProblemType.MULTICLASS)
def FT(df: pd.DataFrame, sensitive_facet_index: pd.Series, predicted_label: pd.Series) -> float:
    """
    Flip Test (FT)
    :param df: array of data points
    :param sensitive_facet_index: boolean facet column indicating sensitive group
    :param predicted_label: boolean column of predicted positive values for target column
    :return: FT difference metric
    """
    # FlipTest - binary case
    # a = adv facet, d = disadv facet
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must of dtype bool")
    require(predicted_label.dtype == bool, "predicted_label must of dtype bool")

    if len(df[sensitive_facet_index]) == 0:
        raise ValueError("FT: Facet set is empty")
    if len(df[~sensitive_facet_index]) == 0:
        raise ValueError("FT: Negated Facet set is empty")

    dataset = np.array(df)

    data_a = (
        [el for idx, el in enumerate(dataset) if ~sensitive_facet_index.iat[idx]],
        [el for idx, el in enumerate(predicted_label) if ~sensitive_facet_index.iat[idx]],
        [el for idx, el in enumerate(sensitive_facet_index) if ~sensitive_facet_index.iat[idx]],
    )
    data_d = (
        [el for idx, el in enumerate(dataset) if sensitive_facet_index.iat[idx]],
        [el for idx, el in enumerate(predicted_label) if sensitive_facet_index.iat[idx]],
        [el for idx, el in enumerate(sensitive_facet_index) if sensitive_facet_index.iat[idx]],
    )
    n_neighbors = 5 if np.array(data_a[0]).size > 16 else 1

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
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
