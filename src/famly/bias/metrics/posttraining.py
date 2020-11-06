"""
Post training metrics
The metrics defined in this file must be computed after training the model.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from famly.bias.metrics.constants import INFINITY, FT_DEFAULT_NEIGHBOR, FT_MIN_NEIGHBOR, FT_SAMPLES_COUNT_THRESHOLD
from . import registry, common
from .common import divide, require

log = logging.getLogger(__name__)


@registry.posttraining
def DPPL(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Difference in Positive Proportions in Predicted Labels (DPPL)

    Indication if initial bias resident in the dataset increases or decreases after training.

    .. math::
        DPPL = \frac{\hat{n_a}^{(1)}}{n_a}-\frac{\hat{n_d}^{(1)}}{n_d}

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return Returns Difference in Positive Proportions, based on predictions rather than labels
    """
    return common.DPL(feature, sensitive_facet_index, positive_predicted_label_index)


@registry.posttraining
def DI(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Disparate Impact (DI)

    Measures adverse effects by the model predictions with respect to labels on different groups selected by
    the facet.

    .. math::
        DI = \frac{\frac{\hat{n_a}^{(1)}}{n_a}}{\frac{\hat{n_d}^{(1)}}{n_d}}

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return Returns disparate impact, the ratio between positive proportions, based on predicted labels
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(positive_predicted_label_index.dtype == bool, "positive_predicted_label_index must be of type bool")

    na1hat = len(feature[positive_predicted_label_index & (~sensitive_facet_index)])
    na = len(feature[~sensitive_facet_index])
    if na == 0:
        raise ValueError("Negated facet set is empty")
    qa = na1hat / na
    nd1hat = len(feature[positive_predicted_label_index & sensitive_facet_index])
    nd = len(feature[sensitive_facet_index])
    if nd == 0:
        raise ValueError("Facet set is empty")
    qd = nd1hat / nd
    return divide(qd, qa)


@registry.posttraining
def DCA(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Difference in Conditional Acceptance (DCA)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return Difference in Conditional Acceptance between advantaged and disadvantaged classes
    """
    dca, _ = common.DCO(feature, sensitive_facet_index, positive_label_index, positive_predicted_label_index)
    return dca


@registry.posttraining
def DCR(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Difference in Conditional Rejection (DCR)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return Difference in Conditional Rejection between advantaged and disadvantaged classes
    """
    _, dcr = common.DCO(feature, sensitive_facet_index, positive_label_index, positive_predicted_label_index)
    return dcr


@registry.posttraining
def RD(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Recall Difference (RD)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return Recall Difference between advantaged and disadvantaged classes
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(positive_label_index.dtype == bool, "positive_label_index must be of type bool")
    require(positive_predicted_label_index.dtype == bool, "positive_predicted_label_index must be of type bool")

    if len(feature[sensitive_facet_index]) == 0:
        raise ValueError("Facet set is empty")
    if len(feature[~sensitive_facet_index]) == 0:
        raise ValueError("Negated Facet set is empty")

    TP_a = len(feature[positive_label_index & positive_predicted_label_index & (~sensitive_facet_index)])
    FN_a = len(feature[positive_label_index & (~positive_predicted_label_index) & (~sensitive_facet_index)])

    rec_a = divide(TP_a, TP_a + FN_a)

    TP_d = len(feature[positive_label_index & positive_predicted_label_index & sensitive_facet_index])
    FN_d = len(feature[positive_label_index & (~positive_predicted_label_index) & sensitive_facet_index])

    rec_d = divide(TP_d, TP_d + FN_d)

    rd = rec_a - rec_d

    if rec_a == rec_d and rec_a == INFINITY:
        rd = 0
    return rd


@registry.posttraining
def DAR(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Difference in Acceptance Rates (DAR)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return Difference in Acceptance Rates
    """

    dar, _ = common.DLR(feature, sensitive_facet_index, positive_label_index, positive_predicted_label_index)
    return dar


@registry.posttraining
def DRR(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Difference in Rejection Rates (DRR)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return Difference in Rejection Rates
    """
    _, drr = common.DLR(feature, sensitive_facet_index, positive_label_index, positive_predicted_label_index)
    return drr


@registry.posttraining
def AD(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Accuracy Difference (AD)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return Accuracy Difference between advantaged and disadvantaged classes
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(positive_label_index.dtype == bool, "positive_label_index must be of type bool")
    require(positive_predicted_label_index.dtype == bool, "positive_predicted_label_index must be of type bool")

    if len(feature[sensitive_facet_index]) == 0:
        raise ValueError("Facet set is empty")
    if len(feature[~sensitive_facet_index]) == 0:
        raise ValueError("Negated Facet set is empty")

    idx_tp_a = positive_label_index & positive_predicted_label_index & ~sensitive_facet_index
    TP_a = len(feature[idx_tp_a])
    idx_fp_a = ~positive_label_index & positive_predicted_label_index & ~sensitive_facet_index
    FP_a = len(feature[idx_fp_a])
    idx_fn_a = positive_label_index & ~positive_predicted_label_index & ~sensitive_facet_index
    FN_a = len(feature[idx_fn_a])
    idx_tn_a = ~positive_label_index & ~positive_predicted_label_index & ~sensitive_facet_index
    TN_a = len(feature[idx_tn_a])

    total_a = TP_a + TN_a + FP_a + FN_a
    acc_a = divide(TP_a + TN_a, total_a)

    idx_tp_d = positive_label_index & positive_predicted_label_index & sensitive_facet_index
    TP_d = len(feature[idx_tp_d])
    idx_fp_d = ~positive_label_index & positive_predicted_label_index & sensitive_facet_index
    FP_d = len(feature[idx_fp_d])
    idx_fn_d = positive_label_index & ~positive_predicted_label_index & sensitive_facet_index
    FN_d = len(feature[idx_fn_d])
    idx_tn_d = ~positive_label_index & ~positive_predicted_label_index & sensitive_facet_index
    TN_d = len(feature[idx_tn_d])

    total_d = TP_d + TN_d + FP_d + FN_d
    acc_d = divide(TP_d + TN_d, total_d)

    ad = acc_a - acc_d
    if acc_a == acc_d and acc_a == INFINITY:
        ad = 0.0
    return ad


@registry.posttraining
def CDDPL(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_predicted_label_index: pd.Series,
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
    :return the weighted average of demographic disparity on all subgroups
    """
    return common.CDD(feature, sensitive_facet_index, positive_predicted_label_index, group_variable)


@registry.posttraining
def TE(
    feature: pd.Series,
    sensitive_facet_index: pd.Series,
    positive_label_index: pd.Series,
    positive_predicted_label_index: pd.Series,
) -> float:
    r"""
    Treatment Equality (TE)

    :param feature: input feature
    :param sensitive_facet_index: boolean column indicating sensitive group
    :param positive_label_index: boolean column indicating positive labels
    :param positive_predicted_label_index: boolean column indicating positive predicted labels
    :return returns the difference in ratios between false negatives and false positives for the advantaged
        and disadvantaged classes
    """
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(positive_label_index.dtype == bool, "positive_label_index must be of type bool")
    require(positive_predicted_label_index.dtype == bool, "positive_predicted_label_index must be of type bool")

    if len(feature[sensitive_facet_index]) == 0:
        raise ValueError("Facet set is empty")
    if len(feature[~sensitive_facet_index]) == 0:
        raise ValueError("Negated Facet set is empty")

    FP_a = len(feature[(~positive_label_index) & positive_predicted_label_index & (~sensitive_facet_index)])
    FN_a = len(feature[positive_label_index & (~positive_predicted_label_index) & (~sensitive_facet_index)])
    FP_d = len(feature[(~positive_label_index) & positive_predicted_label_index & sensitive_facet_index])
    FN_d = len(feature[positive_label_index & (~positive_predicted_label_index) & sensitive_facet_index])

    tau_a = divide(FN_a, FP_a)
    tau_d = divide(FN_d, FP_d)

    te = tau_d - tau_a

    if tau_a == tau_d and tau_a == INFINITY:
        te = 0.0

    return te


def FlipSet_pos(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] > predicted_labels[i]])


def FlipSet_neg(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] < predicted_labels[i]])


def FlipSet(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] != predicted_labels[i]])


@registry.posttraining
def FT(df: pd.DataFrame, sensitive_facet_index: pd.Series, positive_predicted_label_index: pd.Series) -> float:
    r"""
    Flip Test (FT)

    The Flip Test(FT) is an approximation of the test described in (Black et. al paper) to apply for tabular data. In this
    test, we train a k-Nearest Neighbors(k-NN) algorithm on the advantaged samples, run prediction on disadvantaged samples,
    and compute FT metric FT = (FTp - FTn)/ number of disadvangated samples where FTp is the number samples that flipped
    from negative to positive, and FTn is the number samples that flipped from positive to negative.

    :param df: the dataset, excluding facet and label columns
    :param sensitive_facet_index: boolean facet column indicating sensitive group
    :param positive_predicted_label_index: boolean column indicating predicted labels
    :return FT metric
    """
    # FlipTest - binary case
    # a = adv facet, d = disadv facet
    require(sensitive_facet_index.dtype == bool, "sensitive_facet_index must be of type bool")
    require(positive_predicted_label_index.dtype == bool, "positive_predicted_label_index must be of type bool")

    if len(df[sensitive_facet_index]) == 0:
        raise ValueError("Facet set is empty")
    if len(df[~sensitive_facet_index]) == 0:
        raise ValueError("Negated Facet set is empty")
    if len(df.columns) != len(df.select_dtypes([np.number, bool]).columns):
        raise ValueError("FlipTest does not support non-numeric columns")

    dataset = np.array(df)

    data_a = (
        [el for idx, el in enumerate(dataset) if ~sensitive_facet_index.iat[idx]],
        [el for idx, el in enumerate(positive_predicted_label_index) if ~sensitive_facet_index.iat[idx]],
        [el for idx, el in enumerate(sensitive_facet_index) if ~sensitive_facet_index.iat[idx]],
    )
    data_d = (
        [el for idx, el in enumerate(dataset) if sensitive_facet_index.iat[idx]],
        [el for idx, el in enumerate(positive_predicted_label_index) if sensitive_facet_index.iat[idx]],
        [el for idx, el in enumerate(sensitive_facet_index) if sensitive_facet_index.iat[idx]],
    )
    # Set KNN neighbors to 1 if samples less than 10
    # Used at prediction to have enough samples for neighbors
    n_neighbors = FT_DEFAULT_NEIGHBOR if len(data_a[0]) > FT_SAMPLES_COUNT_THRESHOLD else FT_MIN_NEIGHBOR
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

    FTd = divide(len(FS_pos) - len(FS_neg), len(data_d[0]))

    return FTd
