import pandas as pd
import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def DPL(feature: pd.Series, facet: pd.Series, label: pd.Series, positive_label_index: pd.Series) -> float:
    facet = facet.astype(bool)
    positive_label_index = positive_label_index.astype(bool)
    na = len(feature[~facet])
    nd = len(feature[facet])
    na_pos = len(label[~facet & positive_label_index])
    nd_pos = len(label[facet & positive_label_index])
    if na == 0:
        raise ValueError("DPL: negative facet set is empty.")
    if nd == 0:
        raise ValueError("DPL: facet set is empty.")
    qa = na_pos / na
    qd = nd_pos / nd
    dpl = qa - qd
    return dpl


def CDD(feature: pd.Series, facet: pd.Series, label_index: pd.Series, group_variable: pd.Series) -> float:
    """
    :param feature: input feature
    :param facet: boolean column indicating sensitive group
    :param label_index: boolean column indicating positive labels or predicted labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    if group_variable is None or group_variable.empty:
        return float("NAN")
    facet = facet.astype(bool)
    label_index = label_index.astype(bool)
    unique_groups = np.unique(group_variable)

    # Global demographic disparity (DD)]
    denomA = len(feature[label_index])

    if denomA == 0:
        raise ValueError("CDD: No positive labels in set")
    denomD = len(feature[~label_index])

    if denomD == 0:
        raise ValueError("CDD: No negative labels in set")

    # Conditional demographic disparity (CDD)
    CDD = []
    counts = []
    for subgroup_variable in unique_groups:
        counts = np.append(counts, len(group_variable[group_variable == subgroup_variable]))
        numA = len(label_index[label_index & facet & (group_variable == subgroup_variable)])
        denomA = len(feature[label_index & (group_variable == subgroup_variable)])
        A = numA / denomA if denomA != 0 else 0
        numD = len(label_index[(~label_index) & facet & (group_variable == subgroup_variable)])
        denomD = len(feature[(~label_index) & (group_variable == subgroup_variable)])
        D = numD / denomD if denomD != 0 else 0
        CDD = np.append(CDD, D - A)

    wtd_mean_CDD = np.sum(counts * CDD) / np.sum(counts)

    return wtd_mean_CDD
