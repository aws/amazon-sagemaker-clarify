from typing import Any
import pandas as pd
import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def DPL(facet: pd.Series, label: pd.Series, positive_label: Any) -> float:
    positive_label_index = label == positive_label
    facet = facet.astype(bool)
    na = len(facet[~facet])
    nd = len(facet[facet])
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


def CDD(facet: pd.Series, label: pd.Series, group_variable: pd.Series) -> float:
    """
    :param facet: boolean column indicating sensitive group
    :param label: boolean column indicating positive labels
    :param group_variable: categorical column indicating subgroups each point belongs to
    :return: the weighted average of demographic disparity on all subgroups
    """
    unique_groups = np.unique(group_variable)
    label = label.astype(bool)
    facet = facet.astype(bool)

    # Global demographic disparity (DD)]
    denomA = len(facet[label])

    if denomA == 0:
        raise ValueError("CDD: No positive labels in set")
    denomD = len(facet[~label])

    if denomD == 0:
        raise ValueError("CDD: No negative labels in set")

    # Conditional demographic disparity (CDD)
    CDD = []
    counts = []
    for subgroup_variable in unique_groups:
        counts = np.append(counts, len(group_variable[group_variable == subgroup_variable]))
        numA = len(label[label & facet & (group_variable == subgroup_variable)])
        denomA = len(facet[label & (group_variable == subgroup_variable)])
        A = numA / denomA if denomA != 0 else 0
        numD = len(label[(~label) & facet & (group_variable == subgroup_variable)])
        denomD = len(facet[(~label) & (group_variable == subgroup_variable)])
        D = numD / denomD if denomD != 0 else 0
        CDD = np.append(CDD, D - A)

    wtd_mean_CDD = np.sum(counts * CDD) / np.sum(counts)

    return wtd_mean_CDD
