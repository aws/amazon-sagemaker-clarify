import logging
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

log = logging.getLogger(__name__)

INFINITE = float("inf")  # Default return value for all metrics to avoid division by zero errors


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
        raise ValueError("DPPL: Negated facet set is empty")

    qa = na1hat / na
    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])
    nd = len(facet[facet])

    if nd == 0:
        raise ValueError("DPPL: facet set is empty")

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
        raise ValueError("DI: Negated facet set is empty")

    nd1hat = len(predicted_labels[(predicted_labels) & (facet)])
    nd = len(facet[facet])

    if nd == 0:
        raise ValueError("DI: Facet set is empty")

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
        raise ValueError("DCO: Facet set is empty")
    if len(facet[~facet]) == 0:
        raise ValueError("DCO: Negated Facet set is empty")

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

    if ca == cd and ca == INFINITE:
        dca = 0
    if rr_a == rr_d and rr_a == INFINITE:
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
        raise ValueError("RD: Facet set is empty")
    if len(facet[~facet]) == 0:
        raise ValueError("RD: Negated Facet set is empty")

    TP_a = len(labels[(labels) & (predicted_labels) & (~facet)])
    FN_a = len(labels[(labels) & (~predicted_labels) & (~facet)])

    rec_a = TP_a / (TP_a + FN_a) if TP_a + FN_a != 0 else INFINITE

    TP_d = len(labels[(labels) & (predicted_labels) & (facet)])
    FN_d = len(labels[(labels) & (~predicted_labels) & (facet)])

    rec_d = TP_d / (TP_d + FN_d) if TP_d + FN_d != 0 else INFINITE

    rd = rec_a - rec_d

    if rec_a == rec_d and rec_a == INFINITE:
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
        raise ValueError("DLR: Facet set is empty")
    if len(facet[~facet]) == 0:
        raise ValueError("DLR: Negated Facet set is empty")

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

    if ar_a == ar_d and ar_a == INFINITE:
        dar = 0
    if rr_a == rr_d and rr_a == INFINITE:
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
        raise ValueError("AD: Facet set is empty")
    if len(facet[~facet]) == 0:
        raise ValueError("AD: Negated Facet set is empty")

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

    if acc_a == acc_d and acc_a == INFINITE:
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
        raise ValueError("TE: Facet set is empty")
    if len(facet[~facet]) == 0:
        raise ValueError("TE: Negated Facet set is empty")

    FP_a = len(labels[(~labels) & (predicted_labels) & (~facet)])
    FN_a = len(labels[(labels) & (~predicted_labels) & (~facet)])
    FP_d = len(labels[(~labels) & (predicted_labels) & (facet)])
    FN_d = len(labels[(labels) & (~predicted_labels) & (facet)])

    tau_a = FN_a / FP_a if FP_a != 0 else INFINITE
    tau_d = FN_d / FP_d if FP_d != 0 else INFINITE

    te = tau_d - tau_a

    if tau_a == tau_d and tau_a == INFINITE:
        te = 0

    return te


def FlipSet_pos(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] > predicted_labels[i]])


def FlipSet_neg(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] < predicted_labels[i]])


def FlipSet(dataset: np.array, labels: np.array, predicted_labels: np.array) -> np.array:
    return np.array([dataset[i] for i in range(len(dataset)) if labels[i] != predicted_labels[i]])


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
