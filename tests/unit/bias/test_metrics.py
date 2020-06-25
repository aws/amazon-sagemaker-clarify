import pandas as pd
import numpy as np
from ....src.famly.bias.metrics import CI, DPL, KL, JS, LPnorm, TVD, KS, CDD, DPPL, DI, DCA, DCR, RD, DRR, PD, AD, TE, FT, metric_one_vs_all
import pytest
from pytest import approx


def dfBinary():
    """
    :return: dataframe with one column which contains Binary categorical data (length 12)
    """
    data = [['M'], ['F'], ['F'], ['M'], ['F'], ['M'], ['F'], ['F'], ['M'], ['M'], ['F'], ['F']]

    df = pd.DataFrame(data)
    return df

def dfMulticategory():
    """
    :return: dataframe with one column which contains multicategorical data (length 24)
    """
    data = [['M'], ['O'], ['M'], ['M'], ['F'], ['O'], ['O'], ['F'], ['M'], ['M'], ['F'], ['F'], ['O'], ['F'], ['M'], ['F'], ['O'], ['F'], ['M'], ['M'], ['F'], ['F'], ['O'], ['O']]
            #[1,     0,     0,      1,     0,     1,     0,     1,      1,    0,    1,       0,     1,     0,     1,    0,     0,     0,      1,    1,     0,     0,     1,     1]
    df = pd.DataFrame(data)
    return df

def dfContinuous():
    """
    :return: dataframe with one column which contains continuous data (length 12)
    """
    data = pd.Series([1.55255404, 1.87128923, 1.82640675, 0.48706083, 0.21833644,
                      0.45007763, 0.47457823, 1.5346789 , 1.61042132, 1.87130261,
                      1.97768247, 1.05499183])
    df = pd.DataFrame(data)
    return df

dfB = dfBinary()
dfM = dfMulticategory()
dfC = dfContinuous()

def test_ci():
    """test class imbalance"""

    #Binary Facet, Binary Label

    facet = dfB[0] == 'F'
    positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    assert CI(dfB[0], facet, positive_label_index) == approx(-1 / 6)

    facet = dfB[0] == 'M'
    assert CI(dfB[0], facet, positive_label_index) == approx(1 / 6)

    #Continuous Facet, Binary Label
    facet = dfC[0] > 1.0
    assert CI(dfC[0], facet, positive_label_index) == approx(-1 / 3)

    facet = dfC[0] < 1.0
    assert CI(dfC[0], facet, positive_label_index) == approx(1 / 3)

    #Multicategory Facet, Binary Label
    positive_label_index = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]

    response = metric_one_vs_all(CI, dfM[0], dfM[0], positive_label_index)

    assert response['M'] == approx(1 / 3) #6 / 8
    assert response['F'] == approx(1 / 4)  #2 / 9
    assert response['O'] == approx(5 / 12) #4 / 7

def test_dpl():

    #Binary Facet, Binary Label
    facet = dfB[0] == 'F'
    positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    assert DPL(dfB[0], facet, positive_label_index) == approx((2 / 5 - 4 / 7) / (4 / 7 + 2 / 5))

    facet = dfB[0] == 'M'
    assert DPL(dfB[0], facet, positive_label_index) == approx((4 / 7 - 2 / 5) / (4 / 7 + 2 / 5))

    #Multicategory Facet, Binary Label
    positive_label_index = [1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]

    response = metric_one_vs_all(DPL, dfM[0], dfM[0], positive_label_index)
    assert response['M'] == approx((6 / 16 - 6 / 8) / (6 / 16 + 6 / 8))
    assert response['F'] == approx((10 / 15 - 2 / 9) / (10 / 15 + 2 / 9))
    assert response['O'] == approx((8 / 17 - 4 / 7) / (8 / 17 + 4 / 7))

def test_KL():

    facet = dfB[0] == 'F'
    positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    assert KL(dfB[0], facet, positive_label_index) == approx(0.059213364)

    facet = dfB[0] == 'M'
    assert KL(dfB[0], facet, positive_label_index) == approx(0.059611866)

def test_FT():
    facet = dfB[0] == 'F'
    positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    assert FT()
# if __name__ == "__main__":
#     facet = dfB[0] == 'F'
#     positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
#     print('KL Binary F, ', KL(dfB[0], facet, positive_label_index))
#
#     facet = dfB[0] == 'M'
#     print('KL Binary M, ', KL(dfB[0], facet, positive_label_index))

#     for metric in [CI, DPL, KL, JS, LPnorm, TVD, KS, CDD]:
#         print('\n\n' + metric.__name__)
#         metric_name = metric.__name__
#         print('Binary facet, Binary Label')
#         x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F'])
#
#         facet = x == 'F'
#
#         positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
#
#         group_variable = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2])
#
#         if metric_name != 'CDD':
#             print(metric(x, facet, positive_label_index))
#         else:
#             print(metric(x, facet, positive_label_index, group_variable))
#
#         print('Continuous facet, Binary Label')
#         x = pd.Series(np.random.uniform(0, 2, 12))
#
#         facet = x > 1.0
#
#         positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
#
#         group_variable = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2])
#
#         if metric_name != 'CDD':
#             print(metric(x, facet, positive_label_index))
#         else:
#             print(metric(x, facet, positive_label_index, group_variable))
#
#         print('Multicategory facet, Binary Label')
#         x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'O', 'O', 'O', 'O'])
#
#         positive_label_index = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1])
#
#         group_variable = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2, 0, 1, 1, 2])
#
#         if metric_name != 'CDD':
#             print(metric_one_vs_all(metric, x, x, positive_label_index=positive_label_index))
#         else:
#             print(metric_one_vs_all(metric, x, x, positive_label_index=positive_label_index, group_variable=group_variable))
#
#         print('Binary facet, Multicategory Label')
#
#         x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F'])
#
#         facet = x == 'F'
#
#         labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1])
#
#         group_variable = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2])
#
#         if metric_name != 'CDD':
#
#             print(metric_one_vs_all(metric, x, x, labels=labels))
#
#         else:
#
#             print(metric_one_vs_all(metric, x, x, labels=labels, group_variable=group_variable))
#
#         print('Continuous facet, Multicategory Label')
#
#         x = pd.Series(np.random.uniform(0, 2, 12))
#
#         facet = x > 1.0
#
#         labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1])
#
#         group_variable = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2])
#
#         if metric_name != 'CDD':
#             print(metric_one_vs_all(metric, x, x, labels=labels))
#         else:
#             print(metric_one_vs_all(metric, x, x, labels=labels, group_variable=group_variable))
#
#         print('Multicategory Facet, Multicategory Label')
#
#         x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'O', 'O', 'O', 'O'])
#
#         labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1, 2, 1, 0, 1])
#
#         group_variable = pd.Series([0, 2, 0, 0, 1, 1, 2, 1, 0, 1, 0, 2, 0, 1, 2, 0])
#
#         if metric_name != 'CDD':
#             print(metric_one_vs_all(metric, x, x, labels=labels))
#         else:
#             print(metric_one_vs_all(metric, x, x, labels=labels, group_variable=group_variable))
#
#
#     for metric in [DPPL, DI, DCO, RD, DLR, PD, AD, TE]:
#         print('\n\n' + metric.__name__)
#         metric_name = metric.__name__
#         print('Binary facet, Binary Label')
#         x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F'])
#
#         facet = x == 'F'
#
#         labels = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
#         predicted_labels = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
#
#         if metric in [DPPL, DI]:
#             print(metric(x, facet, predicted_labels))
#         else:
#             print(metric(x, facet, labels, predicted_labels))
#
#         print('Continuous facet, Binary Label')
#         x = pd.Series(np.random.uniform(0, 2, 12))
#
#         facet = x > 1.0
#
#         labels = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
#         predicted_labels = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0])
#
#
#         if metric in [DPPL, DI]:
#             print(metric(x, facet, predicted_labels))
#         else:
#             print(metric(x, facet, labels, predicted_labels))
#
#         print('Multicategory facet, Binary Label')
#         x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'O', 'O', 'O', 'O'])
#
#         # facet = x == 'F'
#
#         labels = pd.Series([0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
#         predicted_labels = pd.Series([1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1])
#
#
#         #TODO: Need to change to metric_one_vs_all
#         if metric in [DPPL, DI]:
#             print(metric_one_vs_all(metric, x, facet, predicted_labels = predicted_labels))
#         else:
#             print(metric_one_vs_all(metric, x, facet, labels = labels, predicted_labels=predicted_labels))
#
#         print('Binary facet, Multicategory Label')
#
#         x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F'])
#
#         facet = x == 'F'
#
#         labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1])
#
#         predicted_labels = pd.Series([0, 2, 1, 2, 1, 0, 1, 1, 2, 2, 1, 0])
#
#         if metric in [DPPL, DI]:
#             print(metric_one_vs_all(metric, x, facet, predicted_labels = predicted_labels))
#         else:
#             print(metric_one_vs_all(metric, x, facet, labels = labels, predicted_labels=predicted_labels))
#
#         print('Continuous facet, Multicategory Label')
#
#         x = pd.Series(np.random.uniform(0, 2, 12))
#
#         facet = x > 1.0
#
#         labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1])
#
#         predicted_labels = pd.Series([0, 2, 1, 2, 1, 0, 1, 1, 2, 2, 1, 0])
#
#         if metric in [DPPL, DI]:
#             print(metric_one_vs_all(metric, x, facet, predicted_labels = predicted_labels))
#         else:
#             print(metric_one_vs_all(metric, x, facet, labels = labels, predicted_labels=predicted_labels))
#
#         print('Multicategory Facet, Multicategory Label')
#
#         x = pd.Series(['M', 'F', 'F', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'F', 'O', 'O', 'O', 'O'])
#
#         labels = pd.Series([0, 1, 1, 2, 1, 0, 0, 1, 2, 2, 0, 1, 2, 1, 0, 1])
#         predicted_labels = pd.Series([0, 2, 1, 2, 1, 0, 1, 1, 2, 2, 1, 0, 2, 1, 0, 1])
#
#         if metric in [DPPL, DI]:
#             print(metric_one_vs_all(metric, x, x, predicted_labels = predicted_labels))
#         else:
#             print(metric_one_vs_all(metric, x, x, labels = labels, predicted_labels=predicted_labels))
#
#     X = np.array([[0, 0, 0, 0, 1, 1, 1],
#                   [1, 0, 0, 0, 1, 1, 1],
#                   [1, 0, 0, 0, 1, 1, 1],
#                   [0, 0, 0, 0, 1, 1, 1],
#                   [1, 0, 0, 0, 1, 1, 1],
#                   [0, 0, 0, 0, 1, 1, 1],
#                   [1, 0, 0, 0, 1, 1, 1],
#                   [1, 0, 0, 0, 1, 1, 1],
#                   [0, 0, 0, 0, 1, 1, 1],
#                   [1, 0, 0, 0, 1, 1, 1],
#                   [1, 0, 0, 0, 1, 1, 1],
#                   [1, 0, 0, 0, 1, 1, 1],
#                   [1, 0, 0, 0, 1, 1, 1],
#                   [0, 0, 0, 0, 1, 1, 1],
#                   [0, 0, 0, 0, 1, 1, 1],
#                   [0, 0, 0, 0, 1, 1, 1],
#                   [0, 0, 0, 0, 1, 1, 1]])
#
#     # print('\n\nFT - cont.')
#     # print(FT(X, Xr, y, yhat, pred_type=prediction_type[2]))

# def FT_multicategory(dataset: np.ndarray, facet: np.array, y: np.array, predicted_labels: np.array) -> dict:
#     # FlipTest - multicategory case
#     y_categories = set(predicted_labels)
#     res = {}
#     for y_true in y_categories:
#         y_tmp = collapse_to_binary(y, [y_true])
#         y_hat_tmp = collapse_to_binary(predicted_labels, [y_true])
#         res[y_true] = FT_binary(dataset, facet, y_tmp, y_hat_tmp)[1]
#     return res


# def FT_continuous(dataset: np.ndarray, facet: np.array, y: np.array, predicted_labels: np.array) -> dict:
#     # FlipTest - continuous case
#     # FlipTest - binary case
#     # a = adv facet, d = disadv facet
#     data_a = ([el for idx, el in enumerate(dataset) if facet[idx] == 0],
#               [el for idx, el in enumerate(predicted_labels) if facet[idx] == 0],
#               [el for idx, el in enumerate(facet) if facet[idx] == 0])
#     data_d = ([el for idx, el in enumerate(dataset) if facet[idx] == 1],
#               [el for idx, el in enumerate(predicted_labels) if facet[idx] == 1],
#               [el for idx, el in enumerate(facet) if facet[idx] == 1])
#
#     knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
#                               metric='minkowski', metric_params=None, n_jobs=None)
#
#     # kNN method over a with Labels from the model
#     knn.fit(np.array(data_a[0]), np.array(data_a[1]))
#     # kNN prediction over d, first finding the index of the k neighbors for each example over d
#     kneigh = knn.kneighbors(data_d[0])[1]
#     # print(kneigh)
#     # Finding the average value of the label of the k neighbors for each example
#     data1_a_array = np.array(data_a[1])
#     data1_a_y_model_average = np.mean(data1_a_array[kneigh], axis=1)
#     # print(tmp)
#     d_y_if_a = data1_a_y_model_average
#     # Model predictions over the same test d
#     d_y_model = data_d[1]
#     # print(data_d[1], d_y_model, d_y_if_a)
#
#     FS_pos = FlipSet_pos(dataset=data_d[1], y=d_y_model, predicted_labels=d_y_if_a)
#     FS_neg = FlipSet_neg(dataset=data_d[1], y=d_y_model, predicted_labels=d_y_if_a)
#     FS = FlipSet(dataset=data_d[1], y=d_y_model, predicted_labels=d_y_if_a)
#
#     FTd = (len(FS_pos) - len(FS_neg)) / len(data_d[0])
#     FTs = len(FS) / len(data_d[0])
#
#     return {'y': [FTd, FTs]}


# def FT(dataset: np.ndarray, facet: np.array, y: np.array, predicted_labels: np.array, pred_type: tuple) -> dict:
#     # FlipTest
#     if pred_type[0] == 0:
#         return FT_binary(dataset, facet, y, predicted_labels)
#     elif pred_type[0] == 1:
#         return FT_multicategory(dataset, facet, y, predicted_labels)
#     else:
#         return FT_continuous(dataset, facet, y, predicted_labels)

