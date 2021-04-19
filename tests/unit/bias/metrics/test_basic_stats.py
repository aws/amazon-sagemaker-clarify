from smclarify.bias.metrics import basic_stats

from .test_metrics import dfBinary

from pytest import approx


(dfB, dfB_label, dfB_pos_label_idx, dfB_pred_label, dfB_pos_pred_label_idx) = dfBinary()


def test_proportion():
    sensitive_facet_index = dfB[0] == "F"
    assert basic_stats.proportion(sensitive_facet_index) == approx(7 / 12)

    sensitive_facet_index = dfB[0] == "M"
    assert basic_stats.proportion(sensitive_facet_index) == approx(5 / 12)


def test_confusion_matrix():
    # Binary Facet, Binary Label
    sensitive_facet_index = dfB[0] == "F"
    TP = approx(2 / 7.0)
    TN = approx(2 / 7.0)
    FP = approx(2 / 7.0)
    FN = approx(1 / 7.0)
    assert [TP, FP, FN, TN] == basic_stats.confusion_matrix(
        dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx
    )

    sensitive_facet_index = dfB[0] == "M"
    TP = 0
    TN = approx(1 / 5.0)
    FP = approx(2 / 5.0)
    FN = approx(2 / 5.0)
    assert [TP, FP, FN, TN] == basic_stats.confusion_matrix(
        dfB[0], sensitive_facet_index, dfB_pos_label_idx, dfB_pos_pred_label_idx
    )
