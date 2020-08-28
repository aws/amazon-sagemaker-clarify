from famly.util import compute_pdf, compute_aligned_pdfs
import numpy as np


def test_pdf():
    xs = [1, 2, 2, 3, 3, 3]
    pdf_res = dict(compute_pdf(xs))
    assert pdf_res[1] == 1 / 6
    assert pdf_res[2] == 2 / 6
    assert pdf_res[3] == 3 / 6


def test_pdfs_aligned():
    (pa, pb) = compute_aligned_pdfs([1, 2, 3], [4, 5, 5])
    assert np.array_equal(pa, [])
    assert np.array_equal(pb, [])

    (pa, pb) = compute_aligned_pdfs([1, 2, 3], [1, 1, 4, 5, 5])
    assert np.array_equal(pa, [1 / 3])
    assert np.array_equal(pb, [2 / 5])

    (pa, pb, pc) = compute_aligned_pdfs([1, 2, 3], [1, 1, 4, 5, 5], [1, 1, 1, 2])
    assert np.array_equal(pa, [1 / 3])
    assert np.array_equal(pb, [2 / 5])
    assert np.array_equal(pc, [3 / 4])
