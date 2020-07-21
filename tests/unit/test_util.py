from famly.util import pdf, pdfs_aligned_nonzero
import numpy as np


def test_pdf():
    xs = [1, 2, 2, 3, 3, 3]
    pdf_res = dict(pdf(xs))
    assert pdf_res[1] == 1 / 6
    assert pdf_res[2] == 2 / 6
    assert pdf_res[3] == 3 / 6


def test_pdfs_aligned():
    (pa, pb) = pdfs_aligned_nonzero([1, 2, 3], [4, 5, 5])
    assert np.array_equal(pa, [])
    assert np.array_equal(pb, [])

    (pa, pb) = pdfs_aligned_nonzero([1, 2, 3], [1, 1, 4, 5, 5])
    assert np.array_equal(pa, [1 / 3])
    assert np.array_equal(pb, [2 / 5])
