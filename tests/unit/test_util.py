from famly.util import pdf


def test_pdf():
    xs = [1, 2, 2, 3, 3, 3]
    pdf_res = dict(pdf(xs))
    assert pdf_res[1] == 1 / 6
    assert pdf_res[2] == 2 / 6
    assert pdf_res[3] == 3 / 6
