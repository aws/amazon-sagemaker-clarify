![Python package](https://github.com/aws/famly/workflows/Python%20package/badge.svg)
![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg?style=flat)

# FAMLy

Fairness Aware Machine Learning

Bias detection and mitigation for datasets and models.

## Terminology

### Facet
A facet is column or feature that will be used to measure bias against. A facet can have value(s) that designates that sample as "***protected***".

### Label
The label is a column or feature which is the target for training a machine learning model. The label can have value(s) that designates that sample as having a "***positive***" outcome.

### Bias measure
A bias measure is a function that returns a bias metric.

### Bias metric
A bias metric is a numerical value indicating the level of bias detected as determined by a particular bias measure.

### Bias report
A collection of bias metrics for a given dataset or a combination of a dataset and model.

## Development

```
virtualenv -p(which python3) venv
source venv/bin/activate.fish
pip install -e .[test]
pytest --pspec
pre-commit install && pre-commit run --all-files
```

Always run `pre-commit run --all-files` before commit.


For running unit tests, do `./test.sh` or `pytest --pspec`. If you are using PyCharm, and cannot see the green run button next to the tests, open `Preferences` -> `Tools` -> `Python Integrated tools`, and set default test runner to `pytest`.

test validationt
