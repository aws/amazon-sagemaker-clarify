![Python package](https://github.com/aws/amazon-sagemaker-clarify/workflows/Python%20package/badge.svg)
![Pypi](https://img.shields.io/pypi/v/smclarify.svg?maxAge=60)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat)

# smclarify

Amazon Sagemaker Clarify

Bias detection and mitigation for datasets and models.


# Installation

To install the package from PIP you can simply do:

```
pip install smclarify
```

You can see examples on running the Bias metrics on the notebooks in the [examples folder](https://github.com/aws/amazon-sagemaker-clarify/tree/master/examples).


# Terminology

### Facet
A facet is column or feature that will be used to measure bias against. A facet can have value(s) that designates that sample as "***sensitive***".

### Label
The label is a column or feature which is the target for training a machine learning model. The label can have value(s) that designates that sample as having a "***positive***" outcome.

### Bias measure
A bias measure is a function that returns a bias metric.

### Bias metric
A bias metric is a numerical value indicating the level of bias detected as determined by a particular bias measure.

### Bias report
A collection of bias metrics for a given dataset or a combination of a dataset and model.

# Development

It's recommended that you setup a virtualenv.

```
virtualenv -p(which python3) venv
source venv/bin/activate.fish
pip install -e .[test]
cd src/
../devtool all
```

For running unit tests, do `pytest --pspec`. If you are using PyCharm, and cannot see the green run button next to the tests, open `Preferences` -> `Tools` -> `Python Integrated tools`, and set default test runner to `pytest`.

For Internal contributors, run ```../devtool integ_tests``` after creating virtualenv with the above steps to run the integration tests.
