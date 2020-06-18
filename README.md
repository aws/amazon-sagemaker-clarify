# FAMLy

Fairness Aware Machine Learning

Bias detection and mitigation for datasets and models.

## Terminology

### Facet
A facet is a set of sensitive column or feature values that will be used to measure bias against. A facet can have value(s) that designates that sample as "***sensitive***".

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
pip install -e .[test]
```

Run `pre-commit install && pre-commit run --all-files` before commit.
