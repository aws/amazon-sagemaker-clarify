from typing import NamedTuple, Optional, List, Any
import pandas as pd
import pytest

from smclarify.bias.metrics.common import DataType, series_datatype, ensure_series_data_type


class EnsureSeriesDataTypeInput(NamedTuple):
    data: pd.Series
    values: Optional[List[Any]] = None


class EnsureSeriesDataTypeOutput(NamedTuple):
    data_type: DataType
    new_data: pd.Series


def ensure_series_data_type_test_cases():
    test_cases = []

    # categorical data series
    data = pd.Series([1, 2, 3]).astype("category")
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data)
    test_cases.append([function_input, function_output])

    # categorical values
    data = pd.Series([1, 2, 3])
    function_input = EnsureSeriesDataTypeInput(data=data, values=[1, 2, 3])
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # floating data series
    data = pd.Series([1.0, 2.0, 3.0])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CONTINUOUS, new_data=data)
    test_cases.append([function_input, function_output])

    # object data series, can NOT be converted to numeric
    data = pd.Series(["a", "b", "c"])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # object data series, can be converted to numeric, and uniqueness is high
    data = pd.Series(["1", "2", "3"])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CONTINUOUS, new_data=pd.to_numeric(data))
    test_cases.append([function_input, function_output])

    # object data series, can be converted to numeric, but uniqueness is low
    data = ["1"] * 40
    data.append("2")
    data = pd.Series(data)
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # integer data series, uniqueness is high
    data = pd.Series([1, 2, 3])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CONTINUOUS, new_data=data)
    test_cases.append([function_input, function_output])

    # integer data series, uniqueness is low
    data = [1] * 40
    data.append(2)
    data = pd.Series(data)
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # boolean data series
    data = pd.Series([True, False, True])
    function_input = EnsureSeriesDataTypeInput(data=data)
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CATEGORICAL, new_data=data.astype("category"))
    test_cases.append([function_input, function_output])

    # threshold intervals should be continuous
    data = pd.Series([1, 2, 3])
    function_input = EnsureSeriesDataTypeInput(data=data, values=["[2,3]"])
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CONTINUOUS, new_data=data)
    test_cases.append([function_input, function_output])

    # threshold intervals should be continuous
    data = pd.Series([1, 2, 3])
    function_input = EnsureSeriesDataTypeInput(data=data, values=["(1,2]", "(2,3]"])
    function_output = EnsureSeriesDataTypeOutput(data_type=DataType.CONTINUOUS, new_data=data)
    test_cases.append([function_input, function_output])

    return test_cases


@pytest.mark.parametrize("function_input,function_output", ensure_series_data_type_test_cases())
def test_ensure_series_data_type(function_input, function_output):
    # Test the series_datatype function by the way
    data_type = series_datatype(*function_input)
    assert data_type == function_output.data_type
    # Test the ensure_series_data_type
    data_type, new_data = ensure_series_data_type(*function_input)
    assert data_type == function_output.data_type
    assert new_data.equals(function_output.new_data)
