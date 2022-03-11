# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/
from typing import NamedTuple, Optional, Type

import pytest

from smclarify.bias import Interval, IntervalArray


class IntervalInitInput(NamedTuple):
    left: Optional[float]
    right: Optional[float]
    str_form: str
    closed: str = "both"
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class IntervalInitOutput(NamedTuple):
    str_rep: str


class IntervalError(NamedTuple):
    error: Type[Exception]
    message: str


def interval_init_happy_test_cases():
    test_cases = []

    str_form = "[-inf, inf]"
    function_input = IntervalInitInput(left=None, right=None, str_form=str_form, closed="both")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "(-inf, inf)"
    function_input = IntervalInitInput(left=None, right=None, str_form=str_form, closed="neither")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "(-inf, inf]"
    function_input = IntervalInitInput(left=None, right=None, str_form=str_form, closed="right")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "[-inf, inf)"
    function_input = IntervalInitInput(left=None, right=None, str_form=str_form, closed="left")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "[-inf, ]"
    function_input = IntervalInitInput(left=None, right=None, str_form=str_form, closed="both")
    function_output = IntervalInitOutput(str_rep="[-inf, inf]")
    test_cases.append([function_input, function_output])

    str_form = "[, inf]"
    function_input = IntervalInitInput(left=None, right=None, str_form=str_form, closed="both")
    function_output = IntervalInitOutput(str_rep="[-inf, inf]")
    test_cases.append([function_input, function_output])

    str_form = "[, ]"
    function_input = IntervalInitInput(left=None, right=None, str_form=str_form, closed="both")
    function_output = IntervalInitOutput(str_rep="[-inf, inf]")
    test_cases.append([function_input, function_output])

    str_form = "[13, ]"
    function_input = IntervalInitInput(left=13, right=None, str_form=str_form, closed="both")
    function_output = IntervalInitOutput(str_rep="[13, inf]")
    test_cases.append([function_input, function_output])

    str_form = "[, 13]"
    function_input = IntervalInitInput(left=None, right=13, str_form=str_form, closed="both")
    function_output = IntervalInitOutput(str_rep="[-inf, 13]")
    test_cases.append([function_input, function_output])

    str_form = "[-5, 13]"
    function_input = IntervalInitInput(left=-5, right=13, str_form=str_form, closed="both")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "(-5, 13]"
    function_input = IntervalInitInput(left=-5, right=13, str_form=str_form, closed="right")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "[-5, 13)"
    function_input = IntervalInitInput(left=-5, right=13, str_form=str_form, closed="left")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "(-5, 13)"
    function_input = IntervalInitInput(left=-5, right=13, str_form=str_form, closed="neither")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "(13, 13]"
    function_input = IntervalInitInput(left=13, right=13, str_form=str_form, closed="right")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "[13, 13)"
    function_input = IntervalInitInput(left=13, right=13, str_form=str_form, closed="left")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "[13, 13]"
    function_input = IntervalInitInput(left=13, right=13, str_form=str_form, closed="both")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    str_form = "[13.0, 13.1]"
    function_input = IntervalInitInput(left=13.0, right=13.1, str_form=str_form, closed="both")
    function_output = IntervalInitOutput(str_rep=str_form)
    test_cases.append([function_input, function_output])

    return test_cases


def interval_init_error_cases():
    test_cases = []

    str_form = "(13, 13)"
    function_input = IntervalInitInput(left=13, right=13, str_form=str_form, closed="neither")
    expected_error = IntervalError(error=ValueError, message="Interval has exclusive bounds, but left==right")
    test_cases.append([function_input, expected_error])

    str_form = "(14, 13)"
    function_input = IntervalInitInput(left=14, right=13, str_form=str_form, closed="neither")
    expected_error = IntervalError(error=ValueError, message="left bound cannot be greater than right bound")
    test_cases.append([function_input, expected_error])

    str_form = "(13.001, 13)"
    function_input = IntervalInitInput(left=13.001, right=13, str_form=str_form, closed="neither")
    expected_error = IntervalError(error=ValueError, message="left bound cannot be greater than right bound")
    test_cases.append([function_input, expected_error])

    str_form = "[13, 13]"
    function_input = IntervalInitInput(left=13, right=13, str_form=str_form, closed="unsupportedasdf!!")
    expected_error = IntervalError(
        error=ValueError,
        message="closed=unsupportedasdf!! is not valid.  " "Must be in \['left', 'right', 'both', 'neither'\]",
    )
    test_cases.append([function_input, expected_error])

    return test_cases


def interval_from_string_error_cases():
    test_cases = []

    function_input = "a5,6]"
    expected_error = IntervalError(
        error=ValueError,
        message="interval=a5,6\] is not valid.  " 'Must start with an interval left bound "\[" or "\("',
    )
    test_cases.append([function_input, expected_error])

    function_input = "[5,6b"
    expected_error = IntervalError(
        error=ValueError, message="interval=\[5,6b is not valid.  " 'Must end with an interval right bound "\]" or "\)"'
    )
    test_cases.append([function_input, expected_error])

    function_input = "[5,6,7]"
    expected_error = IntervalError(
        error=ValueError, message="interval=\[5,6,7\] is not valid.  " 'Cannot contain more than one comma ","'
    )
    test_cases.append([function_input, expected_error])

    function_input = "[a,13]"
    expected_error = IntervalError(error=ValueError, message="interval=\[a,13\] is not valid")
    test_cases.append([function_input, expected_error])

    function_input = "[13,b]"
    expected_error = IntervalError(error=ValueError, message="interval=\[13,b\] is not valid")
    test_cases.append([function_input, expected_error])

    return test_cases


@pytest.mark.parametrize("function_input,function_output", interval_init_happy_test_cases())
def test_interval_init_happy_cases(function_input, function_output):
    interval = Interval(function_input.left, function_input.right, function_input.closed)

    if function_input.left is None:
        assert interval.left == float("-inf")
        if interval.left_inclusive:
            assert interval.contains(float("-inf"))
    else:
        assert interval.left == function_input.left
    if function_input.right is None:
        assert interval.right == float("inf")
        if interval.right_inclusive:
            assert interval.contains(float("inf"))
    else:
        assert interval.right == function_input.right

    assert interval.closed == function_input.closed

    closed = function_input.closed
    if closed == "left" or closed == "both":
        assert interval.contains(interval.left)
        assert interval.left_inclusive
    else:
        assert not interval.left_inclusive
        if function_input.left and interval.left != interval.right:
            assert not interval.contains(interval.left)
        elif not function_input.left:
            assert not interval.contains(interval.left)
    if closed == "right" or closed == "both":
        assert interval.contains(interval.right)
        assert interval.right_inclusive
    else:
        assert not interval.right_inclusive
        if function_input.right and interval.left != interval.right:
            assert not interval.contains(interval.right)
        elif not function_input.right:
            assert not interval.contains(interval.right)
    if function_input.left or function_input.right:
        assert interval.contains((interval.left + interval.right) / 2)

    assert function_output.str_rep == str(interval)

    assert Interval.from_string(function_input.str_form) == interval


@pytest.mark.parametrize("function_input,expected_error", interval_init_error_cases())
def test_interval_init_error_cases(function_input, expected_error):
    with pytest.raises(expected_error.error, match=expected_error.message):
        Interval(function_input.left, function_input.right, function_input.closed)


@pytest.mark.parametrize("function_input,expected_error", interval_from_string_error_cases())
def test_interval_from_string_error_cases(function_input, expected_error):
    with pytest.raises(expected_error.error, match=expected_error.message):
        Interval.from_string(function_input)


def test_interval_from_string_min_and_max_values():
    interval = Interval.from_string("[-inf, inf]", min_value=-10, max_value=13)
    assert "[-10, 13]" == str(interval)
    assert interval.left == -10
    assert interval.right == 13


def test_interval__validate_interval_number():
    # success cases
    assert Interval._validate_interval_number(13.0) == 13.0
    assert Interval._validate_interval_number(13) == 13
    assert (
        isinstance(Interval._validate_interval_number("13.0"), float)
        and Interval._validate_interval_number("13.0") == 13.0
    )
    assert isinstance(Interval._validate_interval_number("13"), int) and Interval._validate_interval_number("13") == 13
    assert (
        isinstance(Interval._validate_interval_number("-13.0"), float)
        and Interval._validate_interval_number("-13.0") == -13.0
    )
    assert (
        isinstance(Interval._validate_interval_number("-13"), int) and Interval._validate_interval_number("-13") == -13
    )
    assert (
        isinstance(Interval._validate_interval_number(" -13 "), int)
        and Interval._validate_interval_number(" -13 ") == -13
    )
    # edge cases
    assert Interval._validate_interval_number(True) == 1
    assert Interval._validate_interval_number(False) == 0

    # error cases
    with pytest.raises(ValueError, match="\[1, 2\] is not a valid interval number"):
        Interval._validate_interval_number([1, 2])
    with pytest.raises(ValueError, match="13.0.1 is not a valid interval number"):
        Interval._validate_interval_number("13.0.1")
    with pytest.raises(ValueError, match="- 13 is not a valid interval number"):
        Interval._validate_interval_number("- 13")


def test_interval_array():
    interval_array = IntervalArray([Interval.from_string("(5,13]"), Interval.from_string("(-2,6]")])
    assert interval_array.contains(-1) == [False, True]
    assert interval_array.contains(6) == [True, True]
    assert interval_array.contains(13) == [True, False]
    assert interval_array.contains(10) == [True, False]
    assert interval_array.contains(0) == [False, True]
    assert interval_array.contains(-2) == [False, False]
    assert interval_array.contains(14) == [False, False]
    assert interval_array.contains(-5) == [False, False]
