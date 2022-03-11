# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/
import re
from typing import List, Optional

from smclarify.bias.metrics import constants


class Interval:
    """Represents a numerical interval with a left bound and right bound"""

    _supported_closed = ["left", "right", "both", "neither"]

    interval_regex_pattern = re.compile(r"^ *[\[\(] *-?[0-9]*\.?[0-9]* *, *-?[0-9]*\.?[0-9]* *[\]\)] *$")

    def __init__(self, left: Optional[float], right: Optional[float], closed: str = "both"):
        if closed not in self._supported_closed:
            raise ValueError(f"closed={closed} is not valid.  Must be in {self._supported_closed}")
        if left and right and left > right:
            raise ValueError("left bound cannot be greater than right bound")
        if closed == self._supported_closed[-1] and left and right and left == right:
            raise ValueError("Interval has exclusive bounds, but left==right")
        if left is None:
            self.left = float("-inf")
        else:
            self.left = left
        if right is None:
            self.right = float("inf")
        else:
            self.right = right
        self.closed = closed
        self.left_inclusive = True
        self.right_inclusive = True
        if closed == "left" or closed == "neither":
            self.right_inclusive = False
        if closed == "right" or closed == "neither":
            self.left_inclusive = False

    def __repr__(self):
        if self.left_inclusive:
            left_bound = "["
        else:
            left_bound = "("
        if self.right_inclusive:
            right_bound = "]"
        else:
            right_bound = ")"
        return left_bound + str(self.left) + ", " + str(self.right) + right_bound

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.__dict__ == other.__dict__
        return False

    def __contains__(self, item):
        return self.contains(item)

    def contains(self, item: float) -> bool:
        if item == self.left and item == self.right:
            return self.left_inclusive or self.right_inclusive
        if item == self.left:
            return self.left_inclusive
        elif item == self.right:
            return self.right_inclusive
        else:
            return self.left < item < self.right

    @classmethod
    def from_string(
        cls, interval: str, min_value: Optional[float] = None, max_value: Optional[float] = None
    ) -> "Interval":
        """
        Generates an Interval from a string representation.
        The string must fit the regex pattern -> r'^ *[\[\(] *-?[0-9]*\.?[0-9]* *, *-?[0-9]*\.?[0-9]* *[\]\)] *$'
        Valid examples: "(1, 5]", "[1, 5]", "[1, 5)", "(1, 5)", "(, 5]", "(5, ]", "(-inf, inf]", "[-inf,)"

        :param interval: string representation of an interval
        :param min_value: Minimum bound for this interval.  Useful for when the minimum bound is not specified in the
            interval string, but minimum should default to this min_value instead of float(-inf)
        :param max_value: Maximum bound for this interval.  Similar use to min_value
        :return: Interval object decoded from the interval string
        :raise ValueError: If the input is malformed
        """

        interval = interval.strip()
        if interval.startswith("["):
            left_inclusive = True
        elif interval.startswith("("):
            left_inclusive = False
        else:
            raise ValueError(
                f"interval={interval} is not valid.  " f'Must start with an interval left bound "[" or "("'
            )
        if interval.endswith("]"):
            right_inclusive = True
        elif interval.endswith(")"):
            right_inclusive = False
        else:
            raise ValueError(f"interval={interval} is not valid.  " f'Must end with an interval right bound "]" or ")"')

        interval_bounds = [x.strip() for x in interval[1:-1].split(",")]
        if len(interval_bounds) != 2:
            raise ValueError(f'interval={interval} is not valid.  Cannot contain more than one comma ","')

        try:
            left_bounds = interval_bounds[0]
            if left_bounds == "" or left_bounds == constants.NEGATIVE_INFINITY_STR:
                left = min_value if min_value else None
            else:
                left = cls._validate_interval_number(left_bounds)
            right_bounds = interval_bounds[1]
            if right_bounds == "" or right_bounds == constants.INFINITY_STR:
                right = max_value if max_value else None
            else:
                right = cls._validate_interval_number(right_bounds)
            if left_inclusive and right_inclusive:
                closed = "both"
            elif left_inclusive:
                closed = "left"
            elif right_inclusive:
                closed = "right"
            else:
                closed = "neither"
            return cls(left, right, closed)
        except ValueError as e:
            raise ValueError(f"interval={interval} is not valid") from e

    @staticmethod
    def _validate_interval_number(val) -> float:
        if isinstance(val, float) or isinstance(val, int):
            return val
        elif isinstance(val, str):
            stripped_val = val.strip()
            if stripped_val.startswith("-"):
                stripped_val = stripped_val[1:]
            if stripped_val.isdecimal():
                return int(val)
            elif stripped_val.replace(".", "", 1).isdecimal():
                return float(val)
        raise ValueError(f"{val} is not a valid interval number")


class IntervalArray:
    """Wraps around a list of intervals"""

    def __init__(self, intervals: List[Interval]):
        self.intervals = intervals

    def __iter__(self):
        return iter(self.intervals)

    def contains(self, item: float) -> List[bool]:
        """
        Element-wise contains check for each interval

        :param item: item to check if this IntervalArray contains
        :return: bool list where True is set at the indices where intervals in the list contain the item
        """

        res = []
        for interval in self.intervals:
            res.append(interval.contains(item))
        return res
