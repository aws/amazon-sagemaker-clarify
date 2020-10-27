"""Convenience facility to access datasets"""
import os
from abc import abstractmethod
from typing import Tuple, List
import logging

import pandas as pd
import pandas.io.common
import urllib.parse
import boto3
import botocore
import textwrap

log = logging.getLogger(__name__)


def cache_dir(*paths) -> str:
    """
    :return: cache directory
    """
    j = os.path.join
    path = j(
        os.environ.get("XDG_CACHE_HOME", j(os.environ.get("HOME", ""), ".cache")),  # type: ignore
        "famly",
        "datasets",
    )
    dir = j(path, *paths)
    os.makedirs(dir, exist_ok=True)
    return dir


def url_is_remote(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme and parsed.scheme != "file":
        return True
    return False


class Dataset:
    def __init__(self, id, source, description):
        self.id = id
        self.source = source
        self.description = description

    @abstractmethod
    def ensure_local(self) -> None:
        pass

    @abstractmethod
    def train(self) -> pd.DataFrame:
        """ "
        :returns: the dataframe for the train dataset
        """

    def test(self) -> pd.DataFrame:
        """ "
        :returns: the dataframe for the test dataset
        """
        return pd.DataFrame()

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Custom dataset preprocessing"""
        return df


class S3CachedDatasetMixin:
    def __init__(self, id, source):
        self.id = id
        self.source = source
        self.s3handler = S3Handler()

    @staticmethod
    def relative_file(url: str) -> str:
        """
        :return: the relative file from the url path  s3://aaa/bbb ->  bbb  s3://cc/d/ -> ''
        """
        return os.path.split(urllib.parse.urlparse(url).path)[1]

    def cached_filepath(self, file) -> str:
        assert not os.path.isabs(file)
        dir = os.path.join(cache_dir(self.id))
        path = os.path.join(dir, file)
        return path

    def local_path(self):
        rel_file = self.relative_file(self.source)
        res = self.cached_filepath(rel_file)
        return res

    def ensure_local(self) -> None:
        local_path = self.local_path()
        if os.path.exists(local_path) and not self.s3handler.changed(self.source, local_path):
            log.info("Local file '%s' up to date", local_path)
        else:
            log.info("Download '%s' -> '%s'", self.source, local_path)
            self.s3handler.download(self.source, local_path)


class S3Handler:
    def __init__(self):
        self.s3r = boto3.resource("s3")
        self.s3: botocore.client.S3 = boto3.client("s3")

    @staticmethod
    def url_bucket_key(url) -> Tuple[str, str]:
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme.startswith("s3")
        bucket, path = parsed.netloc, parsed.path[1:]
        return (bucket, path)

    def download(self, s3_url, local) -> None:
        (bucket, path) = self.url_bucket_key(s3_url)
        self.s3.download_file(bucket, path, local)
        o = self.s3r.Object(bucket, path)
        mtime = o.last_modified.timestamp()
        os.utime(local, (mtime, mtime))

    def changed(self, s3_url, local) -> bool:
        """
        :param local: local file
        :param s3_url: s3://bucket/key of remote file
        :return: False if the size and mtime matches from remote, otherwise True
        """
        (bucket, path) = self.url_bucket_key(s3_url)
        assert os.path.isfile(local)
        o = self.s3r.Object(bucket, path)
        if o.last_modified.timestamp() == os.path.getmtime(local) and o.content_length == os.path.getsize(local):
            return False
        return True


class S3Dataset(S3CachedDatasetMixin, Dataset):
    def __init__(self, id, source, description):
        Dataset.__init__(self, id, source, description)
        S3CachedDatasetMixin.__init__(self, id, source)

    def train(self) -> pd.DataFrame:
        self.ensure_local()
        # FIXME use data loader and check CSV, PARQUET etc
        return pd.read_parquet(self.local_path())

    def read_csv_data(self, index_col=False) -> pd.DataFrame:
        self.ensure_local()
        return pd.read_csv(self.local_path(), index_col=index_col)


class Datasets:
    datasets = {
        "lending_club": S3Dataset(
            "lending_club",
            "s3://famly-datasets/lending-club/loan.parquet.gz",
            textwrap.dedent("""Lending club dataset"""),
        ),
        "lending_club_small": S3Dataset(
            "lending_club_small",
            "s3://famly-datasets/lending-club/loan_small.parquet.gz",
            textwrap.dedent("""Lending club dataset. 10K rows."""),
        ),
        "german_lending": S3Dataset(
            "german_lending",
            "s3://famly-datasets/statlog/german.parquet",
            textwrap.dedent("""German Lending dataset"""),
        ),
        "german_csv": S3Dataset(
            "german_csv",
            "s3://famly-datasets/statlog/german_data.csv",
            textwrap.dedent("""German Lending dataset"""),
        ),
        "german_predicted_labels": S3Dataset(
            "german_predicted_labels",
            "s3://famly-datasets/statlog/predicted_labels.csv",
            textwrap.dedent("""German Lending dataset"""),
        ),
    }

    @staticmethod
    def list() -> List[str]:
        return list(Datasets.datasets.keys())

    def __call__(self, dataset_name: str) -> Dataset:
        assert (
            dataset_name in self.list()
        ), f"Dataset {dataset_name} is not a known dataset. Use DataSetInventory.list() to get a list of datasets"
        return self.datasets[dataset_name]


def german_lending_readable_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Statlog German lending dataset to have human readable values
    https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    """
    # df["target"] = df["Class1Good2Bad"].replace([1, 2], ["good", "bad"]).astype('category')
    df["target"] = df["Class1Good2Bad"].replace([1, 2], [1, 0]).astype("category")
    df = df.drop(columns=["Class1Good2Bad"])
    df["CheckingAC_Status"] = (
        df["CheckingAC_Status"]
        .replace(["A11", "A12", "A13", "A14"], ["x < 0 DM", "0 <= x < 200 DM", "x >= 200DM", "no checking account"])
        .astype("category")
    )
    df["CreditHistory"] = (
        df["CreditHistory"]
        .replace(
            ["A30", "A31", "A32", "A33", "A34"],
            ["no credits", "all credits paid", "existing credits paid", "delay", "critical accnt. / other credits"],
        )
        .astype("category")
    )
    df["Purpose"] = (
        df["Purpose"]
        .replace(
            ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"],
            [
                "new car",
                "used car",
                "forniture",
                "radio/tv",
                "appliances",
                "repairs",
                "education",
                "vacation",
                "retraining",
                "business",
                "others",
            ],
        )
        .astype("category")
    )
    df["SavingsAC"] = (
        df["SavingsAC"]
        .replace(
            ["A61", "A62", "A63", "A64", "A65"],
            ["x < 100 DM", "100 <= x < 500 DM", "500 <= x < 1000 DM", "x >= 1000 DM", "unknown"],
        )
        .astype("category")
    )
    df["Employment"] = (
        df["Employment"]
        .replace(
            ["A71", "A72", "A73", "A74", "A75"],
            ["unemployed", "x < 1 year", "1 <= x < 4 years", "4 <= x < 7 years", "x >= 7 years"],
        )
        .astype("category")
    )
    df["SexAndStatus"] = (
        df["SexAndStatus"]
        .replace(
            ["A91", "A92", "A93", "A94", "A95"],
            [
                "male divorced/separated",
                "female divorced/separated/married",
                "male single",
                "male married/widowed",
                "female single",
            ],
        )
        .astype("category")
    )
    df["OtherDebts"] = (
        df["OtherDebts"].replace(["A101", "A102", "A103"], ["none", "co-applicant", "guarantor"]).astype("category")
    )
    df["Property"] = (
        df["Property"]
        .replace(
            ["A121", "A122", "A123", "A124"],
            ["real estate", "soc. savings / life insurance", "car or other", "unknown"],
        )
        .astype("category")
    )
    df["OtherInstalmentPlans"] = (
        df["OtherInstalmentPlans"].replace(["A141", "A142", "A143"], ["bank", "stores", "none"]).astype("category")
    )
    df["Housing"] = df["Housing"].replace(["A151", "A152", "A153"], ["rent", "own", "for free"]).astype("category")
    df["Job"] = (
        df["Job"]
        .replace(
            ["A171", "A172", "A173", "A174"],
            [
                "unemployed / unskilled-non-resident",
                "unskilled-resident",
                "skilled employee / official",
                "management / self-employed / highly qualified employee / officer",
            ],
        )
        .astype("category")
    )
    df["Telephone"] = df["Telephone"].replace(["A191", "A192"], ["none", "yes"]).astype("category")
    df["ForeignWorker"] = df["ForeignWorker"].replace(["A201", "A202"], ["yes", "no"]).astype("category")
    return df
