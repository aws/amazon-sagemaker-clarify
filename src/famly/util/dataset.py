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
        """"
        :returns: the dataframe for the train dataset
        """

    def test(self) -> pd.DataFrame:
        """"
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
        o: boto3.resources.factory.s3.Object = self.s3r.Object(bucket, path)
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
        o: boto3.resources.factory.s3.Object = self.s3r.Object(bucket, path)
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
    }

    @staticmethod
    def list() -> List[str]:
        return Datasets.datasets.keys()

    def __call__(self, dataset_name: str) -> Dataset:
        assert (
            dataset_name in self.list()
        ), f"Dataset {dataset_name} is not a known dataset. Use DataSetInventory.list() to get a list of datasets"
        return self.datasets[dataset_name]
