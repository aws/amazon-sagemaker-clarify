from setuptools import find_packages, setup

INSTALL_REQUIRES = ["boto3", "pyarrow", "pandas", "s3fs", "numpy", "sklearn", "pyfunctional"]

EXTRAS_REQUIRE = {
    "test": [
        "tox",
        "flake8",
        "black",
        "mock",
        "pre-commit",
        "pytest",
        "pytest-pspec",
        "sphinx",
        "coverage",
        "nbconvert",
        "jupyter",
        "seaborn",
    ]
}

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()


setup(
    name="famly",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ],
)
