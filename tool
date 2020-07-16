#!/usr/bin/env bash
set -euo pipefail

install_dependencies_dev() {
    python -m pip install --upgrade pip
    pip install -e .[test]
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
}

unit_tests() {
    pytest --pspec
}

lint() {
    pre-commit run -v -a
    flake8 . --count --show-source --statistics
}

test_with_coverage() {
    coverage run -m pytest --pspec
    coverage report -m --fail-under=72
}

docs() {
    cd docs
    make html
}

all() {
    lint
    test_with_coverage
    docs
}


##############################################################
# MAIN
#
# Run function passed as argument
set +x
if [ $# -gt 0 ]
then
    $@
else
    cat<<EOF

**Developer tool**
==================

$0: Execute a function by passing it as an argument to the script:

Possible commands:
==================

EOF
    declare -F | cut -d' ' -f3
    echo
fi
