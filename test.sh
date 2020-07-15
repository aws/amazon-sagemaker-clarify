#!/usr/bin/env bash
coverage run -m pytest --pspec
coverage report -m --fail-under=72
