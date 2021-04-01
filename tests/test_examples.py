# File: test_examples.py
# File Created: Wednesday, 31st March 2021 9:01:36 pm
# Author: Steven Atkinson (steven@atkinson.mn)

import logging
import os
import sys
from subprocess import check_call

import pytest

logger = logging.getLogger(__name__)

_examples_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples")
)
_examples = ["basic.py --no-show"]


@pytest.mark.parametrize("example", _examples)
def test(example):
    logger.info("Running:\npython examples/{}".format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(_examples_dir, filename)
    check_call([sys.executable, filename] + args)
