import pytest

from ste.utils import logging

@pytest.fixture(autouse=True, scope="function")
def shared_setup():
    logging._TLS.breadcrumbs = ()
