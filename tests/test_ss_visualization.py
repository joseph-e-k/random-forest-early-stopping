import os
import tempfile
from ste.optimization import get_optimal_stopping_strategy, show_stopping_strategy
from tests.utils import assert_directory_of_images_matches_reference


REFERENCES_DIR = os.path.join(os.path.dirname(__file__), "references")


def test_regression_sanity():
    reference_dir = os.path.join(REFERENCES_DIR, "ss_visualization_11_submodels_1e-2_adr")

    n = 11
    adr = 1e-2

    oss = get_optimal_stopping_strategy(n, adr)

    with tempfile.TemporaryDirectory() as output_dir:
        show_stopping_strategy(oss, output_dir)
        assert_directory_of_images_matches_reference(reference_dir, output_dir)
