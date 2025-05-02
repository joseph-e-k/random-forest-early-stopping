import os
import tempfile
from ste.optimization import get_optimal_stopping_strategy, show_stopping_strategy
from tests.utils import compare_images_with_cleanup


REFERENCES_DIR = os.path.join(os.path.dirname(__file__), "references")


def test_regression_sanity():
    reference_dir = os.path.join(REFERENCES_DIR, "ss_visualization_11_submodels_1e-2_adr")

    n = 11
    adr = 1e-2

    oss = get_optimal_stopping_strategy(n, adr)

    with tempfile.TemporaryDirectory() as output_dir:
        show_stopping_strategy(oss, output_dir)

        for file_name in os.listdir(reference_dir):
            diff = compare_images_with_cleanup(
                os.path.join(reference_dir, file_name),
                os.path.join(output_dir, file_name),
                tol=0
            )
            assert diff is None, f"Images differ: {diff}"
