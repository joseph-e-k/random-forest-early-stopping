import gc

from predestined_k_approach.Forest import Forest
from predestined_k_approach.optimization import *
from predestined_k_approach.ForestWithEnvelope import ForestWithEnvelope
from predestined_k_approach.ForestWithStoppingStrategy import ForestWithGivenStoppingStrategy
from predestined_k_approach.utils import TimerContext


def make_sky_from_truncated_theta(truncated_theta):
    n = truncated_theta.shape[0] - 1
    theta = np.ones((n + 1, n + 1))
    theta[:, :truncated_theta.shape[1]] = truncated_theta

    p = np.zeros_like(theta)

    theta_bar = 1 - theta

    p[0, 0] = 1
    for i in range(n):
        p[i + 1, 0] = p[i, 0] * theta_bar[i, 0]

        for j in range(i + 1):
            p[i + 1, j + 1] = (
                ((i - j) / (i + 1)) * p[i, j + 1] * theta_bar[i, j + 1]
                + ((j + 1) / (i + 1)) * p[i, j] * theta_bar[i, j]
            )

    return Sky(p, p * theta, p * theta_bar)


def main():
    aer = 10**-6

    for n_total in [145]:
        n_positive_low = n_total // 2
        n_positive_high = n_positive_low + 1

        low_forest = Forest(n_total, n_positive_low)
        high_forest = Forest(n_total, n_positive_high)

        low_fwe = ForestWithEnvelope.create_greedy(n_total, n_positive_low, aer)
        high_fwe = ForestWithEnvelope(forest=high_forest, envelope=low_fwe.envelope)
        fwe_sky = make_sky_from_truncated_theta(low_fwe.get_prob_stop())
        fwe_theta_reconstructed = make_theta_from_sky(fwe_sky)
        fwe_theta_reconstructed_truncated = fwe_theta_reconstructed[:, :(n_positive_low + 1)]
        assert np.all(np.abs(fwe_theta_reconstructed_truncated - low_fwe.get_prob_stop()) < 0.0001)

        with TimerContext(f"find optimal stopping strategy ({n_total=}, {aer=})"):
            fwss_sky = make_and_solve_optimal_stopping_problem(n_total, aer)
            optimal_stopping_strategy = make_theta_from_sky(fwss_sky)

        low_fwss = ForestWithGivenStoppingStrategy(low_forest, optimal_stopping_strategy)
        high_fwss = ForestWithGivenStoppingStrategy(high_forest, optimal_stopping_strategy)

        low_fwss_time = low_fwss.analyse().expected_runtime
        high_fwss_time = high_fwss.analyse().expected_runtime
        low_fwe_time = low_fwe.analyse().expected_runtime
        high_fwe_time = high_fwe.analyse().expected_runtime

        if low_fwss_time > low_fwe_time and high_fwss_time > high_fwe_time:
            pass
        else:
            continue

        print("***")
        print(f"{low_fwss.analyse().prob_error=}")
        print(f"{high_fwss.analyse().prob_error=}")
        print(f"{low_fwe.analyse().prob_error=}")
        print(f"{high_fwe.analyse().prob_error=}")

        a = make_abstract_probability_matrix(n_total, np.array([n_positive_low, n_positive_high]))

        for label, sky in [("fwss", fwss_sky), ("fwe", fwe_sky)]:
            beta = a * sky.pi
            prob_B_equals = np.sum(beta, axis=2)
            expected_B = np.sum(prob_B_equals * np.arange(n_total + 1), axis=1)
            print(f"theoretical expected runtimes for {label}: {expected_B}")

            print(f"{n_total=}, {low_fwss_time=}, {low_fwe_time=}, {high_fwss_time=}, {high_fwe_time=}")

        with TimerContext(f"find optimal stopping strategy with known solution ({n_total=}, {aer=})"):
            fwss_sky_recomputed = make_and_solve_optimal_stopping_problem(n_total, aer, known_solution=fwe_sky)
            optimal_stopping_strategy_recomputed = make_theta_from_sky(fwss_sky_recomputed)

        low_fwss_recomputed = ForestWithGivenStoppingStrategy(low_forest, optimal_stopping_strategy_recomputed)
        high_fwss_recomputed = ForestWithGivenStoppingStrategy(high_forest, optimal_stopping_strategy_recomputed)

        print(f"{low_fwss_recomputed.analyse().expected_runtime=}")
        print(f"{high_fwss_recomputed.analyse().expected_runtime=}")
        print(f"{low_fwss_recomputed.analyse().prob_error=}")
        print(f"{high_fwss_recomputed.analyse().prob_error=}")

        print("***")
        print()


if __name__ == "__main__":
    main()
