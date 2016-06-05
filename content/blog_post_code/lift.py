import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import json

def lift_simulations(Na, Pa, Nb, Pb, Nsim=10**4):
    """
    Na is the total events for strategy A,
    Pa is positive events (conversions) for A, 
    etc.
    """
    # add 1 to both alpha and beta for a uniform prior 
    cvr_b = np.random.beta(1 + Pb, 1 + Nb - Pb, size=Nsim)
    cvr_a = np.random.beta(1 + Pa, 1 + Na - Pa, size=Nsim)
    return (cvr_b / cvr_a) - 1.0


def sim_conf_int(Na, Pa, Nb, Pb, interval="two-sided", Nsim=10**4, CI=0.95):
    simulations = lift_simulations(Na, Pa, Nb, Pb, Nsim=Nsim)
    if interval == "upper":
        return (np.percentile(simulations, 100 * (1 - CI)), float("inf"))
    if interval == "lower":
        return (-1.0, np.percentile(simulations, 100 * CI))
    if interval == "two-sided":
        return np.percentile(simulations, (100 * (1 - CI)/2, 100*(1 - (1 - CI)/2)))

    raise ValueError("interval must be either 'upper', 'lower', or 'two-sided'")


def altman_interval(Na, Pa, Nb, Pb, CI=0.95, interval="two-sided", e=0.5):
    #lift of B over A
    if interval not in ("two-sided", "upper", "lower"):
        raise ValueError("Interval must be either 'two-sided', 'upper', or 'lower'.")
    #add e to each number to keep weird stuff from happening when Pa or Pb == 0
    Na += e
    Pa += e
    Nb += e
    Pb += e
    log_lift_estimate = math.log((float(Pb) / Nb) / (float(Pa) / Na))
    pval = (1.0 - CI) / 2 if interval == "two-sided" else (1.0 - CI)
    zval = norm.ppf(1.0 - pval)
    se = math.sqrt((1.0 / Pb) - (1.0 / Nb) + (1.0 / Pa) - (1.0 / Na))
    return (
        -1.0 if interval == "lower" else math.exp(
            log_lift_estimate - zval * se) - 1,
        float("inf") if interval == "upper" else math.exp(
            log_lift_estimate + zval * se) - 1
    )


if __name__ == "__main__":
    Na = 20000
    Pa = 400
    Nb = 1000
    Pb = 30
    
    CI_vals = list(np.arange(0.5, 1, 0.001)[::-1])
    altman_intervals = [altman_interval(Na, Pa, Nb, Pb, CI=x) for x in CI_vals]
    altman_low = [i[0] for i in altman_intervals]
    altman_high = [i[1] for i in altman_intervals]
    simulation_intervals = [sim_conf_int(Na, Pa, Nb, Pb, CI=x) for x in CI_vals]
    simulation_low = [i[0] for i in simulation_intervals]
    simulation_high = [i[1] for i in simulation_intervals]
    df = pd.DataFrame({
        "CI": 2 * CI_vals,
        "method": len(CI_vals)*["Altman"] + len(CI_vals)*["Simulation"],
        "low": altman_low + simulation_low,
        "high": altman_high + simulation_high,
        }
    )
    csv_file = "/Users/jarfa/Dropbox/jarfa.github.io/content/blog_post_code/lift.csv"
    with open(csv_file, "w") as f:
        df.to_csv(f, index=False)
        print "Wrote to %s" % csv_file
