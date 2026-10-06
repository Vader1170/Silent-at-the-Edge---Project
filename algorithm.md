# Algorithm 1: ReliabilityChecker

A zero-added-cost second diagnostic for any real-time polynomial least-squares monitoring pipeline. Now available as the open-source Python package `reliabilitychecker` (MIT license, see the `reliabilitychecker/` directory).

## The problem it solves

A polynomial fit to a sliding window of sensor readings reports two things: an estimate, and a confidence interval. Every engineer downstream trusts the second number to say whether the first is safe to act on. This project shows that trust can be misplaced: a design's numerical conditioning and its reported variance can fail independently. On real river-gauge data, a sensor dropout inflated the fitting matrix's condition number by up to nine orders of magnitude while the standard variance diagnostic stayed nearly flat, saturating below 24%. A monitoring pipeline watching only the variance sees no warning.

## The fix

The condition number, kappa(**G**), is a byproduct of the same matrix inversion the fit already performs to compute its estimate and its variance. Tracking it costs nothing extra.

```
ALGORITHM 1: ReliabilityChecker(timestamps, readings, degree m, kappa_warn, kappa_crit)

1. Build the Gram matrix G and its data-weighted analogue G_d from the
   currently available timestamps, exactly as the fit already does.
2. Invert once: G^-1. Report the fit and its variance as usual.
3. At no extra cost, also read off kappa(G), the condition number of the
   matrix already factored in Step 2.
4. Return:
     CRITICAL   if kappa(G) >= kappa_crit
                (redesign sample placement before trusting this fit)
     WARNING    if kappa_warn <= kappa(G) < kappa_crit
     OK         if kappa(G) < kappa_warn
```

**Complexity.** O(m^3) for the Gram inversion, already paid by the fit. The condition number is a byproduct of the same factorization, or can be cheaply estimated from the extreme singular values already available in an SVD- or Cholesky-based solve. Step 3 adds no asymptotic cost over Step 2.

**Threshold calibration.** Set kappa_warn and kappa_crit relative to a station's own healthy baseline kappa0, not one constant across a whole network. Across the seven real stations tested (Table 1 and Appendix C of the paper), kappa(**G**) travels between under two and just over nine orders of magnitude before a design becomes unusable, a roughly seven-order-of-magnitude spread in that ceiling from station to station (kappa0 around 1.1 to 1.4 at health, up to about 10^9 under dropout at the worst station tested). A reference calibration, `kappa_warn = 10 * kappa0`, `kappa_crit = 1e4 * kappa0`, ships with the package and correctly returns `OK` for all seven real stations tested at their current, undropped cadence.

## Reference implementation

`reliability_check()` in the `reliabilitychecker` package. Signature:

```python
reliability_check(t, t_ref=None, m=10, kappa_warn=1e3, kappa_critical=1e6, x0=1.0)
# -> {"kappa": ..., "variance": ..., "leverage": ..., "status": "OK"|"WARNING"|"CRITICAL"}
```

`calibrated_thresholds(t_healthy, m=10)` computes `kappa_warn` and `kappa_critical` from a station's own baseline automatically. `code.py`'s `reliability_check()` and `register_station()` reproduce this for all seven paper stations (three in the main text, four more in Appendix C) directly; the package generalizes it for any new dataset.

## Extending it

`register_station()` in `code.py` adds any new real timestamp record, another CWC/CPCB gauge, or an unrelated sliding-window sensor entirely, to every figure and check in the pipeline with one function call. The `reliabilitychecker` package works the same way outside the paper's own dataset: point `reliability_check()` at any real timestamp array.

## Why this is the engineering contribution

The mathematics (Position Penalty Index, the Classification Principle, Theorem 8) explains why this failure mode exists and when it is severe. ReliabilityChecker is what an engineer actually deploys. It is packaged, tested, and installable, not a script tied to one dataset.
