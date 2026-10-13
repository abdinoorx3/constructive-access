# Joint detector calibration and diffusion geometry

Goal: derive and validate a finite-data entropy certificate that accounts for
uncertain detector parameters and local diffusion geometry, with all additional
structural assumptions and physical acquisition costs stated. Preserve all nine
released editions. A negative identifiability result is a legitimate outcome;
never replace an unproved pointwise envelope with an average estimate.

Research loop: specify access and target; derive exact cases; search for
countermodels and prior art; prove surviving claims; implement and freeze a
matched virtual laboratory; independently review; integrate verified results
into the cumulative reference. No result is assumed in advance.

Assignments:
- population_theory: sufficient calibration structure and joint confidence, theory/.
- geometry_countermodels: impossibility of unsupported calibration, countermodels/.
- geometry_literature: primary methods and appropriate baselines, literature/.
- geometry_experiment: candidate detector experiment and exact moment solver, experiment/.
- geometry_independent_review: adversarial assessment and independent validation, review/.
- root: detector score algebra, integration, code, evidence ledger and release.

Initial candidate: replace a falsely exact local-volatility input by a jointly
covered nuisance region, carrying noise/exposure uncertainty through a finite
entropy score. Distinguish unrestricted hidden diffusions from a calibrated
cosine-circle model. A pointwise envelope cannot be inferred from a sample
average without additional structure. No physical experiment is implied.

## First derivation checkpoint
The unrestricted calibration problem has a compact rare-region obstruction.
The positive route now assumes the explicit local-geometry identity
Gamma_f=gamma*(1-f^2), with unknown scale gamma and known generator caps.
Independent bounded detector replicas estimate polynomial moments without a
noise-variance input. A stationary L1 semigroup lemma supplies finite-exposure
bias. Constant and timed-step reference records cover common detector gain and
uniform shutter width. All nuisance and target statements share simultaneous
coverage; no same-data fitted score is treated as predeclared.

calibration_api owns code/ and tests/. Theory, experiment and independent
review are coordinating formulas and the exact finite-law simulator. The
simulation contract is not yet frozen. Root owns integration and packaging.

## Frozen evidence checkpoint
The theoretical and simulator reviews pass; the frozen benchmark has completed
1,600 datasets. Exact sufficient counts and per-dataset certificates are retained.
Geometry detection is 400/0/0/395 versus 400/0/27/400 for the broader-class
endpoint comparator. All selected geometry prefixes are order zero. Five
simultaneous target confidence events miss their true means, while all
calibration regions and finite gamma bounds cover truth. No entropy overshoot
is observed. These results do not establish uniform method dominance.

A sharp extension to reversible hidden stochastic modulation is proved:
minimum E[c(Z)^2]/gamma can remain positive at zero mean drive despite a fully
reversible scalar path law. Ordinary reversal is essential. This extension is
analytic only. Final presentation, source binding and complete clean cumulative
reproduction are the remaining release tasks.

## Scientific milestone completed; first public launch requested
Recorded 2026-10-13T02:23:49.455932+00:00. All 111 full reproduction stages,
137 clean-source and installed-wheel comparisons, and 768 tests pass. The
internal reference has 365 pages and 548 scientific source bindings.

The user directed research closure at this milestone and preparation of a
coherent first public release. Earlier numbered editions were internal/local
iterations, not prior public publications. No separate tenth public archive is
being issued. Continuation now lives in ../public_launch/ and covers the public
entry point, citation, rights and packaging. Scientific files are frozen;
external posting has not occurred.
