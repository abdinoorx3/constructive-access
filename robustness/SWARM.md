# Goal and research loop

Develop an experimentally simpler, calibration-aware test of a protected
classical dynamical description, with explicit alternatives explaining failure.
The concrete candidate compares cumulative occupation after a trusted reset
with occupation from another preparation. An independent randomized clock
turns the integrated inequality into a single binary readout per trial.

Assignments:
- robust_physics: imperfect preparation, selectivity, delay and current inference.
- prepare_integrate: reset/random-clock protocol, quantum model, established comparator.
- detector_boundary: protected detector interfaces and approximate renewal reset.
- robust_statistics: calibrated sequential testing and finite-record guarantees.
- response_literature: prior art and realistic implementation requirements.
- response_redteam: independently derived counterexamples and theorem/code review.
- root: clock design, integration, goals and evidence ledger, reproduction/release.

Scientific constraints:
- Integrated occupation positivity is an elementary consequence of established
  first-passage theory; no new general Markov mathematics is asserted for it.
- Existing temporal population inequalities already exclude arbitrary-size
  classical Markov networks under protected-state assumptions. Compare them.
- A classical hidden-state or reset error can explain a violated inequality;
  a violation alone is not device-independent evidence of quantum physics.
- Timing and preparation choices must obey the precise null used by the test.

Completion gate: complete proofs, controls, calibrated executable decision
rules, matched comparisons and independent internal review. Physical validation
and external priority review remain separate requirements.

## Integrated results and completion gate
The principal scope extension is |C0-Cw| <= 1 for every homogeneous classical
CTMC with at most three states, arbitrary preparations and a fixed stochastic
binary detector. A tilted qubit preparation gives an exact witness above 1.08;
a calibrated four-arm test uses 6000 records. This complements the sharp reset
and crossing-intensity explanation boundary, matched preparation/current
bounds, clock design and exact statistical inversion. Existing nonlinear
survival tests outperform the simpler orthogonal assay in the matched example.

All six agent assignments are complete. Independent review contains 26 tests,
including exhaustive three-state rate-grid checks and a separate GKSL check.
Root is integrating the cumulative third edition and verifying clean source
and installed-wheel reproduction. There are no laboratory measurements or
external peer-review decisions. Exact historical priority remains unresolved.

## Completed research release
Recorded 2026-10-12T19:30:49.534231+00:00.

The cumulative third edition is complete: 141 pages, toolkit0.3.0,223 passing
tests,38 reproduction stages and31 clean-source/installed-wheel comparisons.
All earlier public releases remain byte-identical. All agent tasks are finished;
no background work or scheduled automation is running.

Next dependency: apparatus-specific validation of stable detector/preparation,
common homogeneous evolution and independent timing; then a preregistered
four-arm realization-size experiment. Comparative empirical power/energy versus
established dynamical dimension witnesses and exact historical priority remain
open. The next mathematical extension should target a robust family of resource
requirements justified by data, not a universal law inferred from a toy model.
