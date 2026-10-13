# Calibrated detector access to diffusion geometry

Goal: determine what a physically specified detector preserves of the sharp
population entropy minimum, and separate identifiability from stable finite-data
inference. This is a new exploratory dependency, not a completed result.

The eighth-edition candidate in observed_geometry is frozen for clean
reproduction. No agent may modify it during this exploration.

Loop: specify the detector map and access; test exact counterexamples; inspect
primary prior art; prove surviving injectivity or impossibility statement;
identify resource and calibration conditions; independently review before any
promotion. A mathematical recovery statement is not a laboratory estimator.

Assignments:
- population_theory: population recovery/injectivity theorem, theory/.
- geometry_countermodels: challenge detector injectivity and stability, countermodels/.
- geometry_literature: closest primary prior art, literature/.
- root: integration, release completion and next experimental design.

Initial question: does known boxcar exposure plus known independent additive
noise preserve the continuous ideal law for a bounded mixing stationary diffusion
signal under arbitrarily dense temporal access? Fixed sampling, unknown detector
parameters and finite-error stability are separate questions. No answer is assumed.

## Development checkpoint

Population recovery proof is in theory/detector_recovery.tex; finite-access
proof is being consolidated in theory/finite_access.tex. Independent review
is in review/. The exact fixed-grid OU countermodel has five passing tests.
The literature audit identifies established noisy diffusion, exposure and
entropy-inference methods; historical priority is not established.

The strongest finite-access comparison now targets the sharp ideal-law
minimum, not arbitrary invisible hidden entropy. Fixed-precision stationary
trials and phase-prepared trials have different count exponents for the stated
simple hypotheses. Under a declared sensor energy/noise law, optimized detector
energy exponents can coincide. Full physical preparation/drive/reset costs
remain outside that comparison.

geometry_experiment owns design/ after two transport failures interrupted the
previous assignee before any files were written. It will implement only
reproducible analytical design bounds, not label them as simulated experiments.
All results remain separate from the eighth edition frozen for reproduction.

## Ninth edition completed
Recorded 2026-10-13T00:52:02.237036+00:00.

The cumulative reference contains 336 pages. Toolkit 0.9.0 passes
702 tests, 103 complete reproduction
stages and 120 clean-source/installed-wheel comparisons.
All required records, computations, figures and reference text reproduce.
All numerical subprocesses pin BLAS threads to one; strict comparisons remain.
The eight earlier public editions remain byte-identical.

The detector layer connects the eighth-edition sharp population minimum to
known exposure, calibrated noise and finite physical access. Full dense-law
recovery under the stated mixing assumptions coexists with exact fixed-grid
ambiguity and unstable finite inference. The noise obstruction targets the
minimum fixed by the ideal circle law, not arbitrary hidden excess entropy.

For the specified resonant binary comparison at fixed positive read precision,
passive stationary trials have matching quartic sample order while oriented
preparation has matching quadratic order. An exact-rational check certifies
10 million prepared trials sufficient where 17 million passive stationary
trials are insufficient. Counts are analytical bounds, not simulated outcomes.
Under a separately declared inverse-energy read-noise law, optimized detector
energies in the two classes both have quadratic order. Preparation, drive,
reset, calibration and total apparatus costs remain additional.

The exact Gaussian OU sampling countermodel uses a noncompact state space
and an unbounded detector. Its actual-entropy ambiguity is not promoted to a
claim about the compact-class minimum. Existing filter inversion, noisy
volatility methods and Gaussian testing identities are explicitly credited.

The new independent review has 11 passing tests and final source bindings.
Historical priority, external peer review and prospective detector validation
remain open. The next practical dependency is a detector-specific finite-sample
joint calibration guarantee for local volatility and the short-time entropy
statistics, with selection and dependence controlled, compared against
established motion-blur/noisy-diffusion inference at stated physical resources.

All release jobs are complete. This is a continuation record, not a scheduled
or background research service. No new universal physical law is claimed.
