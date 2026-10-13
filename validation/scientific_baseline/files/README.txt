TENTH EDITION — JOINT CALIBRATION AND HIDDEN STOCHASTIC DRIVE
Abdinoor / 2D Labs / 13 October 2026

This edition makes uncertain detector calibration part of a finite-data entropy
certificate. It also proves an exact extension of the hidden-dynamics minimum:
a completely time-reversible scalar signal can require positive ordinary-reversal
entropy even when the hidden angular drive has zero mean.

Begin the new layer with "Learning the calibration that an entropy certificate
needs" in manuscript/reference.pdf. The cumulative reference still begins with
basic access and develops the classical, quantum, statistical and physical
framework before these extensions.

The finite certificate assumes a bounded observable with the exact pointwise
relation Gamma_f=gamma*(1-f^2). The positive constant gamma is unknown and is
covered from temporal moments. The functional relation itself, and numerical
supremum bounds on first and second forward/adjoint generator derivatives, are
supplied model assumptions. They are not learned universally from the records.
Distinct conditionally independent readouts of the same exposure estimate
polynomial moments without inserting a known noise variance. Constant and timed
step references cover a common gain and uniform shutter width. One simultaneous
confidence event carries these uncertainties through the entropy calculation.

The exact hidden-drive family has generator

    L=gamma*partial_phi^2+c(z)*partial_phi+L_Z,

where Z is a reversible smooth elliptic diffusion on a compact manifold and the
detector is cos(phi). Its complete stationary scalar law is reversible, yet its
minimum compatible ordinary-reversal entropy is exactly E[c(Z)^2]/gamma over the
declared compact hidden-diffusion class, allowing arbitrary finite hidden
dimension. The joint model attains this minimum. A second diffusing angle with
c(z)=epsilon*cos(z) gives zero mean angular current and minimum
epsilon^2/(2*gamma). This is a proved model-class result. Odd-variable reversal,
jump models and arbitrary physical realizations are outside that minimum.

A finite polynomial hierarchy approaches the minimum with increasing access.
Its sufficient consistency schedule increases replicas, samples, reference
measurements and arithmetic precision while shrinking lag and exposure. It is
not a fixed-exposure or practical sample-efficiency theorem. The installed
fixed-precision API certifies finite designs; the asymptotic theorem requires
the separately specified increasing precision.

The frozen virtual experiment retains substantial acquisition costs and failures.
There are 400 datasets in each of four cases. Every dataset represents
13,333,333 independent stationary target trials, four center replicas plus two
endpoint readings per trial, and 100,000 readings for each of two reference
experiments: 80,199,998 detector readings in total. Preparation, reset, clock,
reference generation, replica construction and apparatus energy remain
additional unmeasured costs. These counts are not a calibrated joule estimate.

Positive geometry certificates versus the broader-class endpoint comparator:

    strong drive          400/400 versus 400/400
    equilibrium             0/400 versus   0/400
    weak drive              0/400 versus  27/400
    changed calibration   395/400 versus 400/400

The mean geometry bounds are approximately 0.577 and 0.599 in the two stronger
cases, whose simulator entropy rates are 9 and 18. Every selected polynomial
prefix is order zero; the higher prefix does not win at this budget. The
endpoint comparator requires less geometric structure, so larger geometry
bounds do not demonstrate dominance under identical assumptions.

The tutorial also exposes the maximum of the geometry and endpoint bounds.
Both hold on the same event, so this combined output uses no extra failure
allocation under the geometry model contract. It retains the endpoint detections
in the weak and changed cases. This is a deterministic post-benchmark combination,
not a change to the frozen per-method experiment.

The independent audit finds the simultaneous target-moment event in 1,595 of
1,600 datasets, with reference-calibration and gamma coverage in all 1,600.
No reported entropy lower bound exceeds the simulator entropy. These are
separate finite simulation diagnostics, not a proof of uniform coverage.
The mathematical per-dataset failure budget is 0.05 under the detector contract;
the enclosed simulator adds less than 1.8e-11 total-variation discrepancy per
dataset under the ideal independent-random-bit sampling assumption. No Brownian
path mesh is used: finite exposure moments have rational numerical enclosures.

The complementary rare-region theorem explains why structural assumptions
cannot simply be deleted. Even ideal continuous records, a common ellipticity
lower bound and a common mixing gap cannot uniformly upper-calibrate arbitrary
unseen global diffusivity or generator suprema over the stated smooth class.
This does not prevent local volatility recovery along an ideal observed path.

New installed API: constructive_access.detector_calibration.certification
Additional installed modules: detector_calibration.exposure_moments and
detector_calibration.rare_bump within constructive_access.

    python examples/calibration_tutorial.py --output results/calibration_tutorial.json
    python verify.py
    python research/clean_reproduction.py

The tutorial exactly replays one retained simulated dataset through the public
API; it does not generate or present laboratory measurements. Proofs, the frozen
contract, retained counts, source audit and independent internal review are in
detector_calibration/. CLAIMS.json separates the new mathematical and synthetic
evidence from the earlier cumulative work. The literature audit credits the
established replica, deconvolution, concentration and diffusion-inference methods.
No new laboratory validation, external peer-review decision, historical priority
or universal physical law is claimed.

The earlier nine editions and their scientific scope are retained below.
-----------------------------------------------------------------------

NINTH EDITION — CALIBRATED DETECTOR ACCESS
Abdinoor / 2D Labs

The cumulative reference begins at basic access and now includes detector
exposure, noise, finite-trial inference and a declared measurement-energy model.
Begin the new layer with the chapter "From ideal access to a physical detector".

New example: python examples/detector_tutorial.py --output results/detector_tutorial.json
New API: constructive_access.detector.counts
Proofs, source audit, countermodels and internal review: detector/

The count table evaluates analytical inequalities at high numerical precision;
it contains no simulated or measured outcomes. A separate exact-rational review
certificate proves one finite-count separation. Noise, exposure, preparation,
and the energy/precision relation are explicit calibration assumptions.

Reproduce offline: python verify.py
Clean source and separately installed wheel: python research/clean_reproduction.py
Both drivers pin numerical BLAS threads to one. Every prior edition remains
unchanged. No external review decision, laboratory validation or historical
priority is claimed by the new detector layer.

The earlier eight editions and their scientific scope are retained below.
-----------------------------------------------------------------------

EIGHTH EDITION — OBSERVED DIFFUSION GEOMETRY
Abdinoor / 2D Labs

Begin with manuscript/reference.pdf, Chapter "What a reversible signal can
require of its hidden dynamics". The cumulative entry chapters remain first.

New API: constructive_access.observed_geometry.finite_resolution
Consistency design: constructive_access.observed_geometry.consistency
Covered fallback: constructive_access.observed_geometry.combination
New example: python examples/geometry_tutorial.py --output results/geometry_tutorial.json
New designs, records, proofs and review: geometry/

Full offline source regeneration: python verify.py
Clean source plus separately installed wheel: python research/clean_reproduction.py
Both drivers pin numerical BLAS threads to one. Exact arithmetic inputs use
integers or rational strings; empirical Bernstein needs exact score sums and
squares of the same bounded quantized records.

The sharp population minimum allows arbitrary finite hidden dimension within
the declared ordinary smooth diffusion class. The finite-record certificate
additionally needs calibrated pointwise volatility and generator bounds.
It does not infer those premises from the supplied triples. No new physical
measurement, external peer-review decision or historical priority is claimed.

The earlier seven editions and their scientific scope are retained below.
-----------------------------------------------------------------------

Constructive Access
A Unified Reference for Inference under Restricted Physical Access
Abdinoor / 2D Labs / 12 October 2026

START HERE

Read manuscript/reference.pdf, beginning with Chapter 1. This single reference
starts with a two-state experiment and develops the common theory through
finite classical and quantum instruments, calibration, statistical decisions,
model adequacy, continuous dynamics, physical energy, and worked applications.
Edition 2 added six connected chapters on finite-pulse response, its conditional
quantum boundary, calibrated tests, graph access completion, energy design and
composition. The final chapters supply applications, a target-specific workflow,
software guidance, research lineage and the next physical validation gate.

The usable object is an access contract: the target, candidate mechanisms,
preparation, allowed controls and records, calibration, error objective, and
physical resources. The output is an obstruction, an executable distinguishing
experiment, a precision certificate, a retained target set, or explicit
abstention when the contract cannot support a conclusion.

SIXTH EDITION: CALIBRATION, ACQUISITION AND DISSIPATION LIMITS

This cumulative edition adds exact calibrated comparison classes, nonlinear
information bounds with dual/primal checks, a calibrated-activity entropy
bridge, a scoped overdamped-diffusion transfer, impossibility results for total dissipation, predictable sequential
tests, dependent-record sampling and direct bounded analog scores. Matched
simulations retain stronger baselines and unfavorable regimes. The joint
calibration test treats a true same-state pair as an additional observation
instead of spending a separate confidence budget on a preliminary interval.

    python -m pip install '.[simulation,empirical]'
    python examples/calibrated_tutorial.py --output results/calibrated_tutorial.json
    python verify.py
    python research/clean_reproduction.py

The calibrated subpackage is available through constructive_access.calibrated.
Frozen designs are in calibrated/contracts; retained simulation statistics and
results are in calibrated/results. Proofs begin with "What additional
calibration can buy". CLAIMS.json distinguishes mathematical, simulated and
descriptive measured-data evidence. Internal checks are not external peer review.

Three unchanged public optical-tweezer data files accompany the release under
CC BY 4.0; see calibrated/data/source_audit/ATTRIBUTION.txt. Their HDF5 loader
reads only verified numeric datasets, never embedded pickles or source code.
The original file named groundtruth is a lower-noise measurement, not the
latent trajectory. Reanalysis is exploratory and explicitly abstains from an
entropy certificate because independent apparatus calibration is incomplete.

    python calibrated/code/empirical_loader.py
    python calibrated/code/empirical_analysis.py

The installed loader accepts an explicit data directory; the research data are
in this archive, not the Python wheel. The provenance manifest also records
sources inspected during research that are not redistributed or required for
offline reproduction. The selected experimental files are listed in ATTRIBUTION.

All inference is conditional on the declared access contract. Neither a useful
lower bound nor a sharper numerical relaxation measures total dissipation by
itself. Evolution time and detector counts are separate resources; uncalibrated
hardware energy, reset and preparation overheads are not silently set to zero.

FIFTH EDITION: VIRTUAL LABORATORIES AND APPARATUS CORRECTIONS

The new virtual laboratories generate physical trajectories and noisy readouts,
then analyze only declared observations. They test timing noise, finite exposure,
serial dependence, detector memory, state preparation, drive miscalibration and
classical null controls. Every number is simulated or mathematically derived;
none is a newly measured apparatus result. Designs/seeds, failed regimes,
Monte Carlo uncertainty, pilot cost and matched comparisons are retained.

    python -m pip install '.[simulation]'
    python examples/virtual_tutorial.py --output results/virtual_tutorial.json

The general improvements are:
- An exact criterion for clock laws that preserve reversible correlation bounds.
  Common independent increments also preserve classical dimension; random-scale
  clocks need not. Disjoint symmetric detector exposures use inter-block gap
  indices and the full physical duration in the entropy denominator.
- Fixed-quota confidence under a declared conditional-mean envelope handles
  dependent records. Marginal stationarity does not justify independent errors.
  A simulated equilibrium system supplies a strong false-positive control.
- Direct fixed polynomial contrasts yield energy certificates with exact
  confidence corrections and closed-form optimal comparison weights. They
  require predeclared or independently selected contrasts; the older joint-box
  method supports greater post-acquisition flexibility.
- A common clock correction repairs coherent phase error in the dimension
  witness. Its pilot cost is charged and equal-total-record baselines retained.

The full reproducibility driver replays all frozen simulations and compares
against independent physical solvers. The new chapters begin with "What a
virtual laboratory can establish". Exact proofs and numerical experiments have
separate status. Historical originality and physical validation remain open.

FOURTH EDITION: RESOURCES REQUIRED BY A COMPATIBLE EXPLANATION

The unified framework now connects finite observations to three distinct
requirements: hidden classical states, nonequilibrium preparation, and sustained
dissipation. Explicit countermodels show when access cannot separate them.

1. A robust temporal probability-table certificate gives a hierarchy of
   classical state requirements. One preparation and one detector at 13 times
   can certify at least five states using 18,300 independent records in the
   specified high-coherence qubit design. Both planned errors are below 5%.
2. Six Erlang-clock settings give a dimension certificate without an upper
   hidden-rate bound or exact pole fitting. The supplied lower-coherence
   example has a conservative 135,780,096-record plan; it is a mathematical
   extension, not evidence of practical superiority over the first design.
3. Finite fresh-reset endpoint experiments admit arbitrarily accurate finite,
   irreducible detailed-balance classical approximations if hidden states,
   transition rates and preparation resources are unrestricted. Their zero
   stationary entropy production does not make preparation free.
4. Stationary noninvasive correlations can require sustained entropy production
   independent of hidden-state count. More generally the observable certificate
   J requires D(preparation||stationary reference) + half the expected integrated
   housekeeping entropy >= J. A separate stationary theorem improves half to
   one quarter. Physical heat interpretation needs a thermodynamic embedding.
5. Exact continuous-domain dual certificates use a valid simultaneous confidence
   box, rational test scores and polynomial bounds. A three-lag curvature
   example yields positive information while reversal and monotonicity tests
   yield zero. Counts/boxes in the examples are explicitly synthetic.

    python examples/frontier_tutorial.py --output results/frontier_tutorial.json

The installed package constructive_access.frontier contains hierarchy, spectral,
limits, thermodynamic and energy_certificate. The numerical score proposer in
energy_certificate.design additionally needs NumPy and SciPy; its proposed scores
are always subjected to an exact continuous-domain check. Certifying fixed scores
requires no numerical optimizer. The mathematical entry is Chapter "What a
compatible explanation must contain"; full proofs and prior-art positioning
follow it. Historical novelty, experimental advantage and external review remain
open. This is a release of scoped mathematical results and usable software.

THIRD EDITION: CLASSICAL REALIZATION AND EXPLANATION BOUNDS

A directly observable inequality now bounds classical realization size:
|C0 - CT| <= 1 for every common homogeneous continuous-time Markov model with
at most three hidden states. C0 compares two preparations at a reference time;
CT compares them at an independent uniform observation time. Preparations may
be arbitrary and the fixed binary detector may be noisy. A specified noisy
qubit violates this bound. Four-state examples show why this observation alone
cannot exclude unrestricted classical models or establish a full quantum law.
The manuscript compares established dynamical dimension-witness methods.

The third-edition sharp theorem relates integrated occupation contrast to reset mismatch
and uniform entry/exit intensity caps. A four-state model attains the boundary.
Its unrestricted-reset endpoint covers history-dependent classical processes
with bounded conditional crossing hazards. Counts can now give quantitative
confidence bounds on the classical reset defect or peak intensity required.
These are model requirements, not unconditional quantum, entropy or energy
measurements. Reference starts inside the detected region; the basic inverse
example starts its comparison outside. The mixed-preparation theorem states
the changes needed for nonzero comparison occupancy.

    python examples/robust_tutorial.py --output results/robust_tutorial.json

The tutorial verifies a gap above 0.144 in a specified depolarizing-qubit model,
an exact 600-record fixed test, a 1100-record sufficient anytime horizon, and
conditional explanation bounds for explicitly synthetic 343/600 counts.
It also verifies a dimension witness above 1.08 and a 6,000-record four-arm
plan: at least 4,560 encoded successes reject the calibrated three-state null.
Both planned error probabilities are below 5% under the supplied contracts;
this is a prospective design, not an observed laboratory result.
The package constructive_access.robust contains preparation, detector_boundary,
robust_physics, clock_design, statistics_design, and explanation modules.
Some physical model APIs require Fraction arguments; executable examples show
exact types. Those examples use synthetic data. The sixth-edition empirical
chapter separately reanalyzes previously published optical-tweezer measurements.

The classical occupation principle follows established first-passage theory.
Existing nonlinear survival tests outperform our simplest assay in the supplied
matched comparison. The candidate contributions are the scoped integrated dimension witness,
sharp reset/intensity interpolation, calibrated inverse interpretation and
reproducible operational integration. Historical priority and physical validation remain unresolved.

EARLIER PHYSICAL AND ACCESS-COMPLETION RESULTS

The all-time finite-pulse response theorem covers a locally specified
intervention in arbitrary finite classical Markov networks. Its proof uses a
positive first-hitting kernel. A driven qubit crosses the protected baseline
above an exactly certified coupling/damping threshold. Classical coarse
detectors, nonstationary preparation and released-pulse delays supply essential
failure controls. This is a conditional classical-model test, not device-
independent quantumness or a newly established universal physical law.

    python examples/completion_tutorial.py --output results/completion_tutorial.json

The installed APIs in constructive_access.completion include exact pulse
certificates, a predeclared three-setting statistical test, graph cuts and
channel ambiguity, calibration nuisance cancellation, optimal fixed-cut energy
allocation, and checkable global energy brackets. The worked statistical plan
requires 110192 records per arm under its stated calibration; its packaged
counts are explicitly synthetic. Energy units require an external apparatus
calibration. A numerical illustration is in manuscript/completion_results.pdf.

Full proofs are in the reference and completion/research/. Standalone tested
source is in completion/code/; the installable copy has explicit import-change
provenance. Historical priority remains unresolved and no new laboratory
measurements or external peer-review decisions are supplied.

INSTALL AND RUN

Python >=3.10. The core library needs SymPy and mpmath.

    python -m pip install .
    constructive-access analyze examples/active.json -o results/active_report.json
    constructive-access verify examples/active.json results/active_report.json

For a source-tree run without installation:

    PYTHONPATH=src python -m constructive_access analyze examples/active.json

The active example finds drive followed by read, charge 3, with complete record
probabilities 3/4 and 1/4. It independently checks that no cheaper word distinguishes
the declared models. Twelve iid records suffice for the stated equal-prior
average-error target 0.1; preparation adds its declared overhead. Charges in this
example count operations. They are not measured energy.

Compare the failure/control contracts:

    constructive-access analyze examples/passive.json
    constructive-access analyze examples/calibrated.json
    constructive-access analyze examples/calibration_gap_closed.json
    constructive-access analyze examples/grammar.json

These exercise exact passive equivalence, robust calibrated inference, a gap
closed by uncertainty, and restrictions on allowed action sequences. A witness
outcome is an event within a complete experiment, not permission to discard
other outcomes or their cost.

A COMPLETE MULTI-MODEL TARGET DECISION

    python examples/suite_tutorial.py --output results/suite_tutorial.json

Five candidate mechanisms share two target labels. Neither of two individual
probes resolves the target. Their selected combination does, while leaving two
identical mechanisms unresolved. The example computes a minimum selection-charge
suite in the supplied finite menu, a conservative 556-record sampling certificate,
and a retained target from explicitly synthetic counts. It does not claim that
this selection objective minimizes total repeated-record charge or energy.

The public functions design_suite, verify_suite and infer_target bind the
candidate target labels, complete record laws, calibration radii and requested
sampling tolerance. Exact set-cover computation has explicit size caps. It
returns unresolved pairs when the robust menu cannot cover the target.

CALIBRATED TARGET AND ENERGY EXAMPLES

    python examples/advanced_tutorial.py --output results/advanced_tutorial.json

The selective-response example recovers one channel parameter as 3 with worst-case
error 1/25 under the full stated error box. Its independent primal/dual certificate
checks both achievable precision and its lower bound. An energy example produces
a conditional gross-work cap 37 from an externally justified counting-process and
apparatus contract. No physical calibration is inferred from these calculations.

PUBLIC API

    load_contract, analyze, verify_report
    RationalInstrument, LinearRepresentation, ActionDFA
    compare_models, compare_linear_representations, minimum_cost_witness
    experiment_law, evaluate_experiment
    certified_wait, protocol_tv_radius, robust_word_gap
    optimal_design, best_fixed_design
    AdequacyMonitor, Interval, interval_bet
    identify_linear_target, minimax_certificate, verify_minimax_certificate
    count_cap, verify_count_cap, declared_energy_cap
    design_suite, verify_suite, infer_target, SuiteError

The JSON workflow supports two finite classical rational instruments. Advanced
Python APIs expose broader linear representations, CTMC approximations, bounded
classical policy design, sequential adequacy and linear-target recovery. A signed
representation alone does not establish a physical quantum instrument. The
finite-menu suite accepts complete candidate record laws from any justified
physical source. Each API's assumptions are stated in the relevant chapter and docstrings.

REPRODUCE AND CHECK

    python -m unittest discover -s tests -v
    python verify.py

Full reference reproduction additionally requires the packages in
requirements-reproduce.txt and a working pdflatex installation. The verifier
makes no network request and does not install dependencies. It runs all tests,
all five JSON workflows and independent verifications, all tutorials, separate
symbolic physics checks, figure generation and repeated LaTeX compilation. It
checks vendored-source hashes, unchanged scientific inputs, and document warnings.
Detailed command logs and hashes are written to results/verification.json.

    python research/clean_reproduction.py

The latter command builds and installs a wheel offline in a separate environment,
repeats the full reference build from a clean source copy, and compares scientific
JSON outputs and PDF text. It uses already installed dependencies.

The release includes a built wheel in dist/. Clean installation and reproduction
results are recorded separately. The library can be used without the plotting
or typesetting dependencies and without any earlier research directory.

WHAT IS AND IS NOT ESTABLISHED

This is a unified mathematical and computational reference, not a claim that
science has accepted a new physical law or that the four application grand
challenges have been solved. Core mathematical tools are attributed to existing
work. Specialized current, quantum and chemical statements are proved within
explicit model classes. New integration supplies a common contract, target-set
handoff, finite-family experimental workflow, independent verification, and
consistent failure outputs.

Internal adversarial review and exact reproduction are documented. External peer
review, newly measured physical data and independent laboratory deployment are
absent. The chapters specify concrete validation tests for each application.
Unknown hidden degrees of freedom, incorrect model families, uncalibrated controls
and omitted apparatus expenditure remain physical questions, not facts certified
by a probability calculation.

RELEASE CONTENTS AND LINEAGE

    manuscript/reference.pdf       Canonical unified reference
    manuscript/reference.tex       Main source and included chapters
    src/constructive_access/       Standalone installable research toolkit
    examples/                      Contracts and complete executable tutorials
    tests/                         Implementation and independent checks
    research/                      Internal reviews, source audit, exact checks
    results/                       Reproducible reports and release evidence
    CLAIMS.json                    Scoped claim and evidence map
    research/provenance.json        All 68 inherited ledger entries, 20 result groups
    src/.../_vendor/PROVENANCE.json Seven preserved-module source mappings
    SWARM.md, logs/events.jsonl      Actual work record and continuation protocol

The cumulative distribution also preserves physical_diagnostics_research_package.zip
with SHA-256 bbaaaef9f83b4ebf8c25555695cecdf4512568b5aebaea18a3639c5bafadea32.
That nested archive contains the full earlier research lineage, including original
boundary classifications and specialized derivations. It is provenance material;
the new library and reference entry path do not require extracting it.


SEVENTH-EDITION DIRECT INFERENCE AND PHYSICAL ATTAINABILITY
The cumulative reference now includes a direct finite-lag entropy bridge,
joint bounded analog inference, a simultaneous contrast library, exact
finite-size circulant minima and temporal-aliasing impossibility results.

Run the complete new entry example after installing the local wheel:
  python examples/attainability_tutorial.py --output results/attainability_tutorial.json

For full offline source reproduction (all earlier stages retained):
  python verify.py

New APIs are in constructive_access.attainability. Exact confidence methods
accept rational inputs; physical_minimum routines explicitly return floating
model calculations. The joint analog result retains the prior quadratic key
for comparisons; best_entropy_production_rate_lower includes the direct bound.
A positive certificate remains conditional on the declared apparatus contract.
New frozen designs, records, proofs and independent review are in attainability/.

The larger guarantee does not mean a 140-fold increase in measured dissipation:
it is a stronger lower bound computed from the same simulated evidence.
All original data attribution and earlier theoretical scope remain applicable.
