# Virtual laboratory research

Abdinoor / 2D Labs. Inherit ../AGENTS.md. Preserve every prior release.
Goal: stress-test and improve Constructive Access using independently implemented
physical simulators, apparatus imperfections, adversarial controls and matched
statistical comparisons. Simulation evidence is not laboratory validation.

Each agent owns its assigned name in code/, tests/, research/, results/ and
manuscript/. No changes to prior releases. Root owns shared contracts, integration,
logs and release. New modules must run from the phase directory and permit later
packaging in constructive_access.virtual. Use explicit seeds, persist experiment
specifications before the production simulation, retain failed regimes, and report
Monte Carlo uncertainty. Separate calibration, pilot design and held-out tests.
Do not treat example physical rates as empirically calibrated hardware values.

Frozen fourth-edition code is available at
../realization_frontier/release/src/constructive_access. Root will copy needed
modules into the cumulative release; avoid hard-coded absolute imports in code.
For development set PYTHONPATH to that source root. Agent code can use stdlib,
NumPy, SciPy, SymPy and mpmath already installed. Exact inference modules retain
rational checking. Simulator solvers may be numerical with cross-checks.

Mandatory controls include equilibrium/null examples, favorable alternatives,
noise/misspecification that reduces power, and outside-contract examples that can
mislead an unqualified interpretation. Where adapting a protocol, use matched
access, record count and time accounting. A baseline that solves a different
inference task is not a fair performance comparison.

No external messages, publication, purchase or laboratory operation is authorized
by this local simulation request. Record meaningful actual events only.
