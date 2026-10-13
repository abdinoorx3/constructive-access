# Constructive Access

**A Unified Reference for Inference under Restricted Physical Access**  
Abdinoor · 2D Labs · First public release 1.0 · 13 October 2026  
[2dlabs.org](https://2dlabs.org) · [helloworld@2dai.org](mailto:helloworld@2dai.org)

A scientific apparatus reveals some distinctions and conceals others. This
research compendium asks which conclusions its records justify, which physical
interventions reveal a missing distinction, and what every compatible hidden
explanation must contain.

**Start with the [complete reference](manuscript/reference.pdf).** It begins
with a two-state example and develops the framework, proofs, counterexamples,
classical and quantum applications, statistical procedures and physical
resource accounting in one monograph. The accompanying toolkit is version
0.10.0. The public compendium and software use separate version numbers.

## What this release contributes

- An executable access-contract framework: specify candidate mechanisms,
  preparation, controls, records, calibration, error objective and physical
  resources; obtain an exact obstruction, a distinguishing protocol, a
  confidence certificate or explicit abstention.
- Scoped physical requirements on compatible hidden explanations, including
  classical state requirements and minimum dissipation, with countermodels
  showing where the claims fail.
- An exact hidden-diffusion result: within the stated compact, smooth,
  ordinary-reversal class, a completely reversible scalar signal can force
  positive minimum entropy production even when its hidden angular drive
  averages to zero. The proved family has minimum E[c(Z)^2]/gamma.
- A finite-data entropy certificate that jointly covers unknown diffusion
  scale, detector gain and uniform exposure width. Its pointwise diffusion
  geometry and generator bounds are explicit premises. A complementary
  impossibility theorem explains why unrestricted global calibration bounds
  cannot be obtained uniformly from finite passive records.

Established identification, realization, statistical and thermodynamic methods
are credited separately from the particular proofs and constructions developed
here. [CLAIMS.json](CLAIMS.json) records the scope and evidence status of each
claim. Historical priority and external expert peer review remain open.

## Read or run

| Purpose | Entry point |
| --- | --- |
| Learn the complete framework | [Reference PDF](manuscript/reference.pdf), starting at Chapter 1 |
| Read the research abstract | [Abstract](ABSTRACT.txt) |
| Inspect precise claims | [Claim ledger](CLAIMS.json) |
| Replay a calibrated simulated dataset | [Calibration tutorial](examples/calibration_tutorial.py) |
| Inspect the latest proofs and detector contract | [Joint calibration sources](detector_calibration/) |
| Check exact finite-instrument access | [Core examples](examples/) and the `constructive-access` command |
| Inspect verification provenance | [Public validation](PUBLIC_VALIDATION.json) and [baseline evidence](validation/scientific_baseline/) |
| Cite the release | [CITATION.cff](CITATION.cff) |
| Understand reuse permissions | [License](LICENSE) and [commercial-use guide](COMMERCIAL_USE.txt) |

Python 3.12 was used for the recorded reproduction. A small working example is:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[simulation,empirical]'
python examples/calibration_tutorial.py --output calibration_example.json
```

The tutorial replays retained **simulated** counts through exact rational
confidence calculations. It reports the geometry certificate, a broader-class
endpoint certificate, and their maximum on a common confidence event. The
combined result retains the geometry method's stronger model assumptions.
A zero lower bound means that the experiment was inconclusive.

## Evidence and limitations

The frozen detector benchmark contains 1,600 datasets. Each dataset contains
13,333,333 target trials and 200,000 reference readings, totaling 80,199,998
detector readings per dataset. Geometry detections were 400/400 for strong drive, 0/400 at equilibrium,
0/400 for weak drive and 395/400 after changing calibration. The broader-class
endpoint comparison detected 400/400, 0/400, 27/400 and 400/400 respectively.
Every selected geometry prefix was order zero; a higher-order benefit was not
demonstrated at this budget. The combined bound retains the endpoint detections.

These are virtual experiments with a bounded numerical probability-law error.
They do not constitute new laboratory measurements. The included reanalysis of
published optical-tweezer data is exploratory and does not certify entropy
because the necessary independent calibration is incomplete. Physical detector
energy, preparation and reset costs are not replaced by readout counts.

The work has not undergone external expert peer review. Proofs and reproducible
evidence are supplied for scrutiny.

## Reproduce the complete compendium

Exact package versions are in [requirements-reproduce.txt](requirements-reproduce.txt).
PDF generation additionally requires a working LaTeX installation with the
packages used by `manuscript/reference.tex`.

```sh
python -m pip install -r requirements-reproduce.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/clean_reproduction.py
```

The full drivers regenerate the required experiments, audits, figures and
reference, then compare a clean source copy and a separately installed wheel.
They perform substantial computations. The GitHub workflow runs a smaller
producer-test and installed-tutorial check; it does not substitute for the
recorded full reproduction.

The original scientific verification is archived under
`validation/scientific_baseline/`. `PUBLIC_VALIDATION.json` records how the
public package preserves those scientific files and validates the declared
editorial changes. Running the full drivers creates fresh runtime reports
without rewriting the archived baseline evidence.

To rebuild the public software distribution after scientific reproduction, run:

```sh
python publication/validate_software.py
```

This final packaging step includes the license and notices in the wheel, checks
that its scientific code matches the verified baseline, and replays the tutorial
from a fresh installed environment. Its report is
[PUBLIC_SOFTWARE_VALIDATION.json](PUBLIC_SOFTWARE_VALIDATION.json).

## Reuse and commercial licensing

Reading and citing the research are free. The covered project materials use a
custom revenue-threshold license: personal and qualifying academic research
use are free, and business use is free while worldwide annual revenue across
the company and its controlled affiliates is at or below **US$10 million**.
Commercial use above that threshold requires a separate written license from
Abdinoor / 2D Labs. Request terms at **helloworld@2dai.org** or
[2dlabs.org](https://2dlabs.org). The license does not set an automatic royalty.

The license governs the covered code and expressive material, not ownership of
scientific facts, mathematical results or independent implementations. Existing
third-party licenses remain controlling for their materials, including the
three CC BY 4.0 optical-tweezer data files. Consult the exact license and
[third-party notices](THIRD_PARTY_NOTICES.txt) before reuse. These custom terms
have not been reviewed by legal counsel.

## Provenance and contributions

This is the first public-release package. Earlier numbered editions mentioned
in retained logs and nested archives are internal research iterations and local
artifacts, not evidence of earlier public publication. The current reference,
claim ledger and public metadata are the authoritative entry points; historical
archives are preserved for provenance.

Reproduction reports, counterexamples and corrections are welcome through the
repository's issue tracker once hosted, or at **helloworld@2dai.org**. See
[CONTRIBUTING.md](CONTRIBUTING.md). No journal acceptance, DOI or repository URL
is claimed before it exists.

## Public release

Source: https://github.com/abdinoorx3/constructive-access

The public presentation edits are recorded in PUBLIC_EDITORIAL_DELTA.json.
PUBLIC_EDITORIAL_VALIDATION.json records the current PDF and scoped checks.
Historical validation records retain their original input hashes. MANIFEST.json
binds the current public files. Interpreter installation paths in retained
records are represented by [recorded-python-runtime].
