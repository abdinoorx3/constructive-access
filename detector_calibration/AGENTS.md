# Discovery research instructions

Owner: Abdinoor, 2D Labs. Started 2026-10-12 UTC.

## User objective

Pursue a major, defensible discovery concerning physical systems or computation, with a reproducible application. The aspirational endpoint is a new physical law or a comparably consequential computational result. Do not claim that this endpoint has been reached without evidence. There is no arbitrary elapsed-time success criterion.

## Non-negotiable evidence rules

- Separate established prior work, proved mathematical statements, conjectures, numerical evidence, and empirical physical evidence.
- A theorem about an assumed stochastic model is not by itself a new physical law. Simulated data are not physical experimental data. Literature searches do not prove novelty.
- State model class, observations, interventions, error criterion, resource accounting, regularity assumptions, and failure cases.
- A hidden time-reversal contrast is not the same as a hidden parameter or a universally uninformative dataset. The original paper's occupation response is a counterexample to that conflation.
- Do not assert a result about recursive self-improvement, P versus NP, arbitrary neural networks, or unrestricted physical systems from a finite-model calculation.
- Attempt disproof before expanding claims. Report negative results and closest prior art.
- No external submission, publication, messages to third parties, purchases, or laboratory actions are authorized by this research task alone. Read-only research, local code, manuscripts, and simulations are authorized.
- Preserve /workspace/boundary_arrow and its released archive. New work belongs here.

## Research loop

1. State a falsifiable candidate claim and its scientific or computational consequence.
2. Map closest prior work and the precise proposed difference.
3. Derive a small exact case; actively search for counterexamples and confounding explanations.
4. Prove the strongest surviving statement with explicit scope.
5. Build an independently checkable implementation, negative controls, and a cost-matched benchmark.
6. Assign independent adversarial mathematical and novelty review.
7. Correct, reject, or promote the claim; update the evidence ledger and next dependency.
8. Repeat on the strongest unresolved claim. Stop a branch when evidence rejects it; do not stop or relabel it as success because resources have been spent.

## Coordination and persistence

- Read SWARM.md before choosing work. Each agent owns distinct files and must not overwrite another agent's work.
- Agents write timestamped findings to their assigned research/*.txt and structured evidence where useful. Root integrates the authoritative ledger.
- Root appends events to logs/events.jsonl. Never fabricate past turns or timestamps. Log meaningful decisions, claims, tests, failures, and resumable state; avoid redundant token-by-token transcripts.
- Record commands, versions, seeds, numerical precision, outputs, and conditions needed for reproduction.
- The Markdown instructions and log are a continuation protocol, not a daemon. Do not claim that work runs outside available execution or that an uninterrupted indefinite service has been established.
- If execution is interrupted, leave a truthful checkpoint specifying completed work, running jobs, unresolved blockers, and the next concrete actions.

## Completion gates

A mathematical result requires a complete proof and an independent internal review, with external review still marked absent. A computational advantage requires a reproducible comparison against appropriate baselines at matched resources. A claimed new physical law additionally requires a distinctive physical prediction and independent empirical support; simulations alone cannot satisfy this gate. A large breakthrough is not assumed in advance.
