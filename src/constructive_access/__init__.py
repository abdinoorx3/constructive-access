"""Constructive Access: model-bound inference under restricted operations."""
from .contract import Contract, ContractError, load_contract
from .workflow import VerificationError, analyze, verify_report
from ._vendor.accesskit import (
    ActionDFA, LinearRepresentation, RationalInstrument, compare_models,
    compare_linear_representations, evaluate_experiment, experiment_law,
    minimum_cost_witness,
)
from ._vendor.ctmc_adapter import certified_wait, protocol_tv_radius, robust_word_gap
from ._vendor.budgeted_design import optimal_design, best_fixed_design
from ._vendor.model_adequacy import AdequacyMonitor, Interval, interval_bet
from ._vendor.intervention_criterion import (
    identify as identify_linear_target, minimax_certificate, verify_minimax_certificate,
)
from ._vendor.energy_tail import count_cap, verify_count_cap, declared_energy_cap
from .suite import SuiteError, design_suite, infer_target, verify_suite

__version__ = "0.10.0"

__all__ = ["Contract", "ContractError", "VerificationError", "load_contract", "analyze", "verify_report",
           "ActionDFA", "LinearRepresentation", "RationalInstrument", "compare_models",
           "compare_linear_representations", "evaluate_experiment", "experiment_law", "minimum_cost_witness",
           "certified_wait", "protocol_tv_radius", "robust_word_gap", "optimal_design", "best_fixed_design",
           "AdequacyMonitor", "Interval", "interval_bet",
           "identify_linear_target", "minimax_certificate", "verify_minimax_certificate",
           "count_cap", "verify_count_cap", "declared_energy_cap",
           "SuiteError", "design_suite", "infer_target", "verify_suite"]
