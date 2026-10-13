"""Strict, portable specifications for the supported two-model workflow."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
from collections.abc import Mapping

from ._vendor.accesskit import ActionDFA, RationalInstrument

SCHEMA = "constructive-access-contract-v1"


class ContractError(ValueError):
    """A specification is invalid or outside the supported workflow."""


def _object(value, required, optional=(), where="object"):
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    if not set(required) <= set(value):
        raise ContractError(f"{where}: missing keys {sorted(set(required)-set(value))}")
    unknown = set(value)-set(required)-set(optional)
    if unknown:
        raise ContractError(f"{where}: unknown keys {sorted(unknown)}")


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a nonempty string")
    return value


def _q(value, where, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ContractError(f"{where}: use an integer or exact rational string, never a float")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ContractError(f"{where}: invalid rational value") from exc
    if (lower is not None and result < lower) or (upper is not None and result > upper):
        raise ContractError(f"{where}: rational value outside allowed range")
    return str(result)


def _int(value, where, lower):
    if type(value) is not int or value < lower:
        raise ContractError(f"{where}: integer >= {lower} required")
    return value


def _strings(value, where, nonempty=False):
    if not isinstance(value, list) or (nonempty and not value):
        raise ContractError(f"{where}: list of strings required")
    return [_text(v, where) for v in value]


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    """Read strict JSON; duplicate keys and nonfinite numeric literals fail."""
    def invalid_constant(value):
        raise ContractError("Nonfinite JSON number: " + value)
    try:
        return json.loads(Path(path).read_text(), object_pairs_hook=_pairs_no_duplicates,
                          parse_constant=invalid_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"Cannot read JSON {path}: {exc}") from exc


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Contract:
    """Immutable canonical specification; ``to_dict`` returns a fresh copy."""
    canonical_json: str

    @property
    def sha256(self):
        return sha256(self.canonical_json.encode()).hexdigest()

    def to_dict(self):
        return json.loads(self.canonical_json)

    def instruments(self):
        data = self.to_dict()
        return tuple(RationalInstrument(m["initial"], m["actions"], m["name"])
                     for m in data["models"])

    def grammar(self):
        raw = self.to_dict().get("grammar")
        if raw is None:
            return None
        return ActionDFA(raw["states"], raw["start"], raw["accepting"],
                         {(e["state"], e["action"]): e["target"] for e in raw["transitions"]})


def load_contract(source):
    """Validate a JSON path, mapping, or Contract and return canonical data.

    Normalization preserves the declared meanings and assumptions. Mathematical
    validation does not establish physical calibration or model completeness.
    """
    if isinstance(source, Contract):
        source = source.to_dict()
    elif isinstance(source, (str, Path)):
        source = read_json(source)
    elif isinstance(source, Mapping):
        source = dict(source)
    # Roundtrip detaches caller data while preserving JSON numeric types.
    try:
        d = json.loads(canonical_json(source))
    except (TypeError, ValueError) as exc:
        raise ContractError("Contract must contain JSON data only") from exc
    _object(d, {"schema", "name", "question", "target", "models", "resources", "preparation", "design"},
            {"grammar", "calibration"}, "contract")
    if d["schema"] != SCHEMA:
        raise ContractError("Unsupported contract schema")
    for key in ("name", "question"):
        _text(d[key], key)
    target = d["target"]
    _object(target, {"name", "labels"}, where="target")
    _text(target["name"], "target.name")
    labels = _strings(target["labels"], "target.labels")
    if len(labels) != 2 or labels[0] == labels[1]:
        raise ContractError("Exactly two distinct target labels are required")
    if not isinstance(d["models"], list) or len(d["models"]) != 2:
        raise ContractError("Exactly two classical candidate models are supported")
    names = []
    models = []
    for index, model in enumerate(d["models"]):
        where = f"models[{index}]"
        _object(model, {"name", "initial", "actions"}, where=where)
        names.append(_text(model["name"], where+".name"))
        initial = model["initial"]
        if not isinstance(initial, list) or not initial:
            raise ContractError(where+".initial must be a nonempty vector")
        model["initial"] = [_q(x, where+".initial", 0, 1) for x in initial]
        n = len(initial)
        actions = model["actions"]
        if not isinstance(actions, dict) or not actions:
            raise ContractError(where+".actions must be a nonempty object")
        for action, outcomes in actions.items():
            _text(action, "action")
            if not isinstance(outcomes, dict) or not outcomes:
                raise ContractError("Every action requires explicit outcomes")
            for outcome, matrix in outcomes.items():
                _text(outcome, "outcome")
                if (not isinstance(matrix, list) or len(matrix) != n or
                        any(not isinstance(row, list) or len(row) != n for row in matrix)):
                    raise ContractError(where+": kernel dimension mismatch")
                outcomes[outcome] = [[_q(x, where+".kernel", 0, 1) for x in row] for row in matrix]
        try:
            models.append(RationalInstrument(model["initial"], actions, model["name"]))
        except (TypeError, ValueError) as exc:
            raise ContractError(where+": "+str(exc)) from exc
    if len(set(names)) != 2:
        raise ContractError("Candidate model names must differ")
    if set(models[0].representation.kernels) != set(models[1].representation.kernels):
        raise ContractError("Models require identical explicit action/outcome alphabets")
    r = d["resources"]
    _object(r, {"charge_name", "unit", "meaning", "action_costs", "per_record_overhead",
                "calibration", "included", "excluded"}, where="resources")
    for key in ("charge_name", "unit", "calibration"):
        _text(r[key], "resources."+key)
    if r["meaning"] not in ("operation_count", "duration_upper_bound", "energy_upper_bound", "abstract_charge"):
        raise ContractError("Unsupported resource meaning")
    _strings(r["included"], "resources.included", True)
    _strings(r["excluded"], "resources.excluded")
    if not isinstance(r["action_costs"], dict) or set(r["action_costs"]) != set(models[0].actions):
        raise ContractError("One action charge is required for each accessible action")
    r["action_costs"] = {a: _q(v, "action charge", 0) for a, v in r["action_costs"].items()}
    r["per_record_overhead"] = _q(r["per_record_overhead"], "per-record overhead", 0)
    p = d["preparation"]
    _object(p, {"description", "iid_records"}, where="preparation")
    _text(p["description"], "preparation.description")
    if type(p["iid_records"]) is not bool:
        raise ContractError("preparation.iid_records must be a Boolean assumption declaration")
    design = d["design"]
    _object(design, {"target_average_error"}, {"max_transcripts", "precision_bits", "max_precision_bits"}, "design")
    design["target_average_error"] = _q(design["target_average_error"], "target average error", 0, Fraction(1, 2))
    if not 0 < Fraction(design["target_average_error"]) < Fraction(1, 2):
        raise ContractError("Target average error must be strictly between zero and one half")
    for key, default, lower in (("max_transcripts", 100000, 1), ("precision_bits", 160, 8), ("max_precision_bits", 2560, 8)):
        design[key] = _int(design.get(key, default), "design."+key, lower)
    if design["max_precision_bits"] < design["precision_bits"]:
        raise ContractError("Maximum precision must be at least initial precision")
    if "grammar" in d:
        g = d["grammar"]
        _object(g, {"states", "start", "accepting", "transitions"}, where="grammar")
        states = _strings(g["states"], "grammar.states", True)
        accepting = _strings(g["accepting"], "grammar.accepting")
        if len(set(states)) != len(states) or len(set(accepting)) != len(accepting):
            raise ContractError("DFA states and accepting states must be unique")
        if g["start"] not in states or not set(accepting) <= set(states):
            raise ContractError("DFA endpoints must be declared states")
        if not isinstance(g["transitions"], list):
            raise ContractError("grammar.transitions must be a list")
        seen = set()
        for e in g["transitions"]:
            _object(e, {"state", "action", "target"}, where="grammar transition")
            if e["state"] not in states or e["target"] not in states or e["action"] not in models[0].actions:
                raise ContractError("Invalid DFA transition")
            key = (e["state"], e["action"])
            if key in seen:
                raise ContractError("Duplicate DFA transition")
            seen.add(key)
        g["transitions"] = sorted(g["transitions"], key=lambda e: (e["state"], e["action"]))
    if "calibration" in d:
        c = d["calibration"]
        _object(c, {"record_tv_radii", "scope", "justification", "joint_failure_probability"}, where="calibration")
        if c["scope"] != "whole_record_for_returned_protocol":
            raise ContractError("Calibration must bound the complete returned fixed-protocol record")
        _text(c["justification"], "calibration.justification")
        if not isinstance(c["record_tv_radii"], list) or len(c["record_tv_radii"]) != 2:
            raise ContractError("One whole-record TV radius per candidate is required")
        c["record_tv_radii"] = [_q(v, "record TV radius", 0, 1) for v in c["record_tv_radii"]]
        c["joint_failure_probability"] = _q(c["joint_failure_probability"], "calibration failure probability", 0, 1)
    return Contract(canonical_json(d))
