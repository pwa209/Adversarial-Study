"""Constrained configuration banks and explicit external NFTsim execution.

The local baseline has one spatial node. This adapter cannot establish a scalp
forward map, region-resolved dynamics, functional restoration, or a fitted human
mechanism. It only preserves and varies interpretable scalar configuration fields.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .core import StudyError, number, sha256_file, write_csv, write_json

FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _without_comments(text: str) -> str:
    # Preserve character offsets and line breaks, so edits target original code.
    def mask(match: re.Match) -> str:
        return "".join("\n" if c == "\n" else " " for c in match[0])
    return re.sub(r"/\*[\s\S]*?\*/|//[^\n]*", mask, text)


def _replace_parameter(text: str, key: str, value: float) -> str:
    """Supported syntax: 'Coupling N.nu', addressing scalar 'Map - nu:'."""
    match = re.fullmatch(r"Coupling ([1-9]\d*)\.nu", key)
    if not match:
        raise StudyError(f"Unsupported NFTsim parameter {key}; supply an explicitly reviewed adapter")
    pattern = re.compile(r"(^[ \t]*Coupling " + match[1] + r":[ \t]*Map[ \t]*-[ \t]*nu:\s*)(" + FLOAT + r")(?=[ \t]*(?:\n|$))", re.MULTILINE)
    hits = list(pattern.finditer(_without_comments(text)))
    if len(hits) != 1:
        raise StudyError(f"Parameter {key} must address exactly one scalar Map - nu field")
    position = hits[0]
    return text[:position.start(2)] + format(value, ".17g") + text[position.end(2):]


def prepare_nftsim_bank(template_path: Path, parameter_sets: list[dict], output_dir: Path,
                        bounds: dict[str, list[float]] | None = None) -> Path:
    """Require empirical/prior-supported bounds; never optimize score alone."""
    if not parameter_sets or not bounds:
        raise StudyError("Provide candidate parameters and explicit, justified bounds")
    text = Path(template_path).read_text(encoding="utf-8-sig")
    node_matches = list(re.finditer(r"^Nodes:[ \t]*(\d+)[ \t]*$", _without_comments(text), re.MULTILINE))
    if len(node_matches) != 1 or int(node_matches[0][1]) < 1:
        raise StudyError("Template must declare exactly one positive spatial node count")
    node_match = node_matches[0]
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise StudyError("Simulator bank destination must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, parameters in enumerate(parameter_sets):
        if not parameters:
            raise StudyError("A candidate must declare its changed parameters")
        candidate = text
        for key, supplied in sorted(parameters.items()):
            value = number(supplied, key)
            if key not in bounds or len(bounds[key]) != 2:
                raise StudyError(f"Missing explicit bounds for {key}")
            lower, upper = [number(v, f"{key}.bound") for v in bounds[key]]
            if lower > upper or not lower <= value <= upper:
                raise StudyError(f"{key}={value} is outside declared bounds")
            candidate = _replace_parameter(candidate, key, value)
        path = output_dir / f"candidate_{index:05d}.conf"
        path.write_text(candidate, encoding="utf-8", newline="\n")
        rows.append({"candidate_id": f"candidate_{index:05d}", "config_path": str(path.resolve()),
                     "config_sha256": sha256_file(path), "nodes": int(node_match[1]),
                     "parameters": __import__("json").dumps(parameters, sort_keys=True)})
    manifest = output_dir / "candidates.csv"
    write_csv(manifest, rows)
    write_json(output_dir / "bank.json", {"template_sha256": sha256_file(template_path),
               "bounds": bounds, "candidates": len(rows), "status": "prepared_not_executed",
               "measurement_mapping": "not_qualified",
               "inference_scope": "simulator population dynamics; no human mechanism established"})
    return manifest


def run_external_simulator(command: list[str], output_dir: Path, timeout_s: float = 300) -> dict:
    """Explicit invocation only. No shell interpolation or automatic job submission."""
    if not command or not all(isinstance(v, str) and v for v in command):
        raise StudyError("Simulator command must be a nonempty argv list")
    executable = Path(command[0]).resolve()
    if not executable.is_file():
        raise StudyError("Use an explicit path to a qualified simulator executable")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [str(executable), *command[1:]]
    completed = subprocess.run(command, cwd=output_dir, capture_output=True, text=True,
                               timeout=timeout_s, check=False, shell=False)
    (output_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    result = {"argv": command, "executable_sha256": sha256_file(executable),
              "returncode": completed.returncode, "status": "completed" if completed.returncode == 0 else "failed",
              "measurement_mapping": "not_qualified", "human_inference": "not_established"}
    write_json(output_dir / "execution.json", result)
    if completed.returncode != 0:
        raise StudyError(f"Simulator failed with return code {completed.returncode}; inspect preserved logs")
    return result
