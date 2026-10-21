#!/usr/bin/env python3
"""Single-command Codex entry point for the generative modeling agent."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def verify_outputs(output_dir, report_name):
    """Check the minimum artifacts for a completed generative run."""
    required = ["data_profile.json", "objective_decision.json", "model_selection.json", report_name]
    missing = [name for name in required if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0]
    model_runs = []
    for child in output_dir.iterdir():
        if not child.is_dir() or child.name.endswith("_smoke"):
            continue
        config = child / "training_config.json"
        metrics = child / "evaluation" / "evaluation_metrics.json"
        synthetic = [child / "synthetic_counts.csv", child / "synthetic_images.npy"]
        if config.exists():
            model_runs.append(child.name)
            if not metrics.is_file():
                missing.append(f"{child.name}/evaluation/evaluation_metrics.json")
            if not any(path.is_file() and path.stat().st_size > 0 for path in synthetic):
                missing.append(f"{child.name}/synthetic_counts.csv or synthetic_images.npy")
    if not model_runs:
        missing.append("at least one completed model directory with training_config.json")
    return missing, model_runs


def main():
    parser = argparse.ArgumentParser(description="Run the Codex generative modeling agent.")
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing train.csv and eval.csv")
    parser.add_argument("--objective", required=True, help="User's generative modeling objective")
    parser.add_argument("--output-dir", help="Directory for generated outputs (default: outputs/run_<dataset>)")
    parser.add_argument("--dry-run", action="store_true", help="Show the prompt without invoking Codex")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if not (root / "AGENTS.md").is_file():
        parser.error(f"AGENTS.md not found in {root}; place run_agent.py at repository root")

    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {data_dir}")
    for name in ("train.csv", "eval.csv"):
        if not (data_dir / name).is_file():
            parser.error(f"Missing {name} in {data_dir}")

    if not args.objective.strip():
        parser.error("--objective cannot be empty")

    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path("outputs") / f"run_{data_dir.name}"
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir = output_dir.resolve()
    # Workspace-write sandbox can write within the repository, not arbitrary filesystem locations.
    if not output_dir.is_relative_to(root):
        parser.error("--output-dir must be inside the repository for workspace-write sandbox")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}; choose a new directory to preserve prior runs")

    suffix = data_dir.name.removeprefix("dataset_")
    report_name = f"generated_report_{suffix}.md" if suffix.isdigit() else "generated_report.md"
    prompt = f"""Execute the complete generative modeling agent workflow for this run.

Dataset directory: {data_dir}
User objective: {args.objective}
Output directory: {output_dir}
Final analysis report: {output_dir / report_name}

Follow AGENTS.md and the relevant .agents/skills/ instructions. Inspect the dataset,
interpret the objective, select preprocessing, model(s), hyperparameters and evaluation
criteria autonomously, run smoke tests and training when appropriate, generate outputs,
evaluate results, make downstream recommendations, and write the dataset-specific final report.
Do not create or modify report/report.pdf during this dataset run.
Do not automatically train every model. Do not ask the user to make modeling decisions.
Use the existing analysis scripts; do not change source code or input datasets.
Preserve all existing outputs. Do not change sandbox permissions or attempt to bypass
restrictions. If GPU acceleration is inaccessible within the sandbox, use CPU when
feasible; otherwise clearly report any execution blocker. Do not claim success for
steps that did not complete. Verify that the reported artifacts actually exist.
"""

    if args.dry_run:
        print(prompt)
        return 0

    if shutil.which("codex") is None:
        print("Error: 'codex' CLI not found on PATH. Install/enable Codex CLI first.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["codex", "exec", "--sandbox", "workspace-write", "-"]
    print(f"Running Codex from {root}\nOutputs: {output_dir}\n", flush=True)
    try:
        result = subprocess.run(cmd, input=prompt, text=True, cwd=root, check=False)
    except OSError as exc:
        print(f"Failed to launch Codex: {exc}", file=sys.stderr)
        return 2
    if result.returncode:
        print(f"Codex exited with status {result.returncode}; inspect output for incomplete steps.", file=sys.stderr)
        return result.returncode
    missing, models = verify_outputs(output_dir, report_name)
    if missing:
        print("Codex exited successfully, but required artifacts are missing:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"Completed and verified output structure: {output_dir} (models: {', '.join(models)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
