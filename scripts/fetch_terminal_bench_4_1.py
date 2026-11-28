"""Download the Terminal-Bench 4.1 leaderboard and the trials linked to each row from Harbor Hub, with the official harbor CLI.

Reading trials needs a free Harbor Hub account:

    uv tool install harbor==0.27.0   # or any environment with the harbor package
    harbor auth login
    python scripts/fetch_terminal_bench_4_1.py

Writes data/raw/terminal_bench_4_1/rows.json (the leaderboard rows) and trials/<row id>.json (one record per linked trial).
Rows already fetched are skipped, so an interrupted run can be restarted; the leaderboard itself is fetched once (fetched 2026-11-27 for the committed results).
"""

import argparse
import json
import subprocess
from pathlib import Path

LEADERBOARD = "terminal-bench/terminal-bench/4-1-0"
TRIAL_COLUMNS = "id,task,reward,error,retry,started,finished,cost,job,agent,model"


def run_harbor(harbor_binary: str, *arguments: str) -> str:
    return subprocess.run(
        [harbor_binary, *arguments], capture_output=True, text=True, check=True
    ).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harbor-binary", default="harbor")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/raw/terminal_bench_4_1")
    )
    cli_arguments = parser.parse_args()
    harbor_binary = cli_arguments.harbor_binary
    (cli_arguments.output_dir / "trials").mkdir(parents=True, exist_ok=True)

    # note (Haoling Pu): the rows are fetched once and kept, so a restart pairs the trials with the same leaderboard snapshot; delete the folder to refetch.
    rows_path = cli_arguments.output_dir / "rows.json"
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
    else:
        row_list = json.loads(
            run_harbor(
                harbor_binary,
                "hub",
                "leaderboard",
                "row",
                "list",
                LEADERBOARD,
                "--limit",
                "1000",
                "--json",
            )
        )
        rows = row_list["items"]
        if len(rows) != row_list["total"]:
            raise RuntimeError(
                f"got {len(rows)} of {row_list['total']} rows; page through them"
            )
        else:
            rows_path.write_text(json.dumps(rows, indent=1))
    trials_by_job_id: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        row_path = cli_arguments.output_dir / "trials" / f"{row['id']}.json"
        if row_path.exists():
            continue
        else:
            pass
        linked_trial_ids = run_harbor(
            harbor_binary,
            "hub",
            "leaderboard",
            "row",
            "trial",
            "list",
            row["id"],
            "--quiet",
        ).split()
        linked_trials: dict[str, dict[str, str]] = {}
        # note (Haoling Pu): a row's trials usually come from one job; one trial per job is looked up to find it, then the whole job is listed at once.
        for trial_id in linked_trial_ids:
            if trial_id in linked_trials:
                continue
            else:
                pass
            job_id = json.loads(
                run_harbor(harbor_binary, "hub", "trial", "show", trial_id, "--json")
            )["job_id"]
            if job_id not in trials_by_job_id:
                tsv_lines = run_harbor(
                    harbor_binary,
                    "hub",
                    "job",
                    "trials",
                    job_id,
                    "--columns",
                    TRIAL_COLUMNS,
                    "--no-trunc",
                    "--include-retries",
                ).splitlines()
                header = tsv_lines[0].split("\t")
                trials_by_job_id[job_id] = {
                    fields[0]: dict(zip(header, fields))
                    for fields in (line.split("\t") for line in tsv_lines[1:])
                }
            else:
                pass
            for linked_trial_id in linked_trial_ids:
                if linked_trial_id in trials_by_job_id[job_id]:
                    linked_trials[linked_trial_id] = trials_by_job_id[job_id][
                        linked_trial_id
                    ]
                else:
                    pass
            if trial_id not in linked_trials:
                raise RuntimeError(
                    f"linked trial {trial_id} of row {row['id']} is missing from its job {job_id}"
                )
            else:
                pass
        # note (Haoling Pu): write then rename, so an interrupted run never leaves a half-written file that a restart would skip.
        temporary_path = row_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "row_id": row["id"],
                    "num_linked_trials": len(linked_trial_ids),
                    "trials": list(linked_trials.values()),
                },
                indent=1,
            )
        )
        temporary_path.rename(row_path)
        print(
            f"{row['metadata']['model_display']['label']}: {len(linked_trials)} of {len(linked_trial_ids)} linked trials",
            flush=True,
        )


if __name__ == "__main__":
    main()
