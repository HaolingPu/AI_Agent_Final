"""Sanity check: run our selection methods on tinyBenchmarks' own MMLU data (Open LLM Leaderboard, 395 models x 14,042 questions).

If our code is right, anchor points and gp-IRT should beat random selection here by about as much as the paper reports.
Then a method that does no better than random on agent benchmarks does so because of the data, not because of our code.

Data: data/lb.pickle of github.com/felipemaiapolo/efficbench (commit c9df154), saved as data/raw/efficbench/lb.pickle:

    curl -L -o data/raw/efficbench/lb.pickle https://raw.githubusercontent.com/felipemaiapolo/efficbench/c9df1547a460d5f1d2ad1ad88b6a117979858fbc/data/lb.pickle

Protocol, as in the authors' code: one split, 5 selection seeds, 100 questions.
    random split: every fourth model is tested (99 models), the other 296 select.
    date split: the 98 newest models are tested (the pickle lists models newest first).

Known differences, so a small mismatch with the paper is not by itself a bug:
    The authors' truth is the macro-average over the 57 subjects, and their anchors and p-IRT weigh questions to match it.
    Our methods weigh every question equally and target the plain mean; the error is reported against both truths.
    The authors' random baseline takes 1-2 questions per subject; it is added here as random_by_subject.
    Our IRT is a MAP fit; the authors use py-irt's variational fit.
"""

import argparse
import logging
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from subset_eval.methods import (
    SELECTION_METHODS,
    estimate_solve_rates,
    fit_selection_group,
    select_task_weights,
)
from subset_eval.seeding import purpose_generator

logger = logging.getLogger(__name__)

MMLU_SCENARIO_PREFIX = "harness_hendrycksTest_"
SUBSET_SIZE = 100
NUM_DATE_TEST_MODELS = 98
RANDOM_SPLIT_TEST_STRIDE = 4
# note (Haoling Pu): lb.pickle needs only these two; numpy 2 renamed numpy.core to numpy._core.
ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy.core.numeric", "_frombuffer"),
    ("numpy._core.numeric", "_frombuffer"),
}
# note (Haoling Pu): the paper gives these only as figures. Recomputed from the authors' accs_mmlu_split-*_iterations-5.pickle and results_mmlu_split-*_iterations-5.pickle at 100 questions:
# error = |estimate - macro-average truth| averaged over seeds and test models; Spearman across test models per seed, averaged over seeds.
PAPER_MMLU_100 = {
    ("random", "random_by_subject"): (0.0368, 0.920),
    ("random", "anchor_correctness"): (0.0206, 0.970),
    ("random", "anchor_irt"): (0.0285, 0.941),
    ("random", "random_gp_irt"): (0.0181, 0.973),
    ("random", "anchor_correctness_gp_irt"): (0.0173, 0.975),
    ("random", "anchor_irt_gp_irt"): (0.0176, 0.971),
    ("date", "random_by_subject"): (0.0328, 0.891),
    ("date", "anchor_correctness"): (0.0180, 0.926),
    ("date", "anchor_irt"): (0.0248, 0.901),
    ("date", "random_gp_irt"): (0.0193, 0.909),
    ("date", "anchor_correctness_gp_irt"): (0.0183, 0.897),
    ("date", "anchor_irt_gp_irt"): (0.0194, 0.918),
}


class NumpyArrayUnpickler(pickle.Unpickler):
    """The pickle comes from the internet: rebuild numpy arrays only, never run other code."""

    def find_class(self, module: str, name: str) -> Callable[..., object]:
        if (module, name) in ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        else:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-repeats", type=int, default=5)
    parser.add_argument(
        "--pickle-path", type=Path, default=Path("data/raw/efficbench/lb.pickle")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    cli_arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    with cli_arguments.pickle_path.open("rb") as pickle_file:
        leaderboard = NumpyArrayUnpickler(pickle_file).load()
    # note (Haoling Pu): each scenario stores correctness as (questions, models).
    outcomes = pd.concat(
        [
            pd.DataFrame(
                scenario["correctness"].T,
                index=leaderboard["models"],
                columns=[
                    f"{scenario_name}/{question}"
                    for question in range(scenario["correctness"].shape[0])
                ],
            )
            for scenario_name, scenario in leaderboard["data"].items()
            if scenario_name.startswith(MMLU_SCENARIO_PREFIX)
        ],
        axis=1,
    )
    subject_by_question = pd.Series(
        outcomes.columns.str.split("/").str[0], index=outcomes.columns
    )
    question_ids_by_subject = {
        subject: question_ids.tolist()
        for subject, question_ids in subject_by_question.groupby(
            subject_by_question
        ).groups.items()
    }
    logger.info(
        f"MMLU: {outcomes.shape[0]} models x {outcomes.shape[1]} questions in {len(question_ids_by_subject)} subjects"
    )

    is_random_split_test = np.arange(len(outcomes)) % RANDOM_SPLIT_TEST_STRIDE == 0
    is_date_split_test = np.arange(len(outcomes)) < NUM_DATE_TEST_MODELS
    result_rows = []
    for split_kind, is_test in (
        ("random", is_random_split_test),
        ("date", is_date_split_test),
    ):
        selection_outcomes = outcomes[~is_test]
        test_outcomes = outcomes[is_test]
        plain_truth = test_outcomes.mean(axis=1)
        subject_truth = (
            test_outcomes.T.groupby(subject_by_question).mean().T.mean(axis=1)
        )
        selection_group_fit = fit_selection_group(
            selection_outcomes,
            random_generator=purpose_generator(
                seed=cli_arguments.seed, repeat=0, purpose=f"fit_{split_kind}"
            ),
        )
        logger.info(
            f"{split_kind} split: {len(selection_outcomes)} select, {len(test_outcomes)} tested, "
            f"IRT dimension {selection_group_fit.tiny_benchmarks_fit.irt.discrimination.shape[1]}"
        )
        for repeat in range(cli_arguments.num_repeats):
            estimated_solve_rate_by_method = {}
            for selection_method in SELECTION_METHODS:
                task_weights = select_task_weights(
                    selection_method,
                    selection_group_fit,
                    subset_size=SUBSET_SIZE,
                    random_generator=purpose_generator(
                        seed=cli_arguments.seed,
                        repeat=repeat,
                        purpose=f"select_{split_kind}_{selection_method}",
                    ),
                )
                estimated_solve_rate_by_method |= estimate_solve_rates(
                    selection_method,
                    task_weights,
                    test_outcomes,
                    selection_group_fit,
                    subset_size=SUBSET_SIZE,
                )
            # note (Haoling Pu): the authors' random baseline: shuffle the subjects, give each 100 // 57 questions and the first 100 % 57 one more, then take the plain mean.
            subject_generator = purpose_generator(
                seed=cli_arguments.seed,
                repeat=repeat,
                purpose=f"select_{split_kind}_random_by_subject",
            )
            shuffled_subjects = subject_generator.permutation(
                list(question_ids_by_subject)
            )
            num_questions_by_position = np.full(
                len(shuffled_subjects), SUBSET_SIZE // len(shuffled_subjects)
            )
            num_questions_by_position[: SUBSET_SIZE % len(shuffled_subjects)] += 1
            by_subject_question_ids = [
                question_id
                for subject, num_questions in zip(
                    shuffled_subjects, num_questions_by_position
                )
                for question_id in subject_generator.choice(
                    question_ids_by_subject[subject],
                    size=num_questions,
                    replace=False,
                )
            ]
            estimated_solve_rate_by_method["random_by_subject"] = test_outcomes[
                by_subject_question_ids
            ].mean(axis=1)

            for method, estimated_solve_rate in estimated_solve_rate_by_method.items():
                result_rows.append(
                    {
                        "split": split_kind,
                        "repeat": repeat,
                        "method": method,
                        "error_vs_plain_truth": (estimated_solve_rate - plain_truth)
                        .abs()
                        .mean(),
                        "error_vs_subject_truth": (estimated_solve_rate - subject_truth)
                        .abs()
                        .mean(),
                        "spearman_vs_plain_truth": spearmanr(
                            estimated_solve_rate, plain_truth
                        ).statistic,
                        "spearman_vs_subject_truth": spearmanr(
                            estimated_solve_rate, subject_truth
                        ).statistic,
                    }
                )
            logger.info(f"{split_kind} split: repeat {repeat} done")

    results = pd.DataFrame(result_rows)
    cli_arguments.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        cli_arguments.output_dir / "tiny_benchmarks_reproduction.csv", index=False
    )
    summary = results.groupby(["split", "method"]).mean().drop(columns="repeat")
    paper_rows = [
        PAPER_MMLU_100.get(split_and_method, (np.nan, np.nan))
        for split_and_method in summary.index
    ]
    summary["paper_error"] = [paper_error for paper_error, _ in paper_rows]
    summary["paper_spearman"] = [paper_spearman for _, paper_spearman in paper_rows]
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
