import fire
import sys

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness


def entry_point(
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results)


if __name__ == '__main__':
    # entry_point('../codet5p-2b_samples.jsonl', k='1', problem_file='/home/yangguang/PycharmProjects/RADAR-zero-shot/human_eval/HumanEval.jsonl')
    entry_point('../graphcodebert/humaneval/codegen-350M-mono_samples.jsonl', k='1',
                problem_file='../HumanEvalPlus-v0.1.7.jsonl')

