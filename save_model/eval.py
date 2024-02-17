
import pandas as pd

from nlgeval import NLGEval

nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

def compute_metrics(ref_file, hyp_file):
    df = pd.read_csv(ref_file, header=None)
    ref_list = df[0].tolist()
    df = pd.read_csv(hyp_file, header=None)
    hyp_list = df[0].tolist()

    assert len(ref_list) == len(hyp_list)

    score = nlgeval.compute_metrics(ref_list=[ref_list], hyp_list=hyp_list)
    print(score)

compute_metrics('result/humaneval/gold.csv', 'result/humaneval/codellama.csv')
# compute_metrics('result/humaneval/gold.csv', 'result/humaneval/codet5.csv')
# compute_metrics('result/humaneval/gold.csv', 'result/humaneval/codet5p.csv')
# compute_metrics('result/humaneval/gold.csv', 'result/humaneval/codegeex2.csv')
compute_metrics('result/openeval/gold.csv', 'result/openeval/codellama.csv')