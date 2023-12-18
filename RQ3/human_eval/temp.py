from tqdm import tqdm

from human_eval.data import read_problems

problems = read_problems()
for task_id in tqdm(problems):
    prompt = problems[task_id]["prompt"]
    if(prompt[-4:] != '"""\n'):
        print(prompt)