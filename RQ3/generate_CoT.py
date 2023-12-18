import numpy as np
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from nlp2 import set_seed
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import estimate_pass_at_k
from human_eval.execution import check_correctness

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

set_seed(42)

def evaluate(
    completion_seq: str,
    task_id: str,
    n_workers: int = 4,
    timeout: float = 3.0
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    if dataset == 'humaneval':
        problems = read_problems('HumanEval.jsonl')
    if dataset == 'openeval':
        problems = read_problems('OpenEval.jsonl')
    if dataset == 'humaneval-plus':
        problems = read_problems('HumanEvalPlus-v0.1.7.jsonl')
    problem = problems[task_id]

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        completion = completion_seq
        # entry_point = sample["entry_point"]
        args = (problem, completion, timeout, completion_id[task_id])
        future = executor.submit(check_correctness, *args)
        futures.append(future)
        completion_id[task_id] += 1
        n_samples += 1

        assert len(completion_id) == 1, "Some problems are not attempted."

        # print("Running test suites...")
        for future in as_completed(futures):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        result.sort()
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)

    ks = [1]
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}
    if pass_at_k['pass@1'] == 0.0:
        return False


def generate_one(prompt, tokenizer, model, device, max_len, return_sequences, model_type):
    if 'starcoder' in model_type:
        prompt = "<fim_prefix>" + prompt + "<fim_suffix><fim_middle>"
    STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']
    BEGIN_SQE = '"""'
    # print(prompt)
    prompt_batch = [prompt]
    prompt_batch_decoder = [prompt]
    encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=512).to(device)
    encoding_decoder = tokenizer(prompt_batch_decoder, return_tensors="pt", truncation=True,
                                 max_length=512).to(device)
    with torch.no_grad():

        if model_type == 'codet5p-2b' or model_type == 'codet5p-6b':
            gen_tokens = model.generate(**encoding,
                                        decoder_input_ids=encoding_decoder['input_ids'],
                                        do_sample=True,
                                        temperature=0.2,
                                        max_new_tokens=max_len,
                                        num_return_sequences=return_sequences,
                                        decoder_start_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id,
                                        top_p=0.95)
        else:
            gen_tokens = model.generate(**encoding,
                                        do_sample=True,
                                        temperature=0.2,
                                        max_new_tokens=max_len,
                                        num_return_sequences=return_sequences,
                                        eos_token_id=tokenizer.eos_token_id,
                                        top_p=0.95)

    if model_type == 'codet5p-220m' or model_type == 'codet5p-770m':
        gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
    else:
        gen_tokens = gen_tokens[:, encoding_decoder['input_ids'].shape[-1]:]
        gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    completion_seqs = []
    for gen_seq in gen_seqs:
        completion_seq = gen_seq
        for stop_seq in STOP_SEQS:
            index = completion_seq.find(stop_seq)
            if index != -1:
                completion_seq = completion_seq[:index]
        if model_type != 'codegeex2-6b':
            begin_index = completion_seq.find(BEGIN_SQE)
            if begin_index != -1:
                completion_seq = completion_seq[begin_index+len(BEGIN_SQE):]
        completion_seq = completion_seq.replace('\t', '    ')
        # print(completion_seq)
        completion_seqs.append(completion_seq)
    return completion_seqs


def generate_prompt(prompt, model, index):
    if method == 'self-cot':
        if dataset == 'humaneval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ1/humaneval/'+model+'.csv'
        elif dataset == 'openeval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ1/openeval/'+model+'.csv'
    if method == 'codellama':
        if dataset == 'humaneval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ1/humaneval/codellama.csv'
        elif dataset == 'openeval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ1/openeval/codellama.csv'
    if method == 'chatglm':
        if dataset == 'humaneval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ1/humaneval/chatglm.csv'
        elif dataset == 'openeval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ1/openeval/chatglm.csv'
    if method == 'codecot':
        if dataset == 'humaneval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/humaneval/codecot.csv'
        elif dataset == 'openeval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/openeval/codecot.csv'
    if method == 'codegeex2':
        if dataset == 'humaneval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/humaneval/codegeex2.csv'
        elif dataset == 'openeval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/openeval/codegeex2.csv'
    if method == 'natgen':
        if dataset == 'humaneval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/humaneval/natgen.csv'
        elif dataset == 'openeval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/openeval/natgen.csv'
    if method == 'codegpt-adapter':
        if dataset == 'humaneval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/humaneval/codegpt-adapter.csv'
        elif dataset == 'openeval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/openeval/codegpt-adapter.csv'
    if method == 'graphcodebert':
        if dataset == 'humaneval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/humaneval/graphcodebert.csv'
        elif dataset == 'openeval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/RQ2-generate-CoT/openeval/graphcodebert.csv'
    if method == 'teacher':
        if dataset == 'humaneval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/humaneval.csv'
        elif dataset == 'openeval':
            file_path = '/home/yangguang/PycharmProjects/CoT-dataset/openeval.csv'
    if method == 'teacher':
        df = pd.read_csv(file_path)
        cots = df['tgt'].tolist()
    else:
        df = pd.read_csv(file_path, header=None)
        cots = df[0].tolist()
    cot = str(cots[index])
    cot = cot.replace("\n", "\n    ")
    new = prompt[:-4] + '\n    ' + cot + '\n    ' + prompt[-4:]
    # print(new)
    return new

def main(model_type, max_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if model_type == 'codet5p-2b' or model_type == 'codet5p-6b':
        model = '/home/yangguang/models/'+model_type
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSeq2SeqLM.from_pretrained(model,
                                                      trust_remote_code=True,  # False for 220m and 770m models
                                                      torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True)

    elif model_type == 'codet5p-220m' or model_type == 'codet5p-770m':
        model = '/home/yangguang/models/'+model_type+'-py'
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSeq2SeqLM.from_pretrained(model,
                                                      trust_remote_code=False,  # False for 220m and 770m models
                                                      torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True)

    else:
        model = '/home/yangguang/models/' + model_type
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True if 'codegeex2-6b' in model or 'replit' in model else False)
        model = AutoModelForCausalLM.from_pretrained(model,
                                                      trust_remote_code=True,  # False for 220m and 770m models
                                                      torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True)

    model.eval()
    model.to(device)

    # for larger LLMs such as 2B, 6B, and 16B, we need to pass the text prompt to the decoder
    # prompt_to_decoder = True if any([size in model for size in ['2b', '6b', '16b']]) else False

    # problems = read_problems()
    if dataset == 'humaneval':
        problems = read_problems('HumanEval.jsonl')
    if dataset == 'openeval':
        problems = read_problems('OpenEval.jsonl')
    if dataset == 'humaneval-plus':
        problems = read_problems('HumanEvalPlus-v0.1.7.jsonl')
    samples = []
    index = 0
    for task_id in tqdm(problems):
        prompt = problems[task_id]["prompt"]
        # prompt = generate_prompt(prompt, index)
        # completion_seqs = generate_one(prompt, tokenizer=tokenizer, model=model, device=device, max_len=max_len,
        #                                return_sequences=N, model_type=model_type)
        # completion_seq = completion_seqs[0]

        completion_seqs = generate_one(prompt, tokenizer=tokenizer, model=model, device=device, max_len=max_len,
                                       return_sequences=1, model_type=model_type)
        completion_seq = completion_seqs[0]
        if (evaluate(completion_seq, problems[task_id]['task_id']) == False):
            prompt = generate_prompt(prompt, model_type, index)
            completion_seqs = generate_one(prompt, tokenizer=tokenizer, model=model, device=device, max_len=max_len,
                                           return_sequences=1, model_type=model_type)
            completion_seq = completion_seqs[0]
            # print(completion_seq)
        samples.append(dict(task_id=task_id, completion=completion_seq))
        index += 1
    write_jsonl(method+'/'+dataset+'/'+model_type+'_samples.jsonl', samples)

if __name__ == '__main__':
    max_len = 256
    # dataset = 'humaneval'
    # method = 'teacher'
    model_type = ['starcoderbase-1b', 'starcoderbase-3b', 'starcoderbase-7b'
                  , 'codegen-350M-mono', 'codegen-2B-mono', 'codegen-6B-mono', 'codet5p-220m', 'codet5p-770m', 'codet5p-2b', 'codet5p-6b']

    for dataset in ['humaneval', 'openeval']:
        for method in ['graphcodebert']:
            for model in model_type:
                print(dataset+'-----'+method+'-----'+model)
                main(model, max_len)

    for dataset in ['openeval']:
        for method in ['codegpt-adapter']:
            for model in model_type:
                print(dataset + '-----' + method + '-----' + model)
                main(model, max_len)

                # main(model_type, max_len, N, 'cot_signature_desc_codet5p')