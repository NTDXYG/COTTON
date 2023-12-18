import os
import re

from nlp2 import set_seed
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

import pandas as pd
set_seed(42)

def prompt_utils(src):
    content = '''
### Given a piece of code, output the corresponding implementation idea.
### Example:
#### Input:
from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account falls below zero, and
    at that point function should return True. Otherwise it should return False.
    """

#### Output:
How to solve:
Step 1. Initialize account balance as 0.
Step 2. Iterate through operations.
    -add value to account balance.
    -If account balance < 0, return True.
Step 3. Return False.

### Input:

'''
    src = content+src+'\n\n#### Output:'
    return src

def generate_one(prompt, tokenizer, model, device, max_len, return_sequences, model_type):
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
        if '### Input:' in gen_seq:
            gen_seq = gen_seq[:gen_seq.find('### Input:')]
        gen_seq = gen_seq.strip()
        completion_seqs.append(gen_seq)
    return completion_seqs

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

    df = pd.read_csv("../openeval.csv")
    srcs = df['src'].tolist()
    tgts = df['tgt'].tolist()
    hyps = []
    for src in tqdm(srcs):
        src = prompt_utils(src)
        # print(src)
        if 'starcoder' in model_type:
            src = "<fim_prefix>" + src + "<fim_suffix><fim_middle>"
        completion_seqs = generate_one(src, tokenizer=tokenizer, model=model, device=device, max_len=max_len, return_sequences=1, model_type=model_type)
        for completion_seq in completion_seqs:
            # print(completion_seq)
            hyps.append(completion_seq)
    compute_metrics(tgts, hyps)
    df = pd.DataFrame(hyps)
    df.to_csv('openeval/'+model_type+".csv", index=False, header=None)

from nlgeval import NLGEval

nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

def compute_metrics(ref_list, hyp_list):
    # df = pd.read_csv(ref_file, header=None)
    # ref_list = df[0].tolist()
    # df = pd.read_csv(hyp_file, header=None)
    # hyp_list = df[0].tolist()

    assert len(ref_list) == len(hyp_list)

    score = nlgeval.compute_metrics(ref_list=[ref_list], hyp_list=hyp_list)
    print(score)

if __name__ == '__main__':
    max_len = 256
    main('codet5p-220m', max_len)