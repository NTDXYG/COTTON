import os
import re

from nlp2 import set_seed
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, \
    CodeLlamaTokenizer

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
    # prompt_batch_decoder = [prompt]
    encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=512).to(device)
    # encoding_decoder = tokenizer(prompt_batch_decoder, return_tensors="pt", truncation=True,
    #                              max_length=512).to(device)
    with torch.no_grad():

        gen_tokens = model.generate(**encoding,
                                    do_sample=True,
                                    temperature=0.2,
                                    max_new_tokens=max_len,
                                    num_return_sequences=return_sequences,
                                    eos_token_id=tokenizer.eos_token_id,
                                    top_p=0.95)

    gen_tokens = gen_tokens[:, encoding['input_ids'].shape[-1]:]
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
    model = LlamaForCausalLM.from_pretrained(
        "/home/yangguang/models/CodeLlama-7b-Python-hf",
        # quantization_config=q_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = CodeLlamaTokenizer.from_pretrained(
        "/home/yangguang/models/CodeLlama-7b-Python-hf",
        trust_remote_code=True
    )
    model.eval()
    model.to(device)

    # for larger LLMs such as 2B, 6B, and 16B, we need to pass the text prompt to the decoder
    # prompt_to_decoder = True if any([size in model for size in ['2b', '6b', '16b']]) else False

    df = pd.read_csv("../humaneval.csv")
    srcs = df['src'].tolist()
    tgts = df['tgt'].tolist()
    hyps = []
    for src in tqdm(srcs):
        src = prompt_utils(src)

        completion_seqs = generate_one(src, tokenizer=tokenizer, model=model, device=device, max_len=max_len, return_sequences=1, model_type=model_type)
        for completion_seq in completion_seqs:
            # print(completion_seq)
            hyps.append(completion_seq)
    compute_metrics(tgts, hyps)
    df = pd.DataFrame(hyps)
    df.to_csv('humaneval/'+model_type+".csv", index=False, header=None)

from nlgeval import NLGEval

nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)

def compute_metrics(ref_list, hyp_list):

    assert len(ref_list) == len(hyp_list)

    score = nlgeval.compute_metrics(ref_list=[ref_list], hyp_list=hyp_list)
    print(score)

if __name__ == '__main__':
    max_len = 256
    main('codellama', max_len)