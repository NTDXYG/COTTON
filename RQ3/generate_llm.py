import re

from nlp2 import set_seed
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from human_eval.data import write_jsonl, read_problems

set_seed(42)

def generate_one(prompt, tokenizer, model, device, max_len, model_type):
    STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']
    BEGIN_SQE = '"""'
    # print(prompt)
    prompt_batch = [prompt]
    prompt_batch_decoder = [prompt]
    encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    encoding_decoder = tokenizer(prompt_batch_decoder, return_tensors="pt", truncation=True,
                                 max_length=max_len).to(device)
    with torch.no_grad():

        if model_type == 'codet5p-2b' or model_type == 'codet5p-6b':
            gen_tokens = model.generate(**encoding,
                                        decoder_input_ids=encoding_decoder['input_ids'],
                                        do_sample=True,
                                        temperature=0.2,
                                        max_new_tokens=max_len,
                                        num_return_sequences=1,
                                        decoder_start_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        pad_token_id=tokenizer.eos_token_id,
                                        top_p=0.95)
        else:
            gen_tokens = model.generate(**encoding,
                                        do_sample=True,
                                        temperature=0.2,
                                        max_new_tokens=max_len,
                                        num_return_sequences=1,
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

def main(model_type, max_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'codet5p-2b' or model_type == 'codet5p-6b':
        model = '/home/yangguang/models/' + model_type
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSeq2SeqLM.from_pretrained(model,
                                                      trust_remote_code=True,  # False for 220m and 770m models
                                                      torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True)

    elif model_type == 'codet5p-220m' or model_type == 'codet5p-770m':
        model = '/home/yangguang/models/' + model_type + '-py'
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSeq2SeqLM.from_pretrained(model,
                                                      trust_remote_code=False,  # False for 220m and 770m models
                                                      torch_dtype=torch.float16,
                                                      low_cpu_mem_usage=True)

    else:
        model = '/home/yangguang/models/' + model_type
        tokenizer = AutoTokenizer.from_pretrained(model,
                                                  trust_remote_code=True if 'codegeex2-6b' in model or 'replit' in model else False)
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
    for task_id in tqdm(problems):
        prompt = problems[task_id]["prompt"]
        if model_type == 'codegeex2-6b':
            prompt = '# language: Python' + '\n' + prompt
        if 'starcoder' in model_type:
            prompt = "<fim_prefix>" + prompt + "<fim_suffix><fim_middle>"
        completion_seqs = generate_one(prompt, tokenizer=tokenizer, model=model, device=device, max_len=max_len, model_type=model_type)
        for completion_seq in completion_seqs:
            samples.append(dict(task_id=task_id, completion=completion_seq))
    write_jsonl('pass_1/'+dataset+'/'+model_type+'_samples.jsonl', samples)

if __name__ == '__main__':
    max_len = 256
    dataset = 'openeval'
    model_type = ['codegen-350M-mono', 'codegen-2B-mono', 'codegen-6B-mono', 'starcoderbase-1b', 'starcoderbase-3b', 'starcoderbase-7b', 'codet5p-220m'
                  , 'codet5p-770m', 'codet5p-2b', 'codet5p-6b']

#'origin_result', 'only_signature', 'signature_stand', 'signature_desc', 'signature_desc_stand', 'cot_signature_desc_gpt3.5', 'cot_signature_desc_gpt3.5_code'
    for model in model_type:
        main(model, max_len)
    # main(model_type, max_len, N, 'cot_signature_desc_codet5p')