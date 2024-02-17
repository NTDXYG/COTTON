from time import sleep
import openai
from tqdm import tqdm

openai.api_base = 'https://api.openai-proxy.org/v1'
openai.api_key = 'sk-k9ZTt7Lchrgu1pFv4iZGt5XAbKmh8jB7V85rCweHBzLc2FkI'

def prompt_gen(code):
    demo = '''# Instruct: Suppose you are a professional computer programming expert.
Please understand the given signature, then write the corresponding implementation python code.
Only returns the remaining code.

# Input:
'''
    return demo + code + '\n\n# Result:\n'

def generate_by_openai(code, model):
    try:
        new_prompt = prompt_gen(code)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": new_prompt},
                {"role": "user", "content": code},
            ],
            temperature=0
        )
        code = response['choices'][0]['message']['content']
        if code.startswith("def"):
            code = '\n'.join(code.split('\n')[1:])
        elif code[0]!= ' ':
            code = '    ' + '\n    '.join(code.split('\n'))
    except:
        code = 'pass'
    return code

def generate_scot(code, model):
    content = '''### Please understand the requirement and write a rough solving process. It starts with a input-output structure. You should use three basic structures to build the solving process, including sequences, branches, and loops. The necessary details should be written in natural languages.
### Example:
#### Input:
def first_Repeated_Char(str):
    """
    Write a python function to find the first repeated 
    character in a given string.
    """

#### Output:
Input: str: a string
Output: ch: a repeated character in str
1: for each character ch in str:
2:     if ch appears more than once in str:
3:         return ch
4: return None

### Input:

    '''
    src = content + code + '\n\n#### Output:'
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": src},
                {"role": "user", "content": code},
            ],
            temperature=0
        )
        code = response['choices'][0]['message']['content']
    except:
        code = 'pass'
    return code

def generate_self_planning(code, model):
    content = '''### Example:
#### Input:
def encrypt(s):
    """
    Create a function encrypt that takes a string as an argument and returns a string encrypted with the alphabet being rotated. 
    The alphabet should be rotated in a manner such that the letters shift down by two multiplied to two places.
    """

#### Output:
Let's think step by step.
1. Create a alphabet, bias two places multiplied by two.
2. Loop the input, find the latter bias letter in alphabet.
3. Return result.

### Input:
    '''
    src = content + code + '\n\n#### Output:'
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": src},
                {"role": "user", "content": code},
            ],
            temperature=0
        )
        code = response['choices'][0]['message']['content']
    except:
        code = 'pass'
    return code

if __name__ == '__main__':

    import pandas as pd
    df = pd.read_csv("../humaneval.csv")
# df = pd.read_csv("../humaneval.csv")

    srcs = df['src'].tolist()
    tgts = df['tgt'].tolist()
    scots = []
    self_plannings = []
    for src in tqdm(srcs):
        scot = generate_scot(src, model='gpt-3.5-turbo')
        self_planning = generate_self_planning(src, model='gpt-3.5-turbo')
        scots.append(scot)
        self_plannings.append(self_planning)
    df = pd.DataFrame(scots)
    df.to_csv('SCoT_humaneval/GPT3_5.csv', index=False, header=None)
    df = pd.DataFrame(self_plannings)
    df.to_csv('SelfPlanning_humaneval/GPT3_5.csv', index=False, header=None)

    df = pd.read_csv("../openeval.csv")
# df = pd.read_csv("../humaneval.csv")

    srcs = df['src'].tolist()
    tgts = df['tgt'].tolist()
    scots = []
    self_plannings = []
    for src in tqdm(srcs):
        scot = generate_scot(src, model='gpt-3.5-turbo')
        self_planning = generate_self_planning(src, model='gpt-3.5-turbo')
        scots.append(scot)
        self_plannings.append(self_planning)
    df = pd.DataFrame(scots)
    df.to_csv('SCoT_openeval/GPT3_5.csv', index=False, header=None)
    df = pd.DataFrame(self_plannings)
    df.to_csv('SelfPlanning_openeval/GPT3_5.csv', index=False, header=None)
