While these previous studies demonstrate the potential of CoTs in improving the performance of LLMs for code generation, they possess certain limitations as the current methods for CoT generation heavily rely on manual writing of CoTs or the utilization of LLMs , leading to high costs. 

These limitations motivate us to investigate the following two main questions. 

**(1) Can ℓLMs independently generate high-quality CoTs to guide code generation?**

**(2) can ℓLMs benefit from generated CoTs? **

Here, “independently” means no model training or model parameter updating.

## How to use COTTON?

```python
from nlp2 import set_seed

from LLAMA_Model import LLAMASeq2Seq

set_seed(42)

model = LLAMASeq2Seq(base_model_path="codellama/CodeLlama-7b-Python-hf", add_eos_token=False, adapter="lora", load_adapter_path="save_model/checkpoint-best-bleu", source_len=256, cutoff_len=512)

prompt = '''def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    """"
'''

cot = model.predict(prompt)
```

