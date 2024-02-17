from nlp2 import set_seed

from LLAMA_Model import LLAMASeq2Seq

set_seed(42)

model = LLAMASeq2Seq(base_model_path="/home/yangguang/models/CodeLlama-7b-Python-hf", add_eos_token=False,
                       adapter="lora", load_adapter_path="valid/checkpoint-best-bleu", source_len=256, cutoff_len=512)

model.test(filename='dataset/humaneval.csv', output_dir='test_humaneval_contrastive/', decoding = 'contrastive')
model.test(filename='dataset/openeval.csv', output_dir='test_openeval_contrastive/', decoding = 'contrastive')