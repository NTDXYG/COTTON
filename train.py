from nlp2 import set_seed

from LLAMA_Model import LLAMASeq2Seq

set_seed(42)


model = LLAMASeq2Seq(base_model_path="/home/yangguang/models/CodeLlama-7b-Python-hf", add_eos_token=False,
                       adapter="lora", load_adapter_path="None", source_len=256, cutoff_len=512)

model.train(train_filename="dataset/train.csv", train_batch_size=1, learning_rate=1e-4, num_train_epochs=20, early_stop=3,
              do_eval=True, eval_filename="dataset/valid.csv", eval_batch_size=1, output_dir='save_model/', do_eval_bleu=True)

model = LLAMASeq2Seq(base_model_path="/home/yangguang/models/CodeLlama-7b-Python-hf", add_eos_token=False,
                       adapter="lora", load_adapter_path="save_model/checkpoint-best-bleu", source_len=256, cutoff_len=512)

model.test(filename='dataset/humaneval.csv', output_dir='test_humaneval/')
model.test(filename='dataset/openeval.csv', output_dir='test_openeval/')