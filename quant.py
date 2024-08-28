import os
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


from huggingface_hub import login

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

login(token="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")

model_path = 'meta-llama/Meta-Llama-3.1-70B'
quant_path = 'Meta-Llama-3.1-70B-awq'
quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }


# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, **{"low_cpu_mem_usage": True}, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)