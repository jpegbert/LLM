from peft import AutoPeftModelForCausalLM


path_to_adapter = "/root/model/qwen2_fine_tuned/checkpoint-1000/" # lora trained adapter path
new_model_path = "/root/model/qwen2_fine_tuned_merged/"

model = AutoPeftModelForCausalLM(
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()

merged_model = model.merge_and_unload()
# max_shard_size and safe_serialization are not necessary
# They respectively work for sharding checkpoint and save the model to safetensors
merged_model.save_pretrained(new_model_path, max_shard_size="2048MB", safe_serialization=True)


# 分词器保存
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(path_to_adapter, trust_remote_code=True)
tokenizer.save_pretrained(new_model_path)