import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Load your mock data from the Garv folder
dataset = load_dataset("json", data_files="Garv/sft_data.json", split="train")

# Use an ultra-tiny model for fast local mock testing
model_id = "meta-llama/Llama-3.2-1B-Instruct"  # Lightweight but incredibly smart
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Configure to run explicitly on CPU to bypass GPU requirements
training_args = SFTConfig(
    output_dir="Garv/sft_checkpoint",
    max_steps=3,  
    per_device_train_batch_size=1,
    logging_steps=1,
    dataset_text_field="text",
    max_length=128,
    use_cpu=True                  # Fixed: Added to force training on CPU
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

print("Starting SFT Phase Mock Test...")
trainer.train()
trainer.save_model("Garv/sft_checkpoint")
print("SFT Test Complete! Checkpoint saved to Garv/sft_checkpoint\n")
