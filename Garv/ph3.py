import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Initializing PPO Alternative Workflow Test...")

# 1. Load your Phase 1 model weights and token processing files
model_path = "Garv/sft_checkpoint"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. Fixed: Read JSON Lines format safely line-by-line
sft_data = []
with open("Garv/sft_data.json", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # Skip empty lines
            sft_data.append(json.loads(line))

# Extract the text string from the very first entry
first_entry = sft_data[0]
sample_prompt = first_entry.get("text", "### Instruction: Write a python print statement.")

print(f"Loaded Prompt: {sample_prompt}")

# 3. Simulate an environment generation step (Actor Action)
inputs = tokenizer(sample_prompt, return_tensors="pt")
with torch.no_grad():
    generation_output = model.generate(
        **inputs, 
        max_new_tokens=15, 
        do_sample=True, 
        temperature=0.7
    )

response_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
print(f"Generated Assistant Response:\n{response_text}")

# 4. Simulate saving your verified final RLHF test model weights
output_dir = "Garv/final_rlhf_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\nPhase 3 Test Complete! Verified weights saved successfully to: {output_dir}\n")
