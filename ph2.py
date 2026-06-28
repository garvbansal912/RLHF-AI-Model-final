from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig

# Load preference data from your Garv folder
dataset = load_dataset("json", data_files="Garv/preference_data.json", split="train")

# Fixed: Shifted to a standard model architecture containing valid text configurations
model_id = "sshleifer/tiny-gpt2"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Force explicitly defined functional text strings across configurations
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

# Configure the reward trainer for your CPU environment
reward_config = RewardConfig(
    output_dir="Garv/reward_model",
    max_steps=3,                   # Fast execution for mock testing
    per_device_train_batch_size=1,
    logging_steps=1,
    use_cpu=True,                  # Keeps training strictly on your CPU
    remove_unused_columns=False    # Prevents Hugging Face from dropping chosen/rejected keys
)

trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,    # Fixed: Explicitly register token variables for internal TRL mapping
    train_dataset=dataset,
    args=reward_config
)

print("Starting Reward Model Phase Mock Test...")
trainer.train()
trainer.save_model("Garv/reward_model")
print("Reward Model Test Complete! Saved to Garv/reward_model\n")
