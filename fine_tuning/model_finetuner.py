import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import json
import os

def load_dataset_from_json(file_path: str) -> Dataset:
    """
    Loads a dataset from a JSON file.
    Assumes the JSON file contains a list of dictionaries, where each dictionary
    represents a training example (e.g., with 'instruction', 'input', 'output' keys).
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def fine_tune_model(
    model_name: str,
    train_dataset_path: str,
    output_dir: str,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-4,
    fp16: bool = True,
    logging_steps: int = 10,
    save_steps: int = 100,
    eval_steps: int = 100,
    push_to_hub: bool = False,
    hub_model_id: str = None
):
    """
    Fine-tunes a small language model using LoRA.
    """
    # 1. Load Model and Tokenizer
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add a pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,  # Enable 4-bit quantization for QLoRA
        torch_dtype=torch.bfloat16, # Use bfloat16 for mixed-precision training
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer)) # Resize embeddings if pad token was added

    # 2. Prepare model for k-bit training (QLoRA)
    model = prepare_model_for_kbit_training(model)

    # 3. Configure LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Load and Prepare Dataset
    print(f"Loading training dataset from {train_dataset_path}...")
    train_dataset = load_dataset_from_json(train_dataset_path)

    # Tokenize the dataset
    def tokenize_function(examples):
        # Concatenate instruction, input, and output for training
        full_text = [
            f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Output:\n{output_text}"
            for instruction, input_text, output_text in zip(examples["instruction"], examples["input"], examples["output"])
        ]
        return tokenizer(full_text, truncation=True, max_length=512)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps, # Use eval_steps for evaluation frequency
        evaluation_strategy="steps", # Evaluate every eval_steps
        load_best_model_at_end=True, # Load the best model at the end of training
        metric_for_best_model="eval_loss", # Metric to use for early stopping
        greater_is_better=False, # For loss, smaller is better
        report_to="none", # Disable experiment tracking for now, will integrate later
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        tokenizer=tokenizer,
    )

    # 7. Train Model
    print("Starting training...")
    trainer.train()
    print("Training complete.")

    # 8. Save the fine-tuned model
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Fine-tuned model saved to {final_model_path}")

    # Optional: Push to Hugging Face Hub
    if push_to_hub:
        print("Pushing model to Hugging Face Hub...")
        trainer.push_to_hub()
        print("Model pushed to Hugging Face Hub.")

if __name__ == "__main__":
    # Example usage (dummy data for testing)
    # Create a dummy dataset file for testing
    dummy_data = [
        {"instruction": "Explain AI", "input": "", "output": "AI stands for Artificial Intelligence."},
        {"instruction": "What is ML?", "input": "", "output": "ML stands for Machine Learning."},
        {"instruction": "Define LLM", "input": "", "output": "LLM stands for Large Language Model."},
    ]
    dummy_dataset_path = "dummy_train_dataset.json"
    with open(dummy_dataset_path, "w") as f:
        json.dump(dummy_data, f)

    # Ensure output directory exists
    os.makedirs("finetuned_model_output", exist_ok=True)

    try:
        fine_tune_model(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # A very small model for quick testing
            train_dataset_path=dummy_dataset_path,
            output_dir="finetuned_model_output",
            num_train_epochs=1, # Small number of epochs for quick test
            per_device_train_batch_size=1, # Small batch size for quick test
            logging_steps=1,
            save_steps=1,
            eval_steps=1,
        )
    except Exception as e:
        print(f"An error occurred during fine-tuning: {e}")
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_dataset_path):
            os.remove(dummy_dataset_path)
        # Note: finetuned_model_output directory will remain for inspection



