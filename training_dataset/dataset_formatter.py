import json
from typing import List, Dict

def format_for_instruction_tuning(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Formats QA pairs for instruction tuning.
    """
    formatted_data = []
    for qa in qa_pairs:
        formatted_item = {
            "instruction": qa.get("instruction", ""),
            "input": qa.get("input", ""),
            "output": qa.get("output", "")
        }
        formatted_data.append(formatted_item)
    return formatted_data

def format_for_conversational_training(qa_pairs: List[Dict]) -> List[Dict]:
    """
    Formats QA pairs for conversational training.
    """
    formatted_data = []
    for qa in qa_pairs:
        conversation = {
            "messages": [
                {"role": "user", "content": qa.get("instruction", "")},
                {"role": "assistant", "content": qa.get("output", "")}
            ]
        }
        formatted_data.append(conversation)
    return formatted_data

def split_dataset(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Dict]]:
    """
    Splits the dataset into train, validation, and test sets.
    """
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }

def save_formatted_dataset(formatted_data: Dict[str, List[Dict]], output_dir: str):
    """
    Saves the formatted dataset splits to separate files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in formatted_data.items():
        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(split_data)} samples to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_qa_pairs = [
        {"instruction": "What are electric vehicles?", "input": "", "output": "Electric vehicles are cars powered by electricity."},
        {"instruction": "Why are charging stations important?", "input": "", "output": "Charging stations are essential for EV adoption."}
    ]
    
    # Format for instruction tuning
    instruction_formatted = format_for_instruction_tuning(sample_qa_pairs)
    
    # Split dataset
    dataset_splits = split_dataset(instruction_formatted)
    
    # Save formatted dataset
    save_formatted_dataset(dataset_splits, "formatted_dataset")
    
    print("Dataset formatting complete.")

