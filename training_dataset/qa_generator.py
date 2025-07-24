import json
from typing import List, Dict

# This is a placeholder for LLM API integration.
# In a real scenario, you would use an actual LLM API (e.g., OpenAI, Hugging Face, etc.)
# to generate questions and answers based on the provided text.

def generate_qa_pairs_with_llm(text: str, num_qa_pairs: int = 3) -> List[Dict]:
    """
    Simulates generating QA pairs using an LLM.
    In a real implementation, this would call an LLM API.
    """
    qa_pairs = []
    for i in range(num_qa_pairs):
        # Placeholder for LLM generated question and answer
        question = f"What is a key point from the text related to topic {i+1}?"
        answer = f"The text discusses various aspects of topic {i+1}, such as..."
        qa_pairs.append({"question": question, "answer": answer, "context": text})
    return qa_pairs

def generate_training_dataset(processed_data: List[Dict], output_path: str) -> List[Dict]:
    """
    Generates a training dataset by creating QA pairs from processed text data.
    """
    training_data = []
    for item in processed_data:
        # Assuming 'text_content' is the field containing the main text
        text_to_process = item.get("text_content", item.get("text", ""))
        if text_to_process:
            qa_pairs = generate_qa_pairs_with_llm(text_to_process)
            for qa in qa_pairs:
                training_data.append({
                    "instruction": qa["question"],
                    "input": qa["context"],
                    "output": qa["answer"]
                })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2, ensure_ascii=False)
    print(f"Training dataset saved to {output_path}")
    
    return training_data

if __name__ == "__main__":
    # Example usage with dummy processed data
    sample_processed_data = [
        {"text_content": "Electric vehicles are powered by electricity and are becoming increasingly popular.", "id": 1},
        {"text_content": "Charging stations for EVs are essential infrastructure for widespread adoption.", "id": 2}
    ]
    
    output_file = "sample_training_dataset.json"
    generated_dataset = generate_training_dataset(sample_processed_data, output_file)
    print(f"Generated {len(generated_dataset)} QA pairs.")


