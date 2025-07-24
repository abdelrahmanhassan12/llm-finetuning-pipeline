import re
import json
from typing import List, Dict

def clean_text(text: str) -> str:
    """
    Cleans text by removing extra whitespace, special characters, etc.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading/trailing whitespace
    return text

def deduplicate_data(data: List[Dict], key: str = 'text') -> List[Dict]:
    """
    Deduplicates a list of dictionaries based on a specified key.
    """
    seen = set()
    deduplicated = []
    for item in data:
        if item.get(key) not in seen:
            deduplicated.append(item)
            seen.add(item.get(key))
    return deduplicated

def quality_filter(data: List[Dict], min_length: int = 50) -> List[Dict]:
    """
    Filters out data entries based on quality criteria (e.g., minimum text length).
    """
    filtered_data = [item for item in data if len(item.get('text', '')) >= min_length]
    return filtered_data

def normalize_text(text: str) -> str:
    """
    Normalizes text to lowercase.
    """
    return text.lower()

def tokenize_text(text: str) -> List[str]:
    """
    Simple tokenization by splitting on whitespace.
    For more advanced tokenization, a dedicated NLP library would be used.
    """\n    return text.split()

def process_raw_data(raw_data: List[Dict]) -> List[Dict]:
    """
    Applies cleaning, deduplication, quality filtering, and normalization to raw data.
    """
    processed_data = []
    for item in raw_data:
        cleaned_text = clean_text(item.get('text', ''))
        normalized_text = normalize_text(cleaned_text)
        processed_data.append({
            **item,  # Keep all original fields
            'text': normalized_text,
            'tokens': tokenize_text(normalized_text)
        })
    
    # Apply deduplication and quality filtering after initial processing
    processed_data = deduplicate_data(processed_data)
    processed_data = quality_filter(processed_data)
    
    return processed_data

def save_processed_data(data: List[Dict], output_path: str):
    """
    Saves processed data to a JSON file.
    """\n    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    sample_raw_data = [
        {"text": "  This is a sample text.  ", "id": 1},
        {"text": "This is a sample text.", "id": 2},
        {"text": "Another short text.", "id": 3},
        {"text": "This is a longer and unique piece of information that should be kept.", "id": 4}
    ]

    processed_data = process_raw_data(sample_raw_data)
    print("\nProcessed Data:")
    for item in processed_data:
        print(item)

    # Example of saving data
    # save_processed_data(processed_data, "processed_data.json")


