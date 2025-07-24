import json
from typing import List, Dict

def generate_domain_specific_benchmark(domain: str = "electric vehicle charging stations") -> List[Dict]:
    """
    Generates a domain-specific benchmark dataset.
    This is a placeholder implementation. In a real scenario, you would use
    domain expertise or automated methods to create comprehensive benchmarks.
    """
    
    if domain.lower() == "electric vehicle charging stations":
        benchmark_questions = [
            {
                "question": "What are the main types of EV charging connectors?",
                "expected_answer": "The main types of EV charging connectors include Type 1 (J1772), Type 2 (Mennekes), CHAdeMO, and CCS (Combined Charging System).",
                "category": "technical_knowledge"
            },
            {
                "question": "How long does it typically take to charge an electric vehicle?",
                "expected_answer": "Charging time varies by charger type: Level 1 (8-12 hours), Level 2 (4-8 hours), and DC fast charging (30 minutes to 1 hour for 80% charge).",
                "category": "charging_time"
            },
            {
                "question": "What is the difference between AC and DC charging?",
                "expected_answer": "AC charging uses alternating current and is slower, typically for home and workplace charging. DC charging uses direct current and is faster, used for public fast charging stations.",
                "category": "technical_knowledge"
            },
            {
                "question": "What factors affect EV charging speed?",
                "expected_answer": "Charging speed is affected by battery capacity, current charge level, charger power output, battery temperature, and vehicle's onboard charger capacity.",
                "category": "charging_factors"
            },
            {
                "question": "How do you find EV charging stations?",
                "expected_answer": "EV charging stations can be found using mobile apps like PlugShare, ChargePoint, or built-in vehicle navigation systems that show nearby charging locations.",
                "category": "practical_usage"
            },
            {
                "question": "What is the cost of charging an electric vehicle?",
                "expected_answer": "Charging costs vary by location and electricity rates, typically ranging from $0.10 to $0.30 per kWh, making it generally cheaper than gasoline.",
                "category": "cost_analysis"
            },
            {
                "question": "What are the benefits of installing a home EV charger?",
                "expected_answer": "Home EV chargers provide convenience, faster charging than standard outlets, potential cost savings, and the ability to charge overnight during off-peak hours.",
                "category": "home_charging"
            },
            {
                "question": "What is smart charging for electric vehicles?",
                "expected_answer": "Smart charging allows EV charging to be controlled remotely, optimized for grid demand, scheduled for off-peak hours, and integrated with renewable energy sources.",
                "category": "smart_technology"
            },
            {
                "question": "How does weather affect EV charging?",
                "expected_answer": "Cold weather can reduce charging efficiency and battery performance, while extreme heat can also affect charging speed. Battery thermal management systems help mitigate these effects.",
                "category": "environmental_factors"
            },
            {
                "question": "What is the future of EV charging infrastructure?",
                "expected_answer": "The future includes ultra-fast charging, wireless charging, vehicle-to-grid technology, increased charging station density, and integration with renewable energy sources.",
                "category": "future_trends"
            }
        ]
    else:
        # Generic benchmark for other domains
        benchmark_questions = [
            {
                "question": f"What are the key concepts in {domain}?",
                "expected_answer": f"The key concepts in {domain} include various fundamental principles and practices specific to this field.",
                "category": "general_knowledge"
            },
            {
                "question": f"How is {domain} applied in practice?",
                "expected_answer": f"{domain} is applied through various methods and techniques depending on the specific use case and requirements.",
                "category": "practical_application"
            }
        ]
    
    return benchmark_questions

def save_benchmark_dataset(benchmark_data: List[Dict], output_path: str):
    """
    Saves the benchmark dataset to a JSON file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
    print(f"Benchmark dataset saved to {output_path}")

def evaluate_model_on_benchmark(model_inference_func, benchmark_data: List[Dict]) -> List[Dict]:
    """
    Evaluates a model on the benchmark dataset.
    `model_inference_func` should be a callable that takes a question and returns an answer.
    """
    results = []
    
    for item in benchmark_data:
        question = item["question"]
        expected_answer = item["expected_answer"]
        category = item["category"]
        
        # Get model's answer
        try:
            model_answer = model_inference_func(question)
        except Exception as e:
            model_answer = f"Error: {str(e)}"
        
        result = {
            "question": question,
            "expected_answer": expected_answer,
            "model_answer": model_answer,
            "category": category
        }
        
        results.append(result)
    
    return results

if __name__ == "__main__":
    # Generate benchmark for EV charging stations
    benchmark = generate_domain_specific_benchmark("electric vehicle charging stations")
    
    # Save benchmark
    save_benchmark_dataset(benchmark, "ev_charging_benchmark.json")
    
    print(f"Generated benchmark with {len(benchmark)} questions")
    
    # Example of evaluating a dummy model
    def dummy_model(question):
        return f"This is a dummy answer to: {question}"
    
    evaluation_results = evaluate_model_on_benchmark(dummy_model, benchmark)
    
    print("\nSample evaluation results:")
    for i, result in enumerate(evaluation_results[:2]):  # Show first 2 results
        print(f"\nQuestion {i+1}: {result['question']}")
        print(f"Expected: {result['expected_answer']}")
        print(f"Model: {result['model_answer']}")
        print(f"Category: {result['category']}")

