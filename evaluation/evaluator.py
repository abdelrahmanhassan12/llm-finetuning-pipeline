from typing import List, Dict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time

def calculate_rouge(predictions: List[str], references: List[str]) -> Dict:
    """
    Calculates ROUGE scores for a list of predictions against references.
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    scores = {"rouge1": {"fmeasure": 0, "precision": 0, "recall": 0},
              "rouge2": {"fmeasure": 0, "precision": 0, "recall": 0},
              "rougeL": {"fmeasure": 0, "precision": 0, "recall": 0}}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key]["fmeasure"] += score[key].fmeasure
            scores[key]["precision"] += score[key].precision
            scores[key]["recall"] += score[key].recall
            
    for key in scores:
        scores[key]["fmeasure"] /= len(predictions)
        scores[key]["precision"] /= len(predictions)
        scores[key]["recall"] /= len(predictions)
        
    return scores

def calculate_bleu(predictions: List[str], references: List[List[str]]) -> float:
    """
    Calculates BLEU score for a list of predictions against references.
    References should be a list of lists of tokens.
    """
    chencherry = SmoothingFunction()
    bleu_scores = []
    for pred, ref in zip(predictions, references):
        # BLEU expects tokenized sentences
        pred_tokens = pred.split()
        ref_tokens = [r.split() for r in ref] # ref can be multiple references
        bleu_scores.append(sentence_bleu(ref_tokens, pred_tokens, smoothing_function=chencherry.method1))
    
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def measure_inference_latency(model_inference_func, *args, num_runs: int = 100, **kwargs) -> Dict:
    """
    Measures the average inference latency and throughput of a model.
    `model_inference_func` should be a callable that takes model inputs and returns outputs.
    """
    latencies = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        _ = model_inference_func(*args, **kwargs)
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)
    
    avg_latency = sum(latencies) / num_runs
    throughput = num_runs / sum(latencies) # items per second
    
    return {"average_latency_ms": avg_latency * 1000, "throughput_items_per_second": throughput}

if __name__ == "__main__":
    # Example Usage
    predictions = [
        "The cat sat on the mat",
        "The dog ran fast"
    ]
    references = [
        "The cat was on the mat",
        "A dog ran quickly"
    ]
    
    # ROUGE
    rouge_scores = calculate_rouge(predictions, references)
    print("ROUGE Scores:", rouge_scores)
    
    # BLEU
    # For BLEU, references should be a list of lists of tokenized sentences
    bleu_references = [[ref] for ref in references] # Each prediction has one reference here
    bleu_score = calculate_bleu(predictions, bleu_references)
    print("BLEU Score:", bleu_score)
    
    # Inference Latency (dummy function)
    def dummy_model_inference(input_text):
        time.sleep(0.01) # Simulate some processing time
        return f"Processed: {input_text}"
    
    latency_metrics = measure_inference_latency(dummy_model_inference, "sample input")
    print("Inference Metrics:", latency_metrics)


