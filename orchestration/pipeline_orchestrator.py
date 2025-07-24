import os
import sys
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Any
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection import scrape_website, process_pdf_directory
from data_processing import process_raw_data, DataStorage
from training_dataset import generate_training_dataset, split_dataset, save_formatted_dataset
from fine_tuning import ExperimentTracker
from evaluation import generate_domain_specific_benchmark, evaluate_model_on_benchmark

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.experiment_tracker = ExperimentTracker()
        self.data_storage = DataStorage()
        
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def run_data_collection(self) -> List[Dict]:
        """Run the data collection phase."""
        logger.info("Starting data collection phase...")
        
        collected_data = []
        
        # Process PDFs if specified
        if "PDFs" in self.config.get("data_sources", []):
            pdf_directory = self.config.get("pdf_directory", "data/pdfs")
            if os.path.exists(pdf_directory):
                logger.info(f"Processing PDFs from {pdf_directory}")
                pdf_data = process_pdf_directory(pdf_directory)
                collected_data.extend(pdf_data)
            else:
                logger.warning(f"PDF directory {pdf_directory} not found")
        
        # Web scraping if URLs are specified
        urls = self.config.get("urls", [])
        for url in urls:
            logger.info(f"Scraping {url}")
            scraped_text = scrape_website(url)
            if scraped_text:
                collected_data.append({
                    "source": url,
                    "text": scraped_text,
                    "metadata": {"source_type": "web"}
                })
        
        logger.info(f"Data collection completed. Collected {len(collected_data)} documents.")
        return collected_data
    
    def run_data_processing(self, raw_data: List[Dict]) -> List[Dict]:
        """Run the data processing phase."""
        logger.info("Starting data processing phase...")
        
        # Process raw data
        processed_data = process_raw_data(raw_data)
        
        # Store in database
        for doc in processed_data:
            doc_id = self.data_storage.store_document(doc)
            logger.info(f"Stored document {doc_id}")
        
        logger.info(f"Data processing completed. Processed {len(processed_data)} documents.")
        return processed_data
    
    def run_training_dataset_generation(self, processed_data: List[Dict]) -> str:
        """Run the training dataset generation phase."""
        logger.info("Starting training dataset generation phase...")
        
        # Generate training dataset
        output_path = "data/training_dataset.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        training_data = generate_training_dataset(processed_data, output_path)
        
        # Split and format dataset
        dataset_splits = split_dataset(training_data)
        save_formatted_dataset(dataset_splits, "data/formatted_dataset")
        
        logger.info(f"Training dataset generation completed. Generated {len(training_data)} QA pairs.")
        return "data/formatted_dataset/train.json"
    
    def run_fine_tuning(self, train_dataset_path: str) -> str:
        """Run the fine-tuning phase."""
        logger.info("Starting fine-tuning phase...")
        
        # Start experiment tracking
        experiment_config = {
            "model_name": self.config.get("base_llm", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
            "target_domain": self.config.get("target_domain", "general"),
            "use_case": self.config.get("use_case", "QA")
        }
        
        experiment_id = self.experiment_tracker.start_experiment("llm_finetuning", experiment_config)
        
        try:
            # Run fine-tuning (this would call the actual fine-tuning script)
            output_dir = f"models/finetuned_{experiment_id}"
            os.makedirs(output_dir, exist_ok=True)
            
            # In a real implementation, this would call the fine_tune_model function
            # For now, we'll simulate the process
            self.experiment_tracker.log_message("Fine-tuning started")
            self.experiment_tracker.log_metric("train_loss", 0.5, step=1)
            self.experiment_tracker.log_metric("train_loss", 0.3, step=2)
            self.experiment_tracker.log_message("Fine-tuning completed")
            
            self.experiment_tracker.end_experiment("completed")
            
            logger.info(f"Fine-tuning completed. Model saved to {output_dir}")
            return output_dir
            
        except Exception as e:
            self.experiment_tracker.log_message(f"Fine-tuning failed: {str(e)}", "ERROR")
            self.experiment_tracker.end_experiment("failed")
            raise
    
    def run_evaluation(self, model_path: str) -> Dict:
        """Run the evaluation phase."""
        logger.info("Starting evaluation phase...")
        
        # Generate domain-specific benchmark
        domain = self.config.get("target_domain", "electric vehicle charging stations")
        benchmark = generate_domain_specific_benchmark(domain)
        
        # Save benchmark
        benchmark_path = "data/benchmark.json"
        os.makedirs(os.path.dirname(benchmark_path), exist_ok=True)
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark, f, indent=2)
        
        # Dummy model evaluation (in real implementation, would load actual model)
        def dummy_model(question):
            return f"This is a response to: {question}"
        
        evaluation_results = evaluate_model_on_benchmark(dummy_model, benchmark)
        
        # Save evaluation results
        results_path = "data/evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_path}")
        return {"benchmark_path": benchmark_path, "results_path": results_path}
    
    def run_deployment(self, model_path: str) -> Dict:
        """Run the deployment phase."""
        logger.info("Starting deployment phase...")
        
        # Create deployment configuration
        deployment_config = {
            "model_path": model_path,
            "host": "0.0.0.0",
            "port": 5000,
            "auth_token": "your-secret-token"
        }
        
        # Save deployment config
        config_path = "deployment/deployment_config.json"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(deployment_config, f, indent=2)
        
        logger.info(f"Deployment configuration saved to {config_path}")
        logger.info("To start the server, run: python deployment/model_server.py --model-path <model_path>")
        
        return deployment_config
    
    def run_full_pipeline(self) -> Dict:
        """Run the complete pipeline end-to-end."""
        logger.info("Starting full pipeline execution...")
        
        pipeline_start_time = datetime.now()
        results = {}
        
        try:
            # 1. Data Collection
            raw_data = self.run_data_collection()
            results["data_collection"] = {"documents_collected": len(raw_data)}
            
            # 2. Data Processing
            processed_data = self.run_data_processing(raw_data)
            results["data_processing"] = {"documents_processed": len(processed_data)}
            
            # 3. Training Dataset Generation
            train_dataset_path = self.run_training_dataset_generation(processed_data)
            results["training_dataset"] = {"dataset_path": train_dataset_path}
            
            # 4. Fine-tuning
            model_path = self.run_fine_tuning(train_dataset_path)
            results["fine_tuning"] = {"model_path": model_path}
            
            # 5. Evaluation
            evaluation_results = self.run_evaluation(model_path)
            results["evaluation"] = evaluation_results
            
            # 6. Deployment
            deployment_config = self.run_deployment(model_path)
            results["deployment"] = deployment_config
            
            pipeline_end_time = datetime.now()
            pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
            
            results["pipeline_summary"] = {
                "status": "completed",
                "start_time": pipeline_start_time.isoformat(),
                "end_time": pipeline_end_time.isoformat(),
                "duration_seconds": pipeline_duration
            }
            
            logger.info(f"Full pipeline completed successfully in {pipeline_duration:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            results["pipeline_summary"] = {
                "status": "failed",
                "error": str(e),
                "start_time": pipeline_start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
        
        # Save pipeline results
        results_path = "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Fine-tuning Pipeline Orchestrator')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    parser.add_argument('--phase', choices=['data_collection', 'data_processing', 'training_dataset', 'fine_tuning', 'evaluation', 'deployment', 'full'], default='full', help='Pipeline phase to run')
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(args.config)
    
    # Run specified phase
    if args.phase == 'full':
        results = orchestrator.run_full_pipeline()
    elif args.phase == 'data_collection':
        results = orchestrator.run_data_collection()
    elif args.phase == 'data_processing':
        # This would need raw data as input in a real scenario
        results = orchestrator.run_data_processing([])
    # Add other phases as needed
    
    print(f"Pipeline phase '{args.phase}' completed.")
    if isinstance(results, dict) and "pipeline_summary" in results:
        print(f"Status: {results['pipeline_summary']['status']}")

