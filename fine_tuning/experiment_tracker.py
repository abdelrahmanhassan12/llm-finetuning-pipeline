import json
import os
from datetime import datetime
from typing import Dict, Any

class ExperimentTracker:
    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = experiment_dir
        os.makedirs(experiment_dir, exist_ok=True)
        self.current_experiment = None
        
    def start_experiment(self, experiment_name: str, config: Dict[str, Any]) -> str:
        """
        Start a new experiment with the given name and configuration.
        Returns the experiment ID.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        os.makedirs(experiment_path, exist_ok=True)
        
        self.current_experiment = {
            "id": experiment_id,
            "name": experiment_name,
            "path": experiment_path,
            "config": config,
            "start_time": datetime.now().isoformat(),
            "metrics": {},
            "logs": []
        }
        
        # Save initial experiment metadata
        self._save_experiment_metadata()
        
        return experiment_id
    
    def log_metric(self, metric_name: str, value: float, step: int = None):
        """Log a metric value."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        if metric_name not in self.current_experiment["metrics"]:
            self.current_experiment["metrics"][metric_name] = []
        
        metric_entry = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        
        if step is not None:
            metric_entry["step"] = step
        
        self.current_experiment["metrics"][metric_name].append(metric_entry)
        
        # Save updated metrics
        self._save_experiment_metadata()
    
    def log_message(self, message: str, level: str = "INFO"):
        """Log a message."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message
        }
        
        self.current_experiment["logs"].append(log_entry)
        
        # Also print to console
        print(f"[{level}] {message}")
        
        # Save updated logs
        self._save_experiment_metadata()
    
    def save_artifact(self, artifact_name: str, artifact_data: Any):
        """Save an artifact (e.g., model, dataset, plot) to the experiment directory."""
        if not self.current_experiment:
            raise ValueError("No active experiment. Call start_experiment() first.")
        
        artifact_path = os.path.join(self.current_experiment["path"], f"{artifact_name}.json")
        
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(artifact_data, f, indent=2, ensure_ascii=False)
        
        self.log_message(f"Artifact '{artifact_name}' saved to {artifact_path}")
    
    def end_experiment(self, status: str = "completed"):
        """End the current experiment."""
        if not self.current_experiment:
            raise ValueError("No active experiment to end.")
        
        self.current_experiment["end_time"] = datetime.now().isoformat()
        self.current_experiment["status"] = status
        
        # Save final experiment metadata
        self._save_experiment_metadata()
        
        self.log_message(f"Experiment {self.current_experiment['id']} ended with status: {status}")
        
        # Reset current experiment
        experiment_id = self.current_experiment["id"]
        self.current_experiment = None
        
        return experiment_id
    
    def _save_experiment_metadata(self):
        """Save experiment metadata to a JSON file."""
        if not self.current_experiment:
            return
        
        metadata_path = os.path.join(self.current_experiment["path"], "experiment_metadata.json")
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.current_experiment, f, indent=2, ensure_ascii=False)
    
    def list_experiments(self) -> list:
        """List all experiments in the experiment directory."""
        experiments = []
        
        if not os.path.exists(self.experiment_dir):
            return experiments
        
        for experiment_folder in os.listdir(self.experiment_dir):
            experiment_path = os.path.join(self.experiment_dir, experiment_folder)
            metadata_path = os.path.join(experiment_path, "experiment_metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    experiment_data = json.load(f)
                    experiments.append(experiment_data)
        
        return experiments
    
    def load_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Load experiment data by ID."""
        experiment_path = os.path.join(self.experiment_dir, experiment_id)
        metadata_path = os.path.join(experiment_path, "experiment_metadata.json")
        
        if not os.path.exists(metadata_path):
            raise ValueError(f"Experiment {experiment_id} not found.")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

if __name__ == "__main__":
    # Example usage
    tracker = ExperimentTracker()
    
    # Start an experiment
    config = {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "learning_rate": 2e-4,
        "batch_size": 4,
        "epochs": 3
    }
    
    experiment_id = tracker.start_experiment("llm_finetuning_test", config)
    
    # Log some metrics
    tracker.log_metric("train_loss", 0.5, step=1)
    tracker.log_metric("train_loss", 0.3, step=2)
    tracker.log_metric("eval_loss", 0.4, step=2)
    
    # Log messages
    tracker.log_message("Training started")
    tracker.log_message("Epoch 1 completed")
    
    # Save an artifact
    tracker.save_artifact("training_config", config)
    
    # End experiment
    tracker.end_experiment("completed")
    
    # List experiments
    experiments = tracker.list_experiments()
    print(f"Found {len(experiments)} experiments")
    
    # Load experiment
    loaded_experiment = tracker.load_experiment(experiment_id)
    print(f"Loaded experiment: {loaded_experiment['name']}")

