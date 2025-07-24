import time
import psutil
import requests
import json
import logging
from datetime import datetime
from typing import Dict, List
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelServerMonitor:
    def __init__(self, server_url: str = "http://localhost:5000", check_interval: int = 30):
        self.server_url = server_url
        self.check_interval = check_interval
        self.metrics_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        
    def check_server_health(self) -> Dict:
        """
        Check the health of the model server.
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=10)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time_ms": response.elapsed.total_seconds() * 1000,
                    "server_data": response.json()
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code,
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_system_metrics(self) -> Dict:
        """
        Get system resource metrics.
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3)
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {str(e)}")
            return {}
    
    def test_model_inference(self, test_prompt: str = "What is an electric vehicle?") -> Dict:
        """
        Test model inference performance.
        """
        try:
            payload = {
                "message": test_prompt
            }
            headers = {
                "Authorization": "Bearer your-secret-token",
                "Content-Type": "application/json"
            }
            
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/chat",
                json=payload,
                headers=headers,
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "success",
                    "total_time_ms": (end_time - start_time) * 1000,
                    "server_inference_time_ms": data.get("inference_time_ms", 0),
                    "response_length": len(data.get("response", ""))
                }
            else:
                return {
                    "status": "error",
                    "status_code": response.status_code,
                    "error": response.text
                }
                
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def collect_metrics(self) -> Dict:
        """
        Collect all metrics in one go.
        """
        timestamp = datetime.now().isoformat()
        
        metrics = {
            "timestamp": timestamp,
            "server_health": self.check_server_health(),
            "system_metrics": self.get_system_metrics(),
            "inference_test": self.test_model_inference()
        }
        
        return metrics
    
    def start_monitoring(self):
        """
        Start continuous monitoring in a separate thread.
        """
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Monitoring started")
    
    def stop_monitoring(self):
        """
        Stop continuous monitoring.
        """
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self):
        """
        Main monitoring loop.
        """
        while self.is_monitoring:
            try:
                metrics = self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 metrics to prevent memory issues
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # Log key metrics
                server_status = metrics["server_health"]["status"]
                cpu_percent = metrics["system_metrics"].get("cpu_percent", 0)
                memory_percent = metrics["system_metrics"].get("memory_percent", 0)
                inference_status = metrics["inference_test"]["status"]
                
                logger.info(f"Server: {server_status}, CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Inference: {inference_status}")
                
                # Check for alerts
                self._check_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
            
            time.sleep(self.check_interval)
    
    def _check_alerts(self, metrics: Dict):
        """
        Check for alert conditions and log warnings.
        """
        # High CPU usage
        cpu_percent = metrics["system_metrics"].get("cpu_percent", 0)
        if cpu_percent > 80:
            logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
        
        # High memory usage
        memory_percent = metrics["system_metrics"].get("memory_percent", 0)
        if memory_percent > 85:
            logger.warning(f"High memory usage: {memory_percent:.1f}%")
        
        # Server unhealthy
        if metrics["server_health"]["status"] != "healthy":
            logger.warning(f"Server unhealthy: {metrics['server_health']}")
        
        # Slow inference
        inference_time = metrics["inference_test"].get("total_time_ms", 0)
        if inference_time > 10000:  # 10 seconds
            logger.warning(f"Slow inference: {inference_time:.1f}ms")
    
    def get_metrics_summary(self) -> Dict:
        """
        Get a summary of recent metrics.
        """
        if not self.metrics_history:
            return {"error": "No metrics available"}
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 metrics
        
        # Calculate averages
        avg_cpu = sum(m["system_metrics"].get("cpu_percent", 0) for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m["system_metrics"].get("memory_percent", 0) for m in recent_metrics) / len(recent_metrics)
        
        # Count successful inferences
        successful_inferences = sum(1 for m in recent_metrics if m["inference_test"]["status"] == "success")
        
        return {
            "total_metrics_collected": len(self.metrics_history),
            "recent_metrics_count": len(recent_metrics),
            "average_cpu_percent": avg_cpu,
            "average_memory_percent": avg_memory,
            "successful_inference_rate": successful_inferences / len(recent_metrics),
            "latest_timestamp": recent_metrics[-1]["timestamp"]
        }
    
    def save_metrics_to_file(self, filename: str):
        """
        Save metrics history to a JSON file.
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            logger.info(f"Metrics saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Server Monitor')
    parser.add_argument('--server-url', default='http://localhost:5000', help='Model server URL')
    parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds')
    parser.add_argument('--duration', type=int, default=300, help='Monitoring duration in seconds')
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = ModelServerMonitor(args.server_url, args.interval)
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Run for specified duration
        time.sleep(args.duration)
        
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Print summary
        summary = monitor.get_metrics_summary()
        print("\nMonitoring Summary:")
        print(json.dumps(summary, indent=2))
        
        # Save metrics
        monitor.save_metrics_to_file("monitoring_metrics.json")
        
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
        monitor.stop_monitoring()

