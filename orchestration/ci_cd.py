import os
import subprocess
import json
import yaml
from datetime import datetime
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CICDPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {}
    
    def run_tests(self) -> Dict:
        """Run unit tests and integration tests."""
        logger.info("Running tests...")
        
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests_passed": 0,
            "tests_failed": 0,
            "test_details": []
        }
        
        # List of test commands to run
        test_commands = [
            "python -m pytest tests/ -v",  # Unit tests
            "python -c 'import data_collection; print(\"Data collection module imported successfully\")'",
            "python -c 'import data_processing; print(\"Data processing module imported successfully\")'",
            "python -c 'import training_dataset; print(\"Training dataset module imported successfully\")'",
            "python -c 'import fine_tuning; print(\"Fine tuning module imported successfully\")'",
            "python -c 'import evaluation; print(\"Evaluation module imported successfully\")'",
            "python -c 'import deployment; print(\"Deployment module imported successfully\")'",
        ]
        
        for cmd in test_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    test_results["tests_passed"] += 1
                    test_results["test_details"].append({
                        "command": cmd,
                        "status": "passed",
                        "output": result.stdout
                    })
                    logger.info(f"✓ Test passed: {cmd}")
                else:
                    test_results["tests_failed"] += 1
                    test_results["test_details"].append({
                        "command": cmd,
                        "status": "failed",
                        "error": result.stderr,
                        "output": result.stdout
                    })
                    logger.error(f"✗ Test failed: {cmd}")
                    
            except subprocess.TimeoutExpired:
                test_results["tests_failed"] += 1
                test_results["test_details"].append({
                    "command": cmd,
                    "status": "timeout",
                    "error": "Test timed out after 60 seconds"
                })
                logger.error(f"✗ Test timed out: {cmd}")
                
            except Exception as e:
                test_results["tests_failed"] += 1
                test_results["test_details"].append({
                    "command": cmd,
                    "status": "error",
                    "error": str(e)
                })
                logger.error(f"✗ Test error: {cmd} - {str(e)}")
        
        # Save test results
        with open("test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Tests completed: {test_results['tests_passed']} passed, {test_results['tests_failed']} failed")
        return test_results
    
    def check_code_quality(self) -> Dict:
        """Run code quality checks."""
        logger.info("Running code quality checks...")
        
        quality_results = {
            "timestamp": datetime.now().isoformat(),
            "checks": []
        }
        
        # Code quality commands
        quality_commands = [
            ("flake8", "flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics"),
            ("pylint", "pylint --errors-only data_collection/ data_processing/ training_dataset/ fine_tuning/ evaluation/ deployment/ orchestration/"),
            ("black_check", "black --check --diff ."),
        ]
        
        for check_name, cmd in quality_commands:
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                quality_results["checks"].append({
                    "name": check_name,
                    "command": cmd,
                    "status": "passed" if result.returncode == 0 else "failed",
                    "output": result.stdout,
                    "error": result.stderr
                })
                
                if result.returncode == 0:
                    logger.info(f"✓ Code quality check passed: {check_name}")
                else:
                    logger.warning(f"⚠ Code quality check failed: {check_name}")
                    
            except subprocess.TimeoutExpired:
                quality_results["checks"].append({
                    "name": check_name,
                    "command": cmd,
                    "status": "timeout",
                    "error": "Check timed out after 120 seconds"
                })
                logger.warning(f"⚠ Code quality check timed out: {check_name}")
                
            except Exception as e:
                quality_results["checks"].append({
                    "name": check_name,
                    "command": cmd,
                    "status": "error",
                    "error": str(e)
                })
                logger.warning(f"⚠ Code quality check error: {check_name} - {str(e)}")
        
        # Save quality results
        with open("quality_results.json", "w") as f:
            json.dump(quality_results, f, indent=2)
        
        return quality_results
    
    def build_and_package(self) -> Dict:
        """Build and package the application."""
        logger.info("Building and packaging application...")
        
        build_results = {
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "artifacts": []
        }
        
        try:
            # Create requirements.txt if it doesn't exist
            if not os.path.exists("requirements.txt"):
                requirements = [
                    "torch>=1.9.0",
                    "transformers>=4.20.0",
                    "peft>=0.3.0",
                    "datasets>=2.0.0",
                    "flask>=2.0.0",
                    "flask-cors>=3.0.0",
                    "requests>=2.25.0",
                    "beautifulsoup4>=4.9.0",
                    "pdfplumber>=0.6.0",
                    "rouge-score>=0.1.0",
                    "nltk>=3.6.0",
                    "psutil>=5.8.0",
                    "pyyaml>=5.4.0",
                    "sqlite3"  # Usually built-in with Python
                ]
                
                with open("requirements.txt", "w") as f:
                    f.write("\n".join(requirements))
                
                build_results["artifacts"].append("requirements.txt")
                logger.info("✓ Created requirements.txt")
            
            # Create setup.py if it doesn't exist
            if not os.path.exists("setup.py"):
                setup_content = '''
from setuptools import setup, find_packages

setup(
    name="llm-finetuning-pipeline",
    version="1.0.0",
    description="End-to-end pipeline for fine-tuning and serving small language models",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "peft>=0.3.0",
        "datasets>=2.0.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "pdfplumber>=0.6.0",
        "rouge-score>=0.1.0",
        "nltk>=3.6.0",
        "psutil>=5.8.0",
        "pyyaml>=5.4.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "llm-pipeline=orchestration.pipeline_orchestrator:main",
        ],
    },
)
'''
                with open("setup.py", "w") as f:
                    f.write(setup_content)
                
                build_results["artifacts"].append("setup.py")
                logger.info("✓ Created setup.py")
            
            # Create Docker files
            dockerfile_content = '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "deployment/model_server.py", "--model-path", "/app/models/finetuned_model"]
'''
            
            with open("Dockerfile", "w") as f:
                f.write(dockerfile_content)
            
            build_results["artifacts"].append("Dockerfile")
            logger.info("✓ Created Dockerfile")
            
            # Create docker-compose.yml
            docker_compose_content = '''
version: '3.8'

services:
  llm-server:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - FLASK_ENV=production
    restart: unless-stopped

  monitoring:
    build: .
    command: python deployment/monitoring.py --server-url http://llm-server:5000
    depends_on:
      - llm-server
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
'''
            
            with open("docker-compose.yml", "w") as f:
                f.write(docker_compose_content)
            
            build_results["artifacts"].append("docker-compose.yml")
            logger.info("✓ Created docker-compose.yml")
            
        except Exception as e:
            build_results["status"] = "failed"
            build_results["error"] = str(e)
            logger.error(f"✗ Build failed: {str(e)}")
        
        # Save build results
        with open("build_results.json", "w") as f:
            json.dump(build_results, f, indent=2)
        
        return build_results
    
    def deploy_to_staging(self) -> Dict:
        """Deploy to staging environment."""
        logger.info("Deploying to staging environment...")
        
        deploy_results = {
            "timestamp": datetime.now().isoformat(),
            "environment": "staging",
            "status": "success"
        }
        
        try:
            # In a real scenario, this would deploy to actual staging environment
            # For now, we'll simulate the deployment process
            
            # Create staging configuration
            staging_config = {
                "environment": "staging",
                "debug": True,
                "host": "0.0.0.0",
                "port": 5001,
                "model_path": "models/staging_model"
            }
            
            os.makedirs("staging", exist_ok=True)
            with open("staging/config.json", "w") as f:
                json.dump(staging_config, f, indent=2)
            
            deploy_results["config_path"] = "staging/config.json"
            logger.info("✓ Staging deployment configuration created")
            
        except Exception as e:
            deploy_results["status"] = "failed"
            deploy_results["error"] = str(e)
            logger.error(f"✗ Staging deployment failed: {str(e)}")
        
        return deploy_results
    
    def run_full_cicd(self) -> Dict:
        """Run the complete CI/CD pipeline."""
        logger.info("Starting full CI/CD pipeline...")
        
        cicd_start_time = datetime.now()
        results = {
            "pipeline_start": cicd_start_time.isoformat(),
            "stages": {}
        }
        
        # Stage 1: Tests
        test_results = self.run_tests()
        results["stages"]["tests"] = test_results
        
        if test_results["tests_failed"] > 0:
            logger.error("Tests failed. Stopping CI/CD pipeline.")
            results["status"] = "failed"
            results["failed_stage"] = "tests"
            return results
        
        # Stage 2: Code Quality
        quality_results = self.check_code_quality()
        results["stages"]["code_quality"] = quality_results
        
        # Stage 3: Build and Package
        build_results = self.build_and_package()
        results["stages"]["build"] = build_results
        
        if build_results["status"] != "success":
            logger.error("Build failed. Stopping CI/CD pipeline.")
            results["status"] = "failed"
            results["failed_stage"] = "build"
            return results
        
        # Stage 4: Deploy to Staging
        deploy_results = self.deploy_to_staging()
        results["stages"]["deploy_staging"] = deploy_results
        
        if deploy_results["status"] != "success":
            logger.error("Staging deployment failed.")
            results["status"] = "failed"
            results["failed_stage"] = "deploy_staging"
            return results
        
        # Pipeline completed successfully
        cicd_end_time = datetime.now()
        pipeline_duration = (cicd_end_time - cicd_start_time).total_seconds()
        
        results["status"] = "success"
        results["pipeline_end"] = cicd_end_time.isoformat()
        results["duration_seconds"] = pipeline_duration
        
        logger.info(f"CI/CD pipeline completed successfully in {pipeline_duration:.2f} seconds")
        
        # Save CI/CD results
        with open("cicd_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CI/CD Pipeline')
    parser.add_argument('--stage', choices=['tests', 'quality', 'build', 'deploy', 'full'], default='full', help='CI/CD stage to run')
    parser.add_argument('--config', default='config.yaml', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create CI/CD pipeline
    cicd = CICDPipeline(args.config)
    
    # Run specified stage
    if args.stage == 'full':
        results = cicd.run_full_cicd()
    elif args.stage == 'tests':
        results = cicd.run_tests()
    elif args.stage == 'quality':
        results = cicd.check_code_quality()
    elif args.stage == 'build':
        results = cicd.build_and_package()
    elif args.stage == 'deploy':
        results = cicd.deploy_to_staging()
    
    print(f"CI/CD stage '{args.stage}' completed.")
    if isinstance(results, dict) and "status" in results:
        print(f"Status: {results['status']}")

