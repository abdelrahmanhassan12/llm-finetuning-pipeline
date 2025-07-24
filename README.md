# LLM Fine-tuning Pipeline

This project implements an end-to-end pipeline for fine-tuning and serving a small language model, designed for domain-specific applications such as electric vehicle charging stations Q&A systems.

## Features

- **Data Collection**: Web scraping and PDF extraction with layout preservation
- **Data Processing**: Cleaning, deduplication, quality filtering, and normalization
- **Training Dataset Generation**: Automated QA pair generation using LLM APIs
- **Fine-tuning**: Memory-efficient training with LoRA/QLoRA techniques
- **Evaluation**: Domain-specific benchmarks with ROUGE, BLEU, and inference metrics
- **Deployment**: Production-ready API server with authentication and monitoring
- **Orchestration**: Automated workflow management and CI/CD pipeline
- **MLOps**: Experiment tracking, model versioning, and performance monitoring

## Project Structure

```
llm-finetuning-pipeline/
├── config.yaml                    # Main configuration file
├── requirements.txt               # Python dependencies
├── data_collection/              # Data collection modules
│   ├── web_scraper.py            # Web scraping functionality
│   ├── pdf_extractor.py          # PDF text extraction with layout
│   └── __init__.py
├── data_processing/              # Data processing modules
│   ├── data_processor.py         # Text cleaning and normalization
│   ├── data_storage.py           # Database storage and retrieval
│   └── __init__.py
├── training_dataset/             # Training dataset generation
│   ├── qa_generator.py           # QA pair generation using LLM
│   ├── dataset_formatter.py      # Dataset formatting and splitting
│   └── __init__.py
├── fine_tuning/                  # Model fine-tuning
│   ├── model_finetuner.py        # LoRA/QLoRA fine-tuning implementation
│   ├── experiment_tracker.py     # Experiment tracking and logging
│   └── __init__.py
├── evaluation/                   # Model evaluation and benchmarking
│   ├── evaluator.py              # ROUGE, BLEU, and inference metrics
│   ├── benchmark_generator.py    # Domain-specific benchmark creation
│   └── __init__.py
├── deployment/                   # Model deployment and serving
│   ├── model_server.py           # Flask API server with authentication
│   ├── monitoring.py             # Server monitoring and alerting
│   └── __init__.py
└── orchestration/                # Pipeline orchestration and CI/CD
    ├── pipeline_orchestrator.py  # End-to-end pipeline execution
    ├── ci_cd.py                  # CI/CD automation
    └── __init__.py
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/abdelrahmanhassan12/llm-finetuning-pipeline.git
cd llm-finetuning-pipeline

# Install dependencies
pip install -r requirements.txt

# Install additional NLTK data (for evaluation)
python -c "import nltk; nltk.download('punkt')"
```

### 2. Configuration

Edit `config.yaml` to specify your target domain and data sources:

```yaml
target_domain: "electric vehicle charging stations"
use_case: "QA"
data_sources:
  - "PDFs"
base_llm: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
pdf_directory: "data/pdfs"
urls:
  - "https://example.com/ev-charging-guide"
```

### 3. Run the Pipeline

#### Full Pipeline Execution
```bash
python orchestration/pipeline_orchestrator.py --config config.yaml --phase full
```

#### Individual Phases
```bash
# Data collection only
python orchestration/pipeline_orchestrator.py --phase data_collection

# Fine-tuning only
python orchestration/pipeline_orchestrator.py --phase fine_tuning

# Evaluation only
python orchestration/pipeline_orchestrator.py --phase evaluation
```

### 4. Deploy the Model

```bash
# Start the model server
python deployment/model_server.py --model-path models/finetuned_model --host 0.0.0.0 --port 5000

# Start monitoring (in another terminal)
python deployment/monitoring.py --server-url http://localhost:5000
```

### 5. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Chat with the model
curl -X POST http://localhost:5000/chat \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main types of EV charging connectors?"}'
```

## Detailed Usage

### Data Collection

The pipeline supports multiple data sources:

1. **PDF Documents**: Place PDF files in the directory specified in `config.yaml`
2. **Web Scraping**: Add URLs to the `urls` list in the configuration

```python
from data_collection import process_pdf_directory, scrape_website

# Process PDFs
pdf_data = process_pdf_directory("data/pdfs")

# Scrape websites
scraped_text = scrape_website("https://example.com")
```

### Data Processing

Data processing includes cleaning, deduplication, and quality filtering:

```python
from data_processing import process_raw_data, DataStorage

# Process raw data
processed_data = process_raw_data(raw_data)

# Store in database
storage = DataStorage()
doc_id = storage.store_document(processed_data[0])
```

### Training Dataset Generation

Generate QA pairs from processed text:

```python
from training_dataset import generate_training_dataset, split_dataset

# Generate QA pairs
training_data = generate_training_dataset(processed_data, "training_dataset.json")

# Split into train/val/test
dataset_splits = split_dataset(training_data)
```

### Fine-tuning

Fine-tune models using LoRA/QLoRA for memory efficiency:

```python
from fine_tuning import fine_tune_model, ExperimentTracker

# Start experiment tracking
tracker = ExperimentTracker()
experiment_id = tracker.start_experiment("llm_finetuning", config)

# Fine-tune model
fine_tune_model(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    train_dataset_path="data/train.json",
    output_dir="models/finetuned_model"
)
```

### Evaluation

Evaluate models using multiple metrics:

```python
from evaluation import calculate_rouge, calculate_bleu, generate_domain_specific_benchmark

# Generate domain-specific benchmark
benchmark = generate_domain_specific_benchmark("electric vehicle charging stations")

# Calculate metrics
rouge_scores = calculate_rouge(predictions, references)
bleu_score = calculate_bleu(predictions, references)
```

## CI/CD Pipeline

Run automated testing and deployment:

```bash
# Run full CI/CD pipeline
python orchestration/ci_cd.py --stage full

# Run individual stages
python orchestration/ci_cd.py --stage tests
python orchestration/ci_cd.py --stage quality
python orchestration/ci_cd.py --stage build
python orchestration/ci_cd.py --stage deploy
```

## Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t llm-pipeline .
docker run -p 5000:5000 -v $(pwd)/models:/app/models llm-pipeline
```

## API Documentation

### Endpoints

#### Health Check
- **GET** `/health`
- Returns server status and model information

#### Text Generation
- **POST** `/generate`
- Headers: `Authorization: Bearer your-secret-token`
- Body: `{"prompt": "Your prompt here", "max_length": 512, "temperature": 0.7}`

#### Chat Interface
- **POST** `/chat`
- Headers: `Authorization: Bearer your-secret-token`
- Body: `{"message": "Your question here"}`

#### Model Information
- **GET** `/model/info`
- Returns loaded model details

## Monitoring and Logging

The pipeline includes comprehensive monitoring:

- **Server Health**: Endpoint availability and response times
- **System Metrics**: CPU, memory, and disk usage
- **Inference Performance**: Latency and throughput measurements
- **Experiment Tracking**: Training metrics and model versioning

Monitor logs:
```bash
tail -f logs/pipeline.log
```

## Configuration Options

### Model Configuration
- `base_llm`: Base model to fine-tune (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- `target_domain`: Domain for specialized training
- `use_case`: Application type (QA, chat, etc.)

### Training Configuration
- LoRA parameters (r, alpha, dropout)
- Training hyperparameters (learning rate, batch size, epochs)
- Memory optimization settings

### Deployment Configuration
- Server host and port
- Authentication settings
- Monitoring intervals

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Model Loading Errors**: Ensure model path is correct and accessible
3. **API Authentication**: Verify the authorization token is correct
4. **Slow Inference**: Check system resources and model size

### Debug Mode

Enable debug logging:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python deployment/model_server.py --debug --model-path models/finetuned_model
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `python orchestration/ci_cd.py --stage tests`
5. Submit a pull request



