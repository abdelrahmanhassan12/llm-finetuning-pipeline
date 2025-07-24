from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import time
import logging
from functools import wraps

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for model and tokenizer
model = None
tokenizer = None
model_loaded = False

def require_auth(f):
    """
    Simple authentication decorator.
    In production, you would use proper authentication mechanisms.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_token = request.headers.get('Authorization')
        if not auth_token or auth_token != 'Bearer your-secret-token':
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated_function

def load_model(model_path: str, base_model_name: str = None):
    """
    Load the fine-tuned model and tokenizer.
    """
    global model, tokenizer, model_loaded
    
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model if specified, otherwise load from model_path
        if base_model_name:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            # Load PEFT model
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        model.eval()  # Set to evaluation mode
        model_loaded = True
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        model_loaded = False

def generate_response(prompt: str, max_length: int = 512, temperature: float = 0.7) -> str:
    """
    Generate a response using the loaded model.
    """
    if not model_loaded:
        return "Model not loaded"
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the response
        response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': time.time()
    })

@app.route('/generate', methods=['POST'])
@require_auth
def generate():
    """
    Generate text based on input prompt.
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'prompt' not in data:
            return jsonify({'error': 'Missing prompt in request'}), 400
        
        prompt = data['prompt']
        max_length = data.get('max_length', 512)
        temperature = data.get('temperature', 0.7)
        
        # Generate response
        response = generate_response(prompt, max_length, temperature)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        return jsonify({
            'response': response,
            'inference_time_ms': inference_time * 1000,
            'prompt_length': len(prompt),
            'response_length': len(response)
        })
        
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
@require_auth
def chat():
    """
    Chat endpoint for conversational interactions.
    """
    start_time = time.time()
    
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message in request'}), 400
        
        user_message = data['message']
        
        # Format as instruction-following prompt
        prompt = f"### Instruction:\nAnswer the following question about electric vehicle charging stations.\n### Input:\n{user_message}\n### Output:\n"
        
        response = generate_response(prompt, max_length=512, temperature=0.7)
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        return jsonify({
            'response': response,
            'inference_time_ms': inference_time * 1000
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model.
    """
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        info = {
            'model_loaded': model_loaded,
            'tokenizer_vocab_size': len(tokenizer) if tokenizer else 0,
            'model_parameters': sum(p.numel() for p in model.parameters()) if model else 0,
            'device': str(next(model.parameters()).device) if model else 'unknown'
        }
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Model Server')
    parser.add_argument('--model-path', required=True, help='Path to the fine-tuned model')
    parser.add_argument('--base-model', help='Base model name (for PEFT models)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load model on startup
    if os.path.exists(args.model_path):
        load_model(args.model_path, args.base_model)
    else:
        logger.warning(f"Model path {args.model_path} does not exist. Server will start without model.")
    
    # Start server
    app.run(host=args.host, port=args.port, debug=args.debug)

