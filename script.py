import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, request, jsonify
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

device = torch.device("cpu")
model_name = "google/flan-t5-small"
try:
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    logger.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    exit(1)

def generate_response(prompt, max_length=100):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = inputs.to(device)
        outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=max_length, num_beams=5, length_penalty=1.0, early_stopping=True, no_repeat_ngram_size=2)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {e}"

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            logger.warning("Invalid API key")
            return jsonify({'error': 'Invalid API key'}), 401
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("Missing or invalid message in request")
            return jsonify({'error': 'Missing message in request'}), 400
        user_input = data['message']
        if not user_input.strip():
            logger.warning("Empty message received")
            return jsonify({'error': 'Empty message'}), 400
        prompt = f"""
You are a helpful assistant. Respond to the user's input in a conversational and friendly manner.
User: {user_input}
Assistant: """
        response = generate_response(prompt)
        logger.info(f"User input: {user_input}, Response: {response}")
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error in /api/chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check requested")
    return jsonify({'status': 'healthy'}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
