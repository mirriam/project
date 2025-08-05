from flask import Flask, request, jsonify
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

app = Flask(__name__)

# Load a paraphrasing model (using a lightweight model for simplicity)
paraphraser = pipeline("text2text-generation", model="t5-small")

def paraphrase_text(text):
    try:
        # Split text into sentences
        sentences = sent_tokenize(text)
        paraphrased_sentences = []
        
        # Paraphrase each sentence
        for sentence in sentences:
            # Generate paraphrased text
            result = paraphraser(f"paraphrase: {sentence}", max_length=50, num_return_sequences=1)
            paraphrased_sentences.append(result[0]['generated_text'])
        
        # Join paraphrased sentences
        return " ".join(paraphrased_sentences)
    except Exception as e:
        return f"Error during paraphrasing: {str(e)}"

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide text to paraphrase"}), 400
    
    input_text = data['text']
    paraphrased_text = paraphrase_text(input_text)
    
    return jsonify({
        "original": input_text,
        "paraphrased": paraphrased_text
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
