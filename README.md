Paraphrasing API
A simple Flask-based API that paraphrases input text using the T5-small model from Hugging Face's Transformers library.
Setup Instructions

Clone the Repository
git clone https://github.com/your-username/paraphrase-api.git
cd paraphrase-api


Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies
pip install -r requirements.txt


Run the Application
python paraphrase_app.py


Test the APIUse a tool like curl or Postman to send a POST request:
curl -X POST -H "Content-Type: application/json" -d '{"text":"This is a sample sentence to paraphrase."}' http://localhost:5000/paraphrase



API Endpoint

POST /paraphrase
Input: JSON object with a text field (e.g., {"text": "Your sentence here"})
Output: JSON object with original and paraphrased fields



Example
Request:
{"text": "The quick brown fox jumps over the lazy dog."}

Response:
{
  "original": "The quick brown fox jumps over the lazy dog.",
  "paraphrased": "The swift brown fox leaps over the idle dog."
}

Deployment

Host on platforms like Render, Heroku, or AWS.
Ensure the hosting service supports Python and has sufficient memory for the T5 model.

Notes

The t5-small model is used for efficiency but may produce less accurate paraphrases than larger models like t5-base or t5-large.
For production, consider adding authentication and rate limiting.
