import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize model and tokenizer
device = torch.device("cpu")  # Always CPU
model_name = "google/flan-t5-large"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

def generate_response(prompt, max_length=100):
    try:
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = inputs.to(device)
        
        # Generate response
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=5,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        return f"Error generating response: {e}"

def chatbot():
    print("Welcome to the Flan-T5 Chatbot!")
    print("Type 'quit' to exit.")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            if not user_input.strip():
                print("Please enter a valid input.")
                continue
                
            # Create prompt with context
            prompt = f"""
You are a helpful assistant. Respond to the user's input in a conversational and friendly manner.

User: {user_input}
Assistant: """
            
            response = generate_response(prompt)
            print(f"Bot: {response}")
            
        except  KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    chatbot()
