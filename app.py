import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL = "EleutherAI/gpt-neo-2.7B"

# Try loading local model once at startup
try:
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL)
    model.to(device)
    model.eval()
    local_model_loaded = True
    print("Model loaded successfully.")
except Exception as e:
    print(f"Local model loading failed: {e}")
    local_model_loaded = False

def generate_response(user_input, max_length=100):
    if not local_model_loaded:
        return "Model is not loaded, cannot generate response."
    try:
        # For causal LM, you can format the prompt conversationally like this:
        prompt = f"User: {user_input}\nAssistant:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
            max_length=inputs["input_ids"].shape[1] + max_length,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        # The output includes the prompt, so remove it to return only the assistant response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt part
        response = generated_text[len(prompt):].strip()
        
        return response
    except Exception as e:
        return f"Error generating response: {e}"

with gr.Blocks() as demo:
    gr.Markdown("# 💬 GPT-Neo 2.7B Chatbot\nType a message and chat with the AI.")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your message")
    clear = gr.Button("Clear")

    def respond(message, chat_history):
        bot_response = generate_response(message)
        chat_history.append((message, bot_response))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch()
