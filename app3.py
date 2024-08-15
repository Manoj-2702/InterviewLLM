import streamlit as st
from adapters import AutoAdapterModel
from transformers import AutoTokenizer
import torch

# Load the base model and tokenizer
base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # You should choose a base model compatible with the adapter.
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoAdapterModel.from_pretrained(base_model_name)

# Load the adapter from the path where it's saved
adapter_path = "adapter_model.safetensors"  # Adjust path as necessary
model.load_adapter(adapter_path, config="adapter_config.json")

def predict(input_text):
    # Encode the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Set the adapter
    model.set_active_adapters("llama-3-8b-judge-interview")  # This should match the adapter's name
    # Get model predictions
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities[:, 1].item()  # Adjust based on your model's specific output

# Streamlit interface
st.title("Judgment Prediction")
user_input = st.text_area("Enter the text to predict:", "Type here...")
if st.button('Predict'):
    prediction = predict(user_input)
    st.write(f"Probability of Judgment:\n {prediction}")
