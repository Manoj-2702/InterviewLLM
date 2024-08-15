# import streamlit as st
# from adapters import AutoAdapterModel
# from transformers import AutoTokenizer
# import torch

# # Load the base model and tokenizer
# base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # You should choose a base model compatible with the adapter.
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# model = AutoAdapterModel.from_pretrained(base_model_name)

# # Load the adapter from the path where it's saved
# adapter_path = "adapter_model.safetensors"  # Adjust path as necessary
# model.load_adapter(adapter_path, config="adapter_config.json")

# def predict(input_text):
#     # Encode the input text
#     inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#     # Set the adapter
#     model.set_active_adapters("llama-3-8b-judge-interview")  # This should match the adapter's name
#     # Get model predictions
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     probabilities = torch.nn.functional.softmax(logits, dim=-1)
#     return probabilities[:, 1].item()  # Adjust based on your model's specific output

# # Streamlit interface
# st.title("Judgment Prediction")
# user_input = st.text_area("Enter the text to predict:", "Type here...")
# if st.button('Predict'):
#     prediction = predict(user_input)
#     st.write(f"Probability of Judgment:\n {prediction}")


import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os

# Streamlit interface
st.title("Judgment Prediction")

# Function to load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    # Get the Hugging Face token from Streamlit secrets
    hf_token = st.secrets["hf_token"]
    
    # Set the token as an environment variable
    os.environ["HF_TOKEN"] = hf_token
    
    # Load the base model and tokenizer
    base_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, token=hf_token, device_map="auto")
    
    # Load the adapter
    adapter_path = "adapter_model.safetensors"  # Replace with your adapter path
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return tokenizer, model

# Load model and tokenizer
tokenizer, model = load_model_and_tokenizer()

def predict(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, num_return_sequences=1)
    
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_text

# Text input
user_input = st.text_area("Enter the text to predict:", "Type here...")
if st.button('Predict'):
    prediction = predict(user_input)
    st.write(f"Model Output: {prediction}")

# File upload for batch processing
uploaded_file = st.file_uploader("Choose a CSV file for batch processing", type="csv")
if uploaded_file is not None:
    import pandas as pd
    df = pd.read_csv(uploaded_file)
    if 'text' in df.columns:
        st.write("Processing CSV file...")
        df['prediction'] = df['text'].apply(predict)
        st.write(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download predictions",
            csv,
            "predictions.csv",
            "text/csv",
            key='download-csv'
        )
    else:
        st.error("CSV file must contain a 'text' column.")