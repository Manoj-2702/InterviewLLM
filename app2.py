import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown
import os
import pandas as pd
import csv

# Streamlit app
st.title('Question Answer Quality Predictor')

@st.cache_resource
def download_model():
    url = "https://drive.google.com/uc?id=1FP6l-vN-gFBzM-zb_E0bsw-jXY0BHr0N"
    output = "bert_model.pth"
    gdown.download(url, output, quiet=False)
    return output

@st.cache_resource
def load_model():
    model_path = download_model()
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None

    try:
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=4,
            output_attentions=False,
            output_hidden_states=False,
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_tokenizer():
    # Load the tokenizer
    return AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()

def prepare_input(text, max_length=512):
    # Tokenize and encode the text
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    return encoded_dict['input_ids'], encoded_dict['attention_mask']

def predict_label(text):
    if model is None:
        st.error("Model not loaded. Cannot make prediction.")
        return None

    # Prepare the input
    input_ids, attention_mask = prepare_input(text)
    
    # Perform the prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get the predicted class
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Convert back to original label (add 1 since we subtracted 1 earlier)
    original_label = predicted_class + 1
    
    return original_label

def interpret_score(score):
    if score == 1:
        return "Poor quality"
    elif score == 2:
        return "Fair quality"
    elif score == 3:
        return "Good quality"
    else:
        return "Excellent quality"

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    df = pd.read_csv(uploaded_file)
    
    # Check if the CSV has the required column
    if 'question_cand_answer' not in df.columns:
        st.error("The CSV file must contain a 'text' column with the question and answer text.")
    else:
        # Process each row and make predictions
        predictions = []
        for text in df['question_cand_answer']:
            prediction = predict_label(text)
            predictions.append(prediction)
        
        # Add predictions to the dataframe
        df['predicted_score'] = predictions
        df['interpretation'] = df['predicted_score'].apply(interpret_score)
        
        # Display the results
        st.write("Predictions:")
        st.dataframe(df)
        
        # Offer to download the results
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
else:
    st.write("Please upload a CSV file to make predictions.")

# Optional: Add a text input for single predictions
st.write("Or enter a single question and answer for prediction:")
text_input = st.text_area("Enter the question and answer text:", height=200)

if st.button('Predict Single Entry'):
    if text_input:
        prediction = predict_label(text_input)
        if prediction is not None:
            st.write(f"Predicted quality score: {prediction}")
            st.write(f"Interpretation: {interpret_score(prediction)}")
    else:
        st.write("Please enter some text to make a prediction.")