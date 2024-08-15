import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gdown
import os

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

# Text input
text_input = st.text_area("Enter the question and answer text:", height=200)

if st.button('Predict'):
    if text_input:
        # Make prediction
        prediction = predict_label(text_input)
        
        if prediction is not None:
            # Display result
            st.write(f"Predicted quality score: {prediction}")
            
            # Interpret the score
            if prediction == 1:
                st.write("Interpretation: Poor quality")
            elif prediction == 2:
                st.write("Interpretation: Fair quality")
            elif prediction == 3:
                st.write("Interpretation: Good quality")
            else:
                st.write("Interpretation: Excellent quality")
    else:
        st.write("Please enter some text to make a prediction.")