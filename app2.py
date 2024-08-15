import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

@st.cache_resource
def load_model():
    # Load the model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels = 4,
        output_attentions = False,
        output_hidden_states = False,
    )
    model.load_state_dict(torch.load('bert_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    # Load the tokenizer
    return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()

def prepare_input(text, max_length=512):
    # Tokenize and encode the text
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens = True,
        max_length = max_length,
        pad_to_max_length = True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )
    
    return encoded_dict['input_ids'], encoded_dict['attention_mask']

def predict_label(text):
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

# Streamlit app
st.title('Question Answer Quality Predictor')

# Text input
text_input = st.text_area("Enter the question and answer text:", height=200)

if st.button('Predict'):
    if text_input:
        # Make prediction
        prediction = predict_label(text_input)
        
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