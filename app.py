import os
import sys
import json
import pickle
# os.chdir("D:/Python/pytorch/handson-transformers/")
# sys.path.append("D:/Python/pytorch/handson-transformers/")
print("working dir:",os.getcwd())
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torch.nn.utils as utils
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from TransformerModel import *

import pandas as pd
import streamlit as st

####################################


st.set_page_config(layout="wide")

# Remove whitespace from the top of the page and sidebar
st.markdown("""
    <style>
    
           /* Remove blank space at top and bottom */ 
           .block-container {
               padding-top: 2rem;
               padding-bottom: 0rem;
            }
           
           /* Remove blank space at the center canvas */ 
           .st-emotion-cache-z5fcl4 {
               position: relative;
               top: -62px;
               }
           
           /* Make the toolbar transparent and the content below it clickable */ 
           .st-emotion-cache-18ni7ap {
               pointer-events: none;
               background: rgb(255 255 255 / 0%)
               }
           .st-emotion-cache-zq5wmm {
               pointer-events: auto;
               background: rgb(255 255 255);
               border-radius: 5px;
               }
    </style>
    """, unsafe_allow_html=True)


device = "cpu"#"cuda" if torch.cuda.is_available() else "cpu"
f = open('Data/vocab.json')  
vocab = json.load(f) 
vocab_size = len(vocab) # 53529
embed_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
forward_expansion = 2048
dropout = 0.1
max_len = 100
def load_model(path:str):
    # Loading the model
    model_load_path = path

    # Ensure you initialize the model with the same architecture as before
    loaded_model = TransformerModel(vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, max_len).to(device)

    # Load the saved state dictionary into the model
    loaded_model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu'), pickle_module=pickle)) # weights_only=True,

    # # Set the model to evaluation mode
    # loaded_model.eval()
    return loaded_model


def generate_text(model, vocab, input_text, max_length=200, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_tokens = [vocab.get(token, vocab['<PAD>']) for token in input_text.split()]
        input_tensor = torch.tensor(input_tokens).unsqueeze(0).to(device)  # Batch size of 1

        generated_text = input_tokens.copy()

        for _ in range(max_length):
            # Ensure positional encoding matches input sequence length
            if input_tensor.size(1) > model.positional_encoding.size(1):
                # If input is longer than positional encoding, pad positional encoding
                positional_encoding = torch.cat([model.positional_encoding, 
                                                 model.positional_encoding[:, -1:, :].repeat(1, input_tensor.size(1) - model.positional_encoding.size(1), 1)], dim=1)
            else:
                positional_encoding = model.positional_encoding[:, :input_tensor.size(1), :]

            # Generate the next token
            embed_x = model.dropout(model.embedding(input_tensor) + positional_encoding)
            output = model.encoder(embed_x, mask=None)
            
            # Apply temperature scaling
            output = output[:, -1, :] / temperature
            
            # Apply softmax and sample
            probabilities = torch.softmax(output, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            
            # Stop if the model starts predicting padding or unknown tokens repeatedly
            if next_token == vocab['<PAD>'] or (len(generated_text) > 1 and next_token == generated_text[-1]):
                break

            generated_text.append(next_token)

            # Update input_tensor for the next iteration
            input_tensor = torch.tensor(generated_text).unsqueeze(0).to(device)

        # Convert generated tokens back to words
        generated_text = [list(vocab.keys())[list(vocab.values()).index(tok)] for tok in generated_text]
        return " ".join(generated_text)
####################################

def main():
    st.title("Generate Paragraph Using Transformer Model")
    st.markdown("---")

    st.info("Here you can play around with the model which is trained on News Articles data. Also, the model is written from scratch so there can be some mistakes. This project is completely for the experiments around NLP model and how text generation using transformer works, as well as good starter point for learning PyTorch and NLP.\nThis will also help in understanding how model behaves internally and what is the base of many SOTA LLMs.")
    st.markdown("** :red[WARNING]: *This model can generate any random text which might be irrelevent, since it is trained on less text records and for less epochs!!!")
    
    st.text("Sample train data:")
    train_data = pd.read_excel("Data/Articles.xlsx")
    st.dataframe(train_data.sample(5))
    st.markdown("---")

    instr = "Input any keyword/phrase here. (E.g. business and economy) [Text geeneration might be slow because, model is trained on GPU and inferencing on CPU instance.]"
    st.text(instr)
    with st.form('chat_input_form'):
        # Create two columns; adjust the ratio to your liking
        col1, col2 = st.columns([3,1]) 

        # Use the first column for text input
        with col1:
            prompt = st.text_input( help = instr, label=instr, label_visibility='collapsed')
        # Use the second column for the submit button
        with col2:
            submitted = st.form_submit_button("Generate ✨")
        
        if prompt and submitted:
            model = load_model("exports/trained-transformer_model.pth")
            output = generate_text(model=model, vocab=vocab, input_text=prompt, max_length=200)
            st.write(f"Model Response: \n{output}")



if __name__ == '__main__':
    main()