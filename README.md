# Hands-on Transformers
**End-to-end implementation of Transformers using PyTorch from scratch**

---

Implementing end to end Transformer model using PyTorch from scratch, and training it to generate paragraphs if given a keyword or phrase as a input. 

### Files and usage:
  - **TransformerModel.py** --> Model class containing all logic and architecture of Transformer model
  - **train_beta.ipynb** --> Jupyter Notebook to train and do the sample inference on trained model
  - **trained-transformer_model.pth** --> Trained model checkpoint _(saved state dict)_
  - **Articles.xlsx** --> Dataset used to train the model (https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles)
  - **requirements.txt** --> pip freeze of dependencies 
--- 

### Working:
_The model takes a keyword or phrase, tokenizes it, and then iteratively generates text by predicting the next token in the sequence.
The model uses embedding, positional encoding, and an encoder-decoder architecture to generate coherent text.
Sampling strategies like temperature scaling and top-k sampling help to produce varied and natural outputs._

### Setup and Usage: 
* Hardware used:
  - CPU: Intel i7-10750H (2.60 GHz)
  - RAM: 16 GB
  - GPU: NVIDIA GeForce RTX 2060 (6 GB)
    
* Create virtual environment
```code
virtualenv env
```

* Activate virtual environment
```code
./env/Scripts/activate
```

* Installing dependancies
```code
pip install -r requirements.txt
```
---

### Dashboard:
Dashboard which can generate the paragrph using the trained model if given a keyword or phrase as a input. 
* Running a Streamlit app
```code
streamlit run app.py
```

![Streamlit App](https://github.com/user-attachments/assets/7d373d93-5bdb-4f27-8686-55547e30801f)

