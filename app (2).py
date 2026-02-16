import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import pdfplumber
import docx2txt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
# TEXT CLEANING FUNCTION
# ---------------------------

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# ---------------------------
# FILE TEXT EXTRACTION
# ---------------------------

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

# ---------------------------
# LOAD DATASET
# ---------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("ResumeDataSet.csv")
    df['cleaned_resume'] = df['Resume'].apply(clean_text)
    return df

df = load_data()

# ---------------------------
# TF-IDF VECTOR CREATION
# ---------------------------

tfidf = TfidfVectorizer(max_features=3000)
res
