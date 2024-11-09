import streamlit as st
import pickle 
import pandas as pd 
import nltk 
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

tfidf = pickle.load(open('vectorizer.pkl', 'rb')) 
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/Sms Spam Classifier")

input_sms = st.text_area("Enter the message")

# 1. preprocess 
# 2. vectorize 
# 3. predict
# 4. display the output

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y=[]
    for item in text:
        if item.isalnum():
            y.append(item)
    text = y[:]
    y.clear() # y lai khaali garaye
    for item in text:
        if item not in stopwords.words('english') and item not in string.punctuation:
            y.append(item)

    text = y[:]
    y.clear() 
    for item in text:
        y.append(ps.stem(item))
    return " ".join(y)


if st.button('Predict'):
    # yedi predict button click garepaar
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]    # coz y_pred = mnb.predict(X_test) dida shape of y_pred is 1D array , so we need to access the 0th element of the array , also mnb.predict(input) expects a 2d array so we need to pass a 2d array


    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

# streamlit run app.py
# pip freeze > requirements.txt , deployment host will install all the packages mentioned in requirements.txt