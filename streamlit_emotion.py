import streamlit as st
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#fromm load model 
@st.cache_resource
def load_saved_model():   
   with open('healthcare.pkl','rb')as file:
        healt=pickle.load(file)
   with open('tfidf.pkl','rb')as file:
       tfidf=pickle.load(file)
   with open('disease_encoder.pkl','rb')as file:
       disease=pickle.load(file) 
       return healt,tfidf,disease
healt,tfidf,disease=load_saved_model()
#import nltk
nltk.download('stopwords')
stop_words=set(stopwords.words('english'))
ps=PorterStemmer()
#text preprocessing function
def clean_text(text):
    words=text.lower().split()
    cleaned=[ps.stem(w) for w in words if w not in stop_words]
    return"".join(cleaned)
st.title('🩺Healthcare Prediction💊')
st.write("Accurate Disease Prediction In All Healthcare Disease")
#user input 
Patient_ID=st.number_input("Patient_ID:",min_value=1,max_value=120,value=25)
gender=st.selectbox("select gender:",['Male','Female','other'])
age=st.number_input("select age:",min_value=1,max_value=120,value=25)
symptoms_input=st.text_area("select symptoms:","fever,back pain,shortness of breath")

if st.button("predict disease"):
    if symptoms_input.strip()!="":
     cleaned_symptoms=clean_text(symptoms_input)
     vectorized_text=tfidf.transform([cleaned_symptoms]).toarray()
     prediction_numeric=healt.predict(vectorized_text)
     predicted_disease=disease.inverse_transform(prediction_numeric)[0]
     st.success(f"Predict Disease:{predicted_disease}")
    else:
        st.warning("symptoms add kara")
        
     
     
     
    



