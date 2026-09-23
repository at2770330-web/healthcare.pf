import tokenize
import re
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('Healthcare.csv')
print(df.head)
print(df.info)
print(df.sample)
print(df.isnull().sum())
print(df.describe)
ps=PorterStemmer()
stop_words=set(stopwords.words("english"))
print(stop_words)

try:
 def clean_text(text):
    words=str(text).lower().split()
    cleaned=[ps.stem(w)for w in words if w not in stop_words]
    return "".join(cleaned)
 df["cleaned_Symptons"]=df['Symptoms'].apply(clean_text) 
except  Exception as e:
   print("as Error:",e)
   
#from tfidf vectorizer applyed
#tfidf=TfidfVectorizer()
#X=tfidf.fit_transform(df['cleaned_Symptons']).toarray()

#from label encording from the labelencording
Disease=LabelEncoder()
Gender=LabelEncoder()
df['Disease']=Disease.fit_transform(df[['Disease']])
df['Gender']=Gender.fit_transform(df['Gender'])
print(df['Disease'])
print(df['Gender'])

#from tfidf vectorizer applyed
tfidf=TfidfVectorizer()
X=tfidf.fit_transform(df['cleaned_Symptons']).toarray()
y=df['Disease']
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
reduced_data=pca.fit_transform(X)
plt.scatter(reduced_data[:,0],reduced_data[:,1],marker='*',color='red')
print(plt.show())

#from sklearn train test data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_train.shape)
print(y_train.shape)


#from sklearn SVM
model=SVC(kernel='rbf',C=1.0,gamma='scale')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
acc=accuracy_score(y_pred,y_test)
cmm=confusion_matrix(y_pred,y_test)
print(acc)
print(cmm)

#from sklearn logistic regresssion
model1=LogisticRegression(max_iter=1000)
model1.fit(X_train,y_train)
y_pred1=model1.predict(X_test)
acc1=accuracy_score(y_pred1,y_test)
cmm1=confusion_matrix(y_pred1,y_test)
print(acc1)
print(cmm1)

#fromm sklearn random state
model2=RandomForestClassifier(n_estimators=1000,random_state=42)
model2.fit(X_train,y_train)
y_pred2=model2.predict(X_test)
acc2=accuracy_score(y_pred2,y_test)
cmm2=confusion_matrix(y_pred2,y_test) 
print(acc2)
print(cmm2)  
   
with open('healthcare.pkl','wb')as file:
   pickle.dump(model,file)
with open('tfidf.pkl','wb')as file:
   pickle.dump(tfidf,file)
with open('disease_encoder.pkl','wb')as file:
   pickle.dump(Disease,file)   
      
   print('pickle modell as saved')   
   
   
   
   
   
    
       
      
      
      
      
      
         
    
    

