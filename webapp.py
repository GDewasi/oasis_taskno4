import numpy as np
import pickle
import streamlit as st
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

# loading our model file (model.sav) into this program
load_model_mnb=pickle.load(open('model.sav','rb'))

# loading tfidf vectorizer object file for text encoding 
load_tfidf=pickle.load(open('tfidf_vectorizer.sav','rb'))

def transform_sms(message):
    
    # to convert all characters in lower case
    message=message.lower()
    
    # to break list into words
    message=nltk.word_tokenize(message)
    # to remove special symbals
    
    temp=[]
    for i in message:
        if i.isalnum():
            temp.append(i)

    message=temp[:]   # create clone of y
    temp.clear()
    
    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            temp.append(i)
    
    message=temp[:]
    temp.clear()
    
    for i in message:
        temp.append(ps.stem(i))
    
    return " ".join(temp)

# main() for web app interface and input tasks
def main():
    
    # for wide look 
    st.set_page_config(layout="wide")


    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.pexels.com/photos/167699/pexels-photo-167699.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    

    html_temp="""

    <div style="background-color:DarkBlue;padding:10xp">
    <h2 style="color:white;text-align:center;">SMS Spam Detection Model </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)


    input_sms=st.text_area("**Enter the message for testing**")

    # sms transformation 
    input_sms=transform_sms(input_sms)

    # tfidf vectorizzer
    input_sms=load_tfidf.transform([input_sms])

    # prediction using model
    pred=load_model_mnb.predict(input_sms)[0]

    # button for prediction
    if st.button("Predict"):
        if pred == 1:
            st.success("**Spam sms **💬 ")
        else:
            st.success("**Not Spam sms **💬")


if __name__ == '__main__':
    main()
