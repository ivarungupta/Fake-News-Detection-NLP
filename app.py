import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(page_title='Fake News Detector', page_icon='📰')

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)

# website
st.title('📰 Fake News Detector')
st.subheader('Is your news real or fake? Enter an article to find out!')
st.image('image.jpg', use_column_width=True)

st.markdown("""
<style>
input {
    height: 50px;
    font-size: 16px;
    padding: 10px;
}
body {
    background-color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)

input_text = st.text_input('Enter news Article')

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.markdown('<h2 style="color: red;">🚫 The News is Fake!</h2>', unsafe_allow_html=True)
    else:
        st.markdown('<h2 style="color: green;">✅ The News is Real!</h2>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("Created by Varun Gupta (BTECH/10029/22)")


