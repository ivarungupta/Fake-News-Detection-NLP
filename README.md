# Fake News Detection Using NLP

## Overview
This project aims to detect fake news articles using machine learning models and natural language processing (NLP) techniques. The detection is based on textual data, where the model classifies news articles as either "fake" or "real." The project explores multiple machine learning algorithms, including Logistic Regression, Support Vector Machines (SVM), and Naive Bayes, combined with text preprocessing techniques such as TF-IDF vectorization.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Description
Fake news detection is a crucial problem in today’s digital age. Fake news can spread quickly through social media and news websites, and it often has significant societal impacts. In this project, we use NLP to extract features from news text and machine learning models to classify them as fake or real.

### Key Objectives:
1. Preprocess textual data from news articles.
2. Build and evaluate machine learning models for fake news classification.
3. Compare performance across various models.

## Dataset
The dataset used in this project contains a collection of labeled news articles, indicating whether they are real or fake. The dataset is divided into training and testing sets to evaluate model performance.

You can download the dataset from [Kaggle's Fake News Dataset](https://www.kaggle.com/c/fake-news/data).

### Dataset Features:
- **Text**: The body of the news article.
- **Label**: A binary indicator (1 for fake news, 0 for real news).

## Preprocessing
The preprocessing steps include:
- **Tokenization**: Splitting the text into individual tokens (words).
- **Stopword Removal**: Removing common words like "the", "is", "in" that do not add significant value to classification.
- **Stemming/Lemmatization**: Reducing words to their root forms (e.g., "running" becomes "run").
- **TF-IDF Vectorization**: Converting the textual data into numerical features based on word frequency and importance.

### Code Example:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['text']).toarray()
```
