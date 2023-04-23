import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob

# Read csv file
df = pd.read_csv('_apple_iphone_11_reviews.csv')
# print(df)

# drop index column
df = df.drop(['index'], axis=1)
# print(df['review_text'])

# Using spacy to preprocess text
sp = spacy.load('en_core_web_sm')

# Preprocessing which involves - lowercasing, removal of stopwords punctuation digits, Lemmatization and combining lemmas, perform sentiment analysis
def preprocess_text(txt):
    # Lowercase the entire text
    doc = sp(str(txt).lower())
    # print(doc)

    # Tokenizing
    # tokens = [token for token in doc]

    # Tokenizing removing punctuations,stop words and digits from the doc
    tokens = [token for token in doc if not token.is_digit and not token.is_punct and not token.is_stop]
    print("tokens =")
    print(tokens)

    # Lemmatization
    # finding the lemma of the tokens
    tokens = [token.lemma_ for token in tokens]
    print("Lemma =")
    print(tokens)
    print()


    return ' '.join(tokens)


df['clean_text'] = df['review_text'].apply(preprocess_text)
print(df['clean_text'])

# Performing Sentiment Analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['polarity'] = df['clean_text'].apply(get_sentiment)
print(df['polarity'])

# Visualization
colors = ['green', 'red']
plt.hist(df['polarity'], bins=20)
plt.xlabel('Sentiment Polarity')
plt.ylabel('Frequency')
plt.title('Sentiment Analysis for Iphone')
plt.show()
