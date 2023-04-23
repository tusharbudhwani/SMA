import pandas as pd
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt

df = pd.read_csv('sma_dataset.csv')
print(df)

df = df.dropna()

sp = spacy.load('en_core_web_sm')

# Preprocessing
def preprocessing_text(text):
    #Lowercasing
    doc = sp(text.lower())

    # TOkenizing and emoving stopwords, puncts and digits
    tokens = [token for token in doc if not token.is_stop and not token.is_digit and not token.is_punct]

    # finding lemmas
    tokens = [token.lemma_ for token in tokens]

    return ' '.join(tokens)

df['processed_text'] = df['Tweet'].apply(preprocessing_text)
print(df['processed_text'])

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

df['polarity'] = df['processed_text'].apply(get_sentiment)

sentiment_list = []
pos = 0
neg = 0
neu = 0

for num in df['polarity']:
    if num > 0:
        pos = pos + 1
        sentiment_list.append('Positive')

    elif num < 0:
        neg = neg + 1
        sentiment_list.append('Negative')

    else:
        neu = neu + 1
        sentiment_list.append('Neutral')

df['sentiment'] = sentiment_list

print("Total positive opinions = ", pos)
print("Total negative opinions = ", neg)
print("Total neutral opinions = ", neu)

print(df[['processed_text', 'sentiment']])

#visualize

plt.hist(df['polarity'], bins=20)
plt.xlabel('frequency')
plt.ylabel('sentiments')
plt.title('haha')
plt.show()