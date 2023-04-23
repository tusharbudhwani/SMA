import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

df = pd.read_csv('sma_dataset.csv')

sp = spacy.load('en_core_web_sm')

def preprocessing_text(text):
    # Lower
    doc = sp(text.lower())

    # tokenizing
    tokens = [token for token in doc if not token.is_stop and not token.is_punct and not token.is_digit and not token.is_bracket]

    # lemma
    tokens = [token.lemma_ for token in tokens]

    return ' '.join(tokens)

df['processed_text'] = df['Tweet'].apply(preprocessing_text)

print(df['processed_text'])
tweets = []

for tweet in df['processed_text']:
    tokens = []
    for word in word_tokenize(tweet):
        if True and not word.startswith('http') and not word.startswith('@') and not word.startswith('$') and not word.startswith('/') and not word.startswith(':'):
            tokens.append(word)
    tweets.append(tokens)
print(tweets)

# Form a dictionary
dictionary = Dictionary(tweets)

# Bag of words
corpus = [dictionary.doc2bow(tweet) for tweet in tweets]

# train
lda_model = LdaModel(corpus=corpus,num_topics=10,id2word=dictionary, passes=10)

for topic in lda_model.show_topics():
    print(topic)

print('Perplexity = ',lda_model.log_perplexity(corpus))