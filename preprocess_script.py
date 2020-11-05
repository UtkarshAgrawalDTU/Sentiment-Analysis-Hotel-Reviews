import nltk
from nltk.corpus import wordnet
import joblib
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def clean_text(text):
    text = str(text).lower()
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    text = [word for word in text if not any(c.isdigit() for c in word)]
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    text = [t for t in text if len(t) > 0]
    pos_tags = pos_tag(text)
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return(text)


def preprocess(reviews_df):
    
    reviews_df["review_clean"] = reviews_df["review"]
    reviews_df["review_clean"] = reviews_df["review_clean"].apply(lambda x: clean_text(x))
    sid = SentimentIntensityAnalyzer()
    reviews_df["sentiments"] = reviews_df["review"]
    reviews_df["sentiments"] = reviews_df["sentiments"].apply(lambda x: sid.polarity_scores(str(x)))
    reviews_df["neg"] = reviews_df["sentiments"]
    reviews_df["neg"] = reviews_df["neg"].apply(lambda x: x['neg'])
    reviews_df["pos"] = reviews_df["sentiments"]
    reviews_df["pos"] = reviews_df["pos"].apply(lambda x: x['pos'])
    reviews_df["neu"] = reviews_df["sentiments"]
    reviews_df["neu"] = reviews_df["neu"].apply(lambda x: x['neu'])
    reviews_df["compound"] = reviews_df["sentiments"]
    reviews_df["compound"] = reviews_df["compound"].apply(lambda x: x['compound'])
    reviews_df.drop('sentiments', axis = 1, inplace = True)
    reviews_df["nb_chars"] = reviews_df["review"].apply(lambda x: len(str(x)))
    reviews_df["nb_words"] = reviews_df["review"].apply(lambda x: len(str(x).split(" ")))
    tfidf = joblib.load('tfidf.pkl')
    tfidf_result = tfidf.transform(reviews_df["review_clean"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = reviews_df.index
    reviews_df = pd.concat([reviews_df, tfidf_df], axis=1)
    return reviews_df