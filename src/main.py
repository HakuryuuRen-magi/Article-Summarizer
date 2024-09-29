import re
import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
import string
from string import punctuation
from nltk.corpus import stopwords
from statistics import mean
from heapq import nlargest
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from rouge_score import rouge_scorer, scoring

stop_words = set(stopwords.words('english'))
punctuation = punctuation + '\n' + '—' + '“' + ',' + '”' + '‘' + '-' + '’'

# https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions_dict = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "doesn’t": "does not",
    "don't": "do not",
    "don’t": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y’all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have",
    "ain’t": "am not",
    "aren’t": "are not",
    "can’t": "cannot",
    "can’t’ve": "cannot have",
    "’cause": "because",
    "could’ve": "could have",
    "couldn’t": "could not",
    "couldn’t’ve": "could not have",
    "didn’t": "did not",
    "doesn’t": "does not",
    "don’t": "do not",
    "don’t": "do not",
    "hadn’t": "had not",
    "hadn’t’ve": "had not have",
    "hasn’t": "has not",
    "haven’t": "have not",
    "he’d": "he had",
    "he’d’ve": "he would have",
    "he’ll": "he will",
    "he’ll’ve": "he will have",
    "he’s": "he is",
    "how’d": "how did",
    "how’d’y": "how do you",
    "how’ll": "how will",
    "how’s": "how is",
    "i’d": "i would",
    "i’d’ve": "i would have",
    "i’ll": "i will",
    "i’ll’ve": "i will have",
    "i’m": "i am",
    "i’ve": "i have",
    "isn’t": "is not",
    "it’d": "it would",
    "it’d’ve": "it would have",
    "it’ll": "it will",
    "it’ll’ve": "it will have",
    "it’s": "it is",
    "let’s": "let us",
    "ma’am": "madam",
    "mayn’t": "may not",
    "might’ve": "might have",
    "mightn’t": "might not",
    "mightn’t’ve": "might not have",
    "must’ve": "must have",
    "mustn’t": "must not",
    "mustn’t’ve": "must not have",
    "needn’t": "need not",
    "needn’t’ve": "need not have",
    "o’clock": "of the clock",
    "oughtn’t": "ought not",
    "oughtn’t’ve": "ought not have",
    "shan’t": "shall not",
    "sha’n’t": "shall not",
    "shan’t’ve": "shall not have",
    "she’d": "she would",
    "she’d’ve": "she would have",
    "she’ll": "she will",
    "she’ll’ve": "she will have",
    "she’s": "she is",
    "should’ve": "should have",
    "shouldn’t": "should not",
    "shouldn’t’ve": "should not have",
    "so’ve": "so have",
    "so’s": "so is",
    "that’d": "that would",
    "that’d’ve": "that would have",
    "that’s": "that is",
    "there’d": "there would",
    "there’d’ve": "there would have",
    "there’s": "there is",
    "they’d": "they would",
    "they’d’ve": "they would have",
    "they’ll": "they will",
    "they’ll’ve": "they will have",
    "they’re": "they are",
    "they’ve": "they have",
    "to’ve": "to have",
    "wasn’t": "was not",
    "we’d": "we would",
    "we’d’ve": "we would have",
    "we’ll": "we will",
    "we’ll’ve": "we will have",
    "we’re": "we are",
    "we’ve": "we have",
    "weren’t": "were not",
    "what’ll": "what will",
    "what’ll’ve": "what will have",
    "what’re": "what are",
    "what’s": "what is",
    "what’ve": "what have",
    "when’s": "when is",
    "when’ve": "when have",
    "where’d": "where did",
    "where’s": "where is",
    "where’ve": "where have",
    "who’ll": "who will",
    "who’ll’ve": "who will have",
    "who’s": "who is",
    "who’ve": "who have",
    "why’s": "why is",
    "why’ve": "why have",
    "will’ve": "will have",
    "won’t": "will not",
    "won’t’ve": "will not have",
    "would’ve": "would have",
    "wouldn’t": "would not",
    "wouldn’t’ve": "would not have",
    "y’all": "you all",
    "y’all": "you all",
    "y’all’d": "you all would",
    "y’all’d’ve": "you all would have",
    "y’all’re": "you all are",
    "y’all’ve": "you all have",
    "you’d": "you would",
    "you’d’ve": "you would have",
    "you’ll": "you will",
    "you’ll’ve": "you will have",
    "you’re": "you are",
    "you’re": "you are",
    "you’ve": "you have",
}

contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))

# Function expand the contractions if there's any
def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)

# Function to preprocess the articles
def preprocessing(article):
    global article_sent
    
    # Converting to lowercase
    article = article.str.lower()
    
    # Removing the '\xa0'
    article = article.apply(lambda x: x.replace("\xa0", " "))
    
    # Removing the contractions
    article = article.apply(lambda x: expand_contractions(x))
    
    # Stripping the possessives
    article = article.apply(lambda x: x.replace("'s", ''))
    article = article.apply(lambda x: x.replace('’s', ''))
    article = article.apply(lambda x: x.replace("\'s", ''))
    article = article.apply(lambda x: x.replace("\’s", ''))
    
    # Removing the Trailing and leading whitespace and double spaces
    article = article.apply(lambda x: re.sub(' +', ' ',x))

     # Copying the article for the sentence tokenization
    article_sent = article.copy()
    
    # Removing punctuations from the article
    article = article.apply(lambda x: ''.join(word for word in x if word not in punctuation))
    
    # Removing the Trailing and leading whitespace and double spaces again as removing punctuation might
    # Lead to a white space
    article = article.apply(lambda x: re.sub(' +', ' ',x))
    
    # Removing the Stopwords
    article = article.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    
    return article

# Function to normalize the word frequency which is used in the function word_frequency
def normalize(li_word):
    global normalized_freq
    normalized_freq = []
    for dictionary in li_word:
        max_frequency = max(dictionary.values())
        for word in dictionary.keys():
            dictionary[word] = dictionary[word]/max_frequency
        normalized_freq.append(dictionary)
    return normalized_freq

# Model for Article Summarizer based on TextRank Algorithmn
# Function to calculate the word frequency
def word_frequency(article_word):
    word_frequency = {}
    li_word = []
    for sentence in article_word:
        for word in word_tokenize(sentence):
            if word not in word_frequency.keys():
                word_frequency[word] = 1
            else:
                word_frequency[word] += 1
        li_word.append(word_frequency)
        word_frequency = {}
    normalized_freq = normalize(li_word)
    return normalized_freq

# Function to score the sentence which is called in the function sent_token
def sentence_score(li, normalized_freq):
    sentence_score_list = []
    for list_, dictionary in zip(li, normalized_freq):
        sentence_score = {}
        for sent in list_:
            for word in word_tokenize(sent):
                if word in dictionary.keys():
                    if sent not in sentence_score.keys():
                        sentence_score[sent] = dictionary[word]
                    else:
                        sentence_score[sent] += dictionary[word]
        sentence_score_list.append(sentence_score)
    return sentence_score_list

# Function to tokenize the sentence
def sent_token(article_sent):
    sentence_list = []
    for sent in article_sent:
        token = sent_tokenize(sent)
        sent_token = []
        for sentence in token:
            token_2 = ''.join(word for word in sentence if word not in punctuation)
            token_2 = re.sub(' +', ' ', token_2)
            sent_token.append(token_2)
        sentence_list.append(sent_token)
    return sentence_list

# Function which generates the summary of the articles (This uses the specified percentage of sentences with the highest score)
def summary(sentence_scores, select_percentage=0.25):
    summary_list = []
    for summ in sentence_scores:
        select_length = int(len(summ) * select_percentage)
        summary_ = nlargest(select_length, summ, key=summ.get)
        summary_list.append(". ".join(summary_))
    return summary_list

# Functions to change the article string (if passed) to generate a pandas series
def make_series(art):
    data_dict = {'article': [art]}
    dataframe = pd.DataFrame(data_dict)['article']
    return dataframe

# Function which is to be called to generate the summary which in further calls other functions altogether
def article_summarize(article_text):
    # Create a Pandas Series from the input article text
    article_series = make_series(article_text)

    # Preprocess the article text
    preprocessed_article = preprocessing(article_series)

    # Tokenize the sentences
    tokenized_sentences = sent_token(article_sent)

    # Calculate word frequencies
    word_freqs = word_frequency(preprocessed_article)

    # Score sentences
    sentence_scores = sentence_score(tokenized_sentences, word_freqs)

    # Generate summary (using the top 25% of scored sentences)
    summarized_article = summary(sentence_scores)

    # Return the first item in the list since you're only summarizing one article
    return summarized_article[0] if summarized_article else ""
