"""tweets_cloud.py: Generates a word cloud from a list of tweets."""

import pandas as pd
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import pathlib
import time
import datetime
import string
import nltk
from pytwitter import Api
import configparser
import configparser
from collections import Counter


# Base path
BASE_PATH = pathlib.Path.cwd()

# Mask for word cloud
mask = np.array(Image.open(BASE_PATH / "data/twitter_mask.png"))


# Let's define our stop words here
# If you're not familiar with stop words, they are the most commonly ccuring
# words in texts, like the, a, and, of, etc.
stop_words = nltk.corpus.stopwords.words('english')
stop_words.extend(list(string.punctuation))

# Let's get out twitter API up and running
# Note that I'm importing my twitter API credentials from a config file
# Obviously the conifg file isn't in the repo, so to make this script work
# you'll need to provide your own credentials.
config = configparser.ConfigParser()
config.read("classified_configs.ini")
config['twitter-app-data']['bearer']
consumer_key = config['twitter-app-data']['consumer_key']
consumer_secret = config['twitter-app-data']['consumer_secret']
access_token = config['twitter-bot-data']['access_token']
access_secret = config['twitter-bot-data']['access_token_secret']

api = Api(consumer_key, consumer_secret, access_token, access_secret)

def fetch_tweets(username:str):
    """Fetches tweets from a given username.
    Args:
        username: The username of the user to fetch tweets from.
    Returns:
        A list of tweets.
    """
    user = api.get_user(username)
    tweets = api.get_timelines(user_id=user.data.id, max_results=100)
    tweets = tweets.data
    return tweets

def preprocess_and_tokenize_tweets(tweets:list):
    """Preprocesses and tokenizes tweets.
    Args:
        tweets: A list of tweets where each tweet is wrapped in
        object of type pytwitter.models.tweet.Tweet.
    Returns:
        A list of words or tokens.
    """
    for tweet in tweets:
        tweet_text = tweet.text
        words = nltk.tokenize.casual.casual_tokenize(tweet_text,
                                    preserve_case=False,
                                    reduce_len=True,
                                    strip_handles=True)
        words = [word for word in words if word not in stop_words]
    return words

def generate_tweets_cloud(
    words:list,
    mode:str='default',
    border:bool=True,
    background:str='black'):
    """Generates a tweets cloud from a list of words extracted from tweets.
    Args:
        words: A list of words to generate a word cloud from.
        mode: The mode of the word cloud. Like one like Spotitfy Wrapped,
        which is default or a skech book one, specified by passing "sketch"
        border: Whether or not to include a border.
    Returns:
        A tweets cloud.
    """
    freq = Counter(words)

    if mode=="default":
        worcloud_generator = WordCloud()
