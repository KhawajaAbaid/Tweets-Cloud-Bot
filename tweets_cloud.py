"""tweets_cloud.py: Generates a word cloud from a list of tweets."""

from turtle import back
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

def get_last_seen_tweet_id():
    """Gets the last seen tweet id."""
    
    with open("last_seen_tweet_id.txt", "r") as f:
        return int(f.read().strip())

def store_last_seen_tweet_id(last_seen_tweet_id:int):
    """Stores the last seen tweet id in a file.
    """
    last_seen_tweet_id = last_seen_tweet_id
    with open("last_seen_tweet_id.txt", "w") as f:
        f.write(str(last_seen_tweet_id))
    return

def get_mentions():
    """Gets the mentions of the bot.
    Returns:
        A list of mentions.
    """
    last_seen_tweet_id = get_last_seen_tweet_id()
    bot_id = "1541369501333209094"
    mentions = api.get_mentions(user_id=bot_id, since_id=last_seen_tweet_id)
    mentions = mentions.data
    return mentions

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
    background_color:str='black'):
    """Generates a tweets cloud from a list of words extracted from tweets.
    Args:
        words: A list of words to generate a tweet cloud from.
        mode: The mode of the tweet cloud. Like one like Spotitfy Wrapped,
        which is default or a skech book one, specified by passing "sketch"
        border: Whether or not to include a border.
        background_color: The background color of the tweet cloud.
    Returns:
        A tweets cloud.
    """

    mask = np.array(Image.open(BASE_PATH / "data/twitter_mask.png"))
    gradient = ImageColorGenerator(np.array(Image.open(BASE_PATH / "data/twitter_gradient.png")))
    freq = Counter(words)

    # generates a random integer which we use when saving the word cloud
    # we're not using one static name like tmp.png because in case we're
    # under multiple requests, we don't wanna overwrite the file as this
    # script is concened with generating word cloud but another script is
    # concerned with handling bot stuff like uploading.
    img_id = np.random.randint(0, 100)

    background_color = background_color

    if mode=="default":
        wordcloud_generator = WordCloud(background_color=background_color,
                                        mask=mask,
                                        width=1200, 
                                        height=1200,
                                        color_func=gradient)
    elif mode=="sketch":
        font_path = BASE_PATH / "fonts/CabinSketch-Bold.ttf"
        wordcloud_generator = WordCloud(background_color=background_color, 
                                        mask=mask,
                                        width=1200,
                                        height=1200,
                                        font_path=font_path.as_posix())
    
    tweets_cloud = wordcloud_generator.generate_from_frequencies(freq) 

    plt.figure(figsize=(4,4))
    plt.imshow(tweets_cloud, interpolation='bilinear')
    
    if border:
        plt.imshow(border)
    
    plt.axis("off")
    plt.text(630, 1150, "Generated with ❤️ by @TweetsCloudBot", fontsize=5, color="white")
    plt.savefig("tmp/tweetscloud_sketch.png", dpi=300, bbox_inches="tight")



def bot_handler():
    """Handles the bot.
    """
    while True:
        mentions = get_mentions()
        if len(mentions) > 0:
            for mention in mentions:
                tweet_id = mention.id
                user_id = api.get_twee
                tweets = fetch_tweets(username)
                words = preprocess_and_tokenize_tweets(tweets)
                generate_tweets_cloud(words, mode="sketch")
                api.upload_media(BASE_PATH / "tmp/tweetscloud_sketch.png")
                api.update_status(f"@{username}", media_ids=[api.media_id])
                store_last_seen_tweet_id(mention.id)
        time.sleep(30)