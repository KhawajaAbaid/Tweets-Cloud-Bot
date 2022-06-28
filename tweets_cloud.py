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
import tweepy
import configparser
import configparser
from collections import Counter
import logging


logging.basicConfig(filename="logs/tweets_cloud.log", level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting tweets_cloud.py")

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

api_v2 = tweepy.Client(consumer_key, consumer_secret, access_token, access_secret)

# We need API v1 to upload media since twitter's api v2 doesn't
# support uploading media yet.
auth_v1 = tweepy.OAuthHandler(consumer_key, consumer_secret,
                                access_token, access_secret)
api_v1 = tweepy.API(auth=auth_v1)


def get_last_seen_tweet_id():
    """Gets the last seen tweet id."""
    
    logging.info("Retrieing last seen tweet ID")
    with open("last_seen_tweet_id.txt", "r") as f:
        return int(f.read().strip())

def store_last_seen_tweet_id(last_seen_tweet_id:int):
    """Stores the last seen tweet id in a file.
    """
    logging.info("Storing last seen tweet ID")
    last_seen_tweet_id = last_seen_tweet_id
    with open("last_seen_tweet_id.txt", "w") as f:
        f.write(str(last_seen_tweet_id))
    return



def get_mentions():
    """Gets the mentions of the bot.
    Returns:
        A list of mentions, along with meta data about users.
    """
    logging.info("Retrieving mentions")
    last_seen_tweet_id = get_last_seen_tweet_id()
    # the twitter id of the TweetsCloudBot
    bot_id = "1541369501333209094"
    mentions = api_v2.get_users_mentions(user_id=bot_id, since_id=last_seen_tweet_id,
                                    expansions="author_id", user_fields=["username"])
    mentions = mentions.data
    users_metadata = mentions.includes["users"]
    return mentions, users_metadata

def fetch_tweets(user_id:str):
    """Fetches tweets from a given username.
    Args:
        user_id: The id of of the user to fetch tweets from.
    Returns:
        A list of tweets.
    """
    logging.info("Fetching tweets")
    # user = api_v2.get_user(username)
    tweets = api_v2.get_users_tweets(user_id=user.data.id, max_results=100)
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
    background_color:str='black',
    user_id:str='123'):
    """Generates a tweets cloud from a list of words extracted from tweets.
    Args:
        words: A list of words to generate a tweet cloud from.
        mode: The mode of the tweet cloud. Like one like Spotitfy Wrapped,
        which is default or a skech book one, specified by passing "sketch"
        border: Whether or not to include a border.
        background_color: The background color of the tweet cloud.
        user_id: The id of the user, used for the filename of the tweet cloud.
    Returns:
        A tweets cloud.
    """
    # A note on privacy: The sole purpose of using user_id in filename is to
    # 1. Uniquely save and retrieve the word cloud image
    # 2. To prevent the word cloud image from being shared with others
    # 3. In case we're under multiple requests, we don't wanna overwrite the file
    # while we're uploading it. Like we may encounter an error while uploading
    # and we want to retry but our file has already been overwritten by the script.
    # Hope it makes sense. All the generated images will be deleted after some time.

    mask = np.array(Image.open(BASE_PATH / "data/twitter_mask.png"))
    gradient = ImageColorGenerator(np.array(Image.open(BASE_PATH / "data/twitter_gradient.png")))
    freq = Counter(words)


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


def reply(tweet_id:str, user_id:str):
    """Replies to a tweet.
    Args:
        tweet_id: The id of the tweet to reply to.
    """
    reply_text = "Hi, here's your requested Tweet Cloud!"
    tweet_cloud_img = f"tmp/tweet_cloud_{user_id}"
    api_v2.create_tweet()
    return

def bot_handler():
    """Handles the bot.
    """
    while True:
        mentions, users_metadata = get_mentions()
        if len(mentions) > 0:
            # We reverse the mentions the reply to the early ones first
            # And we enumerate the mentions to get the index to retrieve
            # username from meta data since both lists have one to one
            # correspondence
            for mention, mention_num in enumerate(reversed(mentions), start=1):
                tweet_id = mention.id
                user_id = mention.author_id
                # We use negative index here since we reversed the mentions list
                username = users_metadata[-mention_num].username
                tweets = fetch_tweets(user_id)
                words = preprocess_and_tokenize_tweets(tweets)
                generate_tweets_cloud(words, mode="sketch", user_id=user_id)
                store_last_seen_tweet_id(mention.id)
        time.sleep(30)