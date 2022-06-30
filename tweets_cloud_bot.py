import matplotlib.pyplot as plt
from collections import Counter
import logging
import time
from PIL import Image
import pathlib
import nltk
import string
import numpy as np
from wordcloud import WordCloud, ImageColorGenerator

from twitter_bot import TwitterBot

logging.basicConfig(filename="logs/tweets_cloud_bot_v2.log", level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class TweetsCloudBot(TwitterBot):
    def __init__(self):
        logging.info('tweets_cloud_bot_v2.py initiated')

        bot_id = "1541369501333209094"
        self.BASE_PATH = pathlib.Path.cwd()
        config_file_path = self.BASE_PATH / "classified_configs.ini"
        
        super().__init__(bot_id=bot_id, config_file_path=config_file_path.as_posix())

        self.required_input = ["make tweets cloud", "make tweet cloud"]
        self.input_params_dict = {'mode': ['default', 'sketch'],
                                  'background_color': ['white', 'black'],
                                  'border': ['no border']}
    
    def generate_tweets_cloud(
            self,
            words: list,
            mode: str = 'default',
            border: bool = True,
            background_color: str = 'black',
            user_id: str = None):
        """Generates a tweets cloud from a list of words extracted from tweets.
        Args:
            words: A list of words to generate a tweet cloud from.
            mode: The mode of the tweet cloud. Like one like Spotitfy Wrapped,
            which is default or a skech book one, specified by passing "sketch"
            border: Whether or not to include a border.
            background_color: The background color of the tweet cloud, white or black.
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

        mask = np.array(Image.open(self.BASE_PATH / "data/twitter_mask.png"))
        gradient = ImageColorGenerator(np.array(Image.open(self.BASE_PATH / "data/twitter_gradient_default.png")))
        freq = Counter(words)

        background_color = background_color

        if mode == "default":
            wordcloud_generator = WordCloud(background_color=background_color,
                                            mask=mask,
                                            width=1500,
                                            height=1500,
                                            color_func=gradient)
            border_img = np.array(Image.open(self.BASE_PATH / "data/twitter_border_default.png"))
        elif mode == "sketch":
            font_path = self.BASE_PATH / "fonts/CabinSketch-Bold.ttf"
            wordcloud_generator = WordCloud(background_color=background_color,
                                            mask=mask,
                                            width=1200,
                                            height=1200,
                                            font_path=font_path.as_posix())
            border_img = np.array(Image.open(self.BASE_PATH / "data/twitter_border_sketch.png"))

        tweets_cloud = wordcloud_generator.generate_from_frequencies(freq)

        plt.figure(figsize=(5, 5))
        plt.imshow(tweets_cloud, interpolation='bilinear')

        if border:
            plt.imshow(border_img)

        plt.axis("off")
        if background_color == "black":
            text_color = "white"
        else:
            text_color = "black"
        plt.text(370, 1150, "Generated with ❤️ by @TweetsCloudBot", fontsize=5, color=text_color)
        plt.savefig(f"tmp/tweetscloud_{mode}_{user_id}.png", dpi=300, bbox_inches="tight")


    def reply_with_tweetcloud(self, tweet_id: str, user_id: str = None,
                              user_screen_name: str = "Elon Musk",
                              username: str = "@ElonMusk",
                              cloud_mode: str = "default"):
        """Replies to a tweet.
        Args:
            tweet_id: The id of the tweet to reply to.
        """
        reply_text = f"Hey @{username}, your Tweets Cloud ☁ is ready!"
        tweet_cloud_img = f"tmp/tweetscloud_{cloud_mode}_{user_id}.png"
        # note the use of api_v1 to upload media since v2 doesn't support media upload
        # as of now
        media = self.api_v1.media_upload(tweet_cloud_img)
        # now we tweet using api_v2, though we cloud with v1 as well.
        self.api_v2.create_tweet(text=reply_text,
                            media_ids=[media.media_id],
                            in_reply_to_tweet_id=tweet_id)


    def bot_handler(self):
        """Handles the bot.
            """
        while True:
            mentions, users_metadata = self.get_mentions()
            if len(mentions) > 0:
                logging.info("GOT NEW MENTIONS")
                # We reverse the mentions the reply to the early ones first
                # And we enumerate the mentions to get the index to retrieve
                # username from meta data since both lists have one to one
                # correspondence
                # We use reverse user_metadeta here since we reverse the mentions list
                users_metadata = list(reversed(users_metadata))
                for mention_num, mention in enumerate(reversed(mentions), start=1):
                    tweet_id = mention.id
                    user_id = mention.author_id

                    # validate input
                    if not self.validate_input(required_input= self.required_input,
                                               tweet_text=mention.text):
                        self.store_last_seen_tweet_id(mention.id)
                        logging.critical("Invalid Input. Moving on.")
                        continue

                    if mention_num > 1 and user_id == users_metadata[mention_num - 2].id:
                        # if the same user mentions the bot multiple times consecutively
                        # the meta data only contains one entry for the user while mentions
                        # contains all instances of mentions, hence the one to one correspondence
                        # b/w the two lists breaks. So we check if user_id is the same as previous mention
                        # if so we pass keep the same credentials otherwise we update them
                        pass
                    else:
                        username = users_metadata[mention_num - 1].username
                        screen_name = users_metadata[mention_num - 1].name
                    # if user has reached their daily limit, reply with appropriate
                    # message and move on to the next one
                    if not self.validate_user(user_id):
                        self.reply_with_limit_reached(tweet_id, screen_name)
                        self.store_last_seen_tweet_id(mention.id)
                        logging.critical(f"{username} has reached their daily limit. Moving on.")
                        continue
                    # Otherwise we fetch tweets and proceed with business as usual
                    tweets = self.fetch_tweets(user_id)
                    words = self.preprocess_and_tokenize_tweets(tweets)
                    params = self.get_params_from_tweet(tweet_text=mention.text,
                                                        params_dict=self.input_params_dict)
                    try:
                        exists = params['mode']
                    except KeyError:
                        params['mode'] = "default"
                    self.generate_tweets_cloud(words, user_id=user_id, **params)
                    self.store_last_seen_tweet_id(mention.id)
                    self.reply_with_tweetcloud(tweet_id, user_id,
                                          user_screen_name=screen_name,
                                          username=username,
                                          cloud_mode=params["mode"])
                    self.update_validation_data(user_id)
                    logging.info(f"User {username} has been replied.")
            logging.info("No mentions, going to sleep for the next 30 seconds.")
            time.sleep(30)

if __name__ == "__main__":
    tweets_cloud_bot = TweetsCloudBot()
    tweets_cloud_bot.bot_handler()