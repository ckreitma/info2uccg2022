# https://towardsdatascience.com/how-to-access-data-from-the-twitter-api-using-tweepy-python-e2d9e4d54978
import configparser
import tweepy
from sentiment_analysis_spanish import sentiment_analysis

sentiment = sentiment_analysis.SentimentAnalysisSpanish()

# read config
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

# authenticate
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
q = 'Nadal Rolland Garros -is_retweet -RT -is:retweet lang:es'
for i, tweet in enumerate(tweepy.Cursor(api.search_tweets, q=q, lang='es').items(15)):
    texto = tweet.text
    print("================================================================================")
    print(f'Tweet Número: {i} ===>  ' + texto + ' Sentimiento:' + str(sentiment.sentiment(texto)))
