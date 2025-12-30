import praw
import pandas as pd
from dotenv import load_dotenv
import os
import time
import ahocorasick
from logging import Logger, FileHandler, Formatter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

### Helper functions ###


def pattern_matching(text, keywords):
    # Build Aho-Corasick Trie
    trie = ahocorasick.Automaton()
    for idx, keyword in enumerate(keywords):
        trie.add_word(keyword, (idx, keyword))
    trie.make_automaton()

    matches = list(trie.iter(text))
    if matches:
        return True
    else:
        return False


def clean_text(text):
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    #remove all characters that are not ascii
    text = ''.join([i if ord(i) < 128 else ' ' for i in text])
    #remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    #remove all stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    #lemmatize the words
    text = ' '.join([wordnet.lemmatize(word) for word in text.split()])
    
    return text

### Scraper function ###

def scrape_reddit(
        subreddits : list, 
        logger : Logger
        ) -> pd.DataFrame:
    #Dictionary to store the data
    data = list()
    seen_titles = set()

    for subreddit in subreddits:
        subreddit = reddit.subreddit(subreddit)

        try:
            for submission in subreddit.top(limit=100, time_filter='all'):
                text = submission.selftext.lower()
                title = submission.title.lower()
                if pattern_matching(text, keywords) and (title not in seen_titles):
                    seen_titles.add(title)
                    data.append({
                        "title" : clean_text(title),
                        "score" : submission.score,
                        "id" : submission.id,
                        "subreddit" : submission.subreddit,
                        "url" : submission.url,
                        "num_comments" : submission.num_comments,
                        "num_upvotes" : submission.ups,
                        "selftext" : clean_text(text),
                        "created" : submission.created
                    }
                    )
                else:
                    logger.info(f"No matches found or already seen title: {submission.title} |||| The title already seen status : {title not in seen_titles}")
            time.sleep(1)
            for submission in subreddit.top(limit = 100, time_filter = 'month'):
                text = submission.selftext.lower()
                title = submission.title.lower()
                if pattern_matching(text, keywords) and (title not in seen_titles):
                    seen_titles.add(title)
                    data.append({
                        "title" : clean_text(title),
                        "score" : submission.score,
                        "id" : submission.id,
                        "subreddit" : submission.subreddit,
                        "url" : submission.url,
                        "num_comments" : submission.num_comments,
                        "num_upvotes" : submission.ups,
                        "selftext" : clean_text(text),
                        "created" : submission.created
                    }
                    )
                else:
                    logger.info(f"No matches found or already seen title: {submission.title} |||| The title already seen status : {title not in seen_titles}")
            time.sleep(1)
            for submission in subreddit.hot(limit = 100):
                text = submission.selftext.lower()
                title = submission.title.lower()
                if pattern_matching(text, keywords) and (title not in seen_titles):
                    seen_titles.add(title)
                    data.append({
                        "title" : clean_text(title),
                        "score" : submission.score,
                        "id" : submission.id,
                        "subreddit" : submission.subreddit,
                        "url" : submission.url,
                        "num_comments" : submission.num_comments,
                        "num_upvotes" : submission.ups,
                        "selftext" : clean_text(text),
                        "created" : submission.created
                    }
                    )
                else:
                    logger.info(f"No matches found or already seen title: {submission.title} |||| The title already seen status : {title not in seen_titles}")
            time.sleep(1)
        except Exception as e:
            print(e)
            continue
    df = pd.DataFrame(data, columns=["title", "score", "id", "subreddit", "url", "num_comments", "num_upvotes", "selftext", "created"])
    df['created'] = pd.to_datetime(df['created'], unit='s')
    return df


if __name__ == "__main__":

    #Script configs
    stop_words = set(stopwords.words('english'))
    wordnet = WordNetLemmatizer()

    load_dotenv(dotenv_path="./../.env")
    CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    CLIENT_SECRET = os.getenv("REDDIT_SECRET_KEY")
    USER_AGENT = os.getenv("REDDIT_USER_AGENT")
    USERNAME = os.getenv("REDDIT_USERNAME")
    PASSWORD = os.getenv("REDDIT_PASSWORD")
    logger = Logger("RedditBot") 
    formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = FileHandler("./../../data/reddit_log.log")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


    #Initialize Reddit API

    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT,
        username=USERNAME,
        password=PASSWORD
    )

    logger.info("Reddit API initialized")

    #Load the keywords

    keywords = ["depressed", "anxiety", "suicidal", "overwhelmed", "addiction help", "self-harm", "mental health", "panic attack", "mental breakdown","intrusive thoughts","exhausted emotionally","burnout","social anxiety","imposter syndrome","emotional numbness","existential crisis", "depression", "drugs", "sober", "alcohol", "addiction", "substance abuse", "relapse", "withrawal", "alienated", "lonely", "isolated", "alone"]

    logger.info(f"Keywords loaded. There are {len(keywords)} keywords and they are : {keywords}")

    #Create a list of subreddits to scrape
    subreddits = ['SuicideWatch', 'Suicidalideations', 'suicidaltendencies', 'Suicidal_Comforters', 'depression', 'depression_help', 'SubstanceAbuseHelp', 'stopdrinking', 'Anxiety', 'Anxietyhelp', 'AnxietyDepression']

    #Scrape the data
    logger.info(f"Scraping data from {len(subreddits)} subreddits : ")
    df = scrape_reddit(subreddits, logger=logger)
    df.to_csv("./../../data/reddit_data.csv", index=False)
    logger.info("Data saved to reddit_data.csv")