import tweepy
import pandas as pd
import os
from dotenv import load_dotenv
from typing import List, Dict, Union
import time
import tweepy.client


def set_api(auth_2 : bool) -> tweepy.API:

    #Loading the .env file
    load_dotenv()
    if not auth_2:
        #Getting the key
        consumer_key = os.getenv('CONSUMER_KEY')
        consumer_secret = os.getenv('CONSUMER_SECRET_KEY')
        access_token = os.getenv('ACCESS_TOKEN')
        access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

        #Setting the auth
        auth = tweepy.OAuth1UserHandler(
            consumer_key,
            consumer_secret,
            access_token,
            access_token_secret
        )

        #Set the API
        api = tweepy.API(
            auth,
            timeout=1,
            wait_on_rate_limit=True,
            )
    else:
        bearer_token = os.getenv('BEARER_TOKEN')
        consumer_key = os.getenv('CONSUMER_KEY')
        consumer_secret = os.getenv('CONSUMER_SECRET_KEY')
        access_token = os.getenv('ACCESS_TOKEN')
        access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

        api = tweepy.Client(
            bearer_token= bearer_token,
            consumer_key= consumer_key,
            consumer_secret= consumer_secret,
            access_token= access_token,
            access_token_secret= access_token_secret,
            wait_on_rate_limit=True,
        )
    

    return api


# We will get tweets related to mental health distress, substance use, or suicidality.
# Observing the search results using the twitter account revealed that merely using mental health/substance use/suicidality as keywords would limit range of topics and tweets that we can get. 
# Queries that are more specific to the topic of interest are needed :
# 1. Mental health distress: depression, anxiety, stress, panic attack, bipolar disorder, schizophrenia, PTSD, OCD, eating disorder, self-harm, self-in, prevention of mental health distress
# 2. Substance use: alcohol, drug, marijuana, cocaine, heroin, meth, opioid, addiction, substance use disorder, alcoholism, prevention of social alcoholism, prevention of drug abuse, prevention of substance use disorder
# 3. Suicidality : how to commit suicide, suicide prevention, suicidal tendencies, suicidal declartions

#Test function
def test_function(
        api : Union[tweepy.Client, tweepy.API], 
        query : List[str], 
        no_of_requests : int, 
        delay : Union[int, time.time]
        ) -> tweepy.client.Response:

    json_object = dict()
    if type(api) == tweepy.Client:
        try: 
            # We are going to call this once every 15 minutes
            

            for i in range(no_of_requests):
                last_req_time = time.time()
                while(time.time() - last_req_time < delay):
                    search =  api.search_recent_tweets(
                        query = query
                    )
                    print(search)
                    json_object.update(search.data)
                    last_req_time = time.time() 
        except BaseException as e:
            print(f"Wasnt able to scrape data. \nReason : {e}")
        
        return json_object
     
    else:

        try : 
            search = api.search_tweets(
                q = query,
                lang = 'en',
                count = 1,
                result_type = 'recent'
            )
            
        except BaseException as e:
            print(f"Wasnt scraped. \nReason: {e}")



if __name__ == "__main__":

    df = pd.read_json(test_function(set_api(auth_2=True), "depression", 5, 16))
    df.to_csv("./../data/basic_request.csv")
    