import pandas as pd
import numpy as np
import pickle
from dotenv import load_dotenv
import os
import requests
import spacy
import re
from spacy.matcher import PhraseMatcher
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
model = spacy.load("en_core_web_sm")
load_dotenv("./.env")

# 1. Data Preprocessing:
# Clean the text by removing irrelevant characters like special symbols or URLs.

# 2. Lexicons:
# - Crisis Lexicon: List of crisis keywords with risk scores 
# - Temporal Lexicon: Time-related words with weights 
# - Severity Lexicon: Intensity modifiers with multipliers 

# 3. Sample Phrases & Clustering:
# - Create sample phrases for High, Medium, and Low risk.
# - Use k-means clustering to find centroids for each risk level (High, Medium, Low).

# 4. Initial Risk Assignment:
# - Generate embedding for the input text.
# - Compare it to centroids and assign a risk (High = 3, Medium = 2, Low = 1).

# 5. Lexicon-Based Scoring:
# - Find crisis keywords, calculate scores based on TF-IDF and risk.
# - Apply severity modifiers to adjust the scores.
# - Look for temporal words and add their weight.
# - Use sentiment analysis (VADER) for extra context.

# 6. Combined Lexicon Score:
# - Add up the modified keyword scores, temporal score, and sentiment score.

# 7. Final Risk Score:
# - Combine the initial risk with the lexicon score.
# - If lexicon score is high, increase risk; if low, decrease risk.
# - Use thresholds to decide how much to change the risk level.




##################### Helper functions  #####################
################################################################




def get_response(data):
    api_key = os.getenv("JINA_API_KEY")
    URL = "https://api.jina.ai/v1/embeddings"

    resp = requests.post(URL, headers={"Authorization": f"Bearer {api_key}"}, json={"input": data, "model" : "jina-embeddings-v3", "task" : "classification"})
    # print(resp.status_code)
    return resp.json()["data"][0]["embedding"]

def calculate_mean_pool(sentences):
    embeddings = []
    for sentence in sentences:
        embeddings.append(get_response(sentence))
    return np.mean(embeddings, axis=0)

def sigmoid_function(x, k = 4.5):
    return 1+ (2 / (1 + np.exp(- k * x)))
def map_similarity_to_risk_range(similarity, sensitivity=4.5):
    """
    Maps cosine similarity (-1 to 1) to a risk scale (1 to 3).
    1.0 = Low Risk, 2.0 = Neutral/Medium, 3.0 = High Risk
    """
    return 1 + (2 / (1 + np.exp(-sensitivity * similarity)))

def cosine_similarity_with_sim_diff(text_embedding, sim_diff_vec):
    return np.dot(text_embedding, sim_diff_vec) / (np.linalg.norm(text_embedding) * np.linalg.norm(sim_diff_vec))
def calculate_risk_projection(text_embedding, risk_axis_vec):
    # Ensure we are calculating cosine similarity (normalized dot product)
    norm_text = np.linalg.norm(text_embedding)
    norm_axis = np.linalg.norm(risk_axis_vec)
    return np.dot(text_embedding, risk_axis_vec) / (norm_text * norm_axis)

def check_crisis_lexicon(text, lexicon):
    nlp = spacy.load("en_core_web_sm")  # Load NLP model
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")  # Match case-insensitive

    # Convert lexicon into spaCy patterns
    patterns = [nlp.make_doc(phrase) for phrase in lexicon]
    matcher.add("CrisisLexicon", patterns)

    doc = nlp(text)  # Process text
    matches = matcher(doc)

    # Extract matched words and phrases
    found_words = [doc[start:end].text for match_id, start, end in matches]
    
    return found_words

def check_dependency_lexicons(text: str, lexicon: dict[str,int], model: spacy.language.Language, crisis_words: list[str]):
    nlp = model  
    matcher = PhraseMatcher(nlp.vocab, attr="LEMMA")
    
    # Process text
    doc = nlp(text)  
    
    # Separate single-word and multi-word lexicon terms
    single_word_terms = {word for word in lexicon if " " not in word}
    multi_word_terms = {word for word in lexicon if " " in word}

    # Ensure multi-word terms are processed through full NLP pipeline for lemmatization
    if multi_word_terms:
        patterns = [nlp(phrase) for phrase in multi_word_terms] 
        matcher.add("LexiconMatcher", patterns)
    
    found_words = []

    #Check for multi-word matches
    for match_id, start, end in matcher(doc):
        matched_span = doc[start:end]
        if matched_span.root.head.lemma_ in crisis_words:
            found_words.append(matched_span.text)
    
    #Check for single-word matches directly
    for token in doc:
        if token.lemma_ in single_word_terms:  
            found_words.append(token.text)

    return found_words


def return_lexicon_score(word, lexicon):
    if word in lexicon:
        return lexicon[word]/10
    else:
        return 0

def score_to_sentiment(score : float, threshold = 0.5):

        #Using threshold to classify sentiment
        if score > threshold:
            return "positive"
        elif score < -threshold:
            return "negative"
        else:
            return "neutral"
        



####################### Lexicons #####################
#####################################################





crisis_lexicon = {
    "suicidal": 10, "suicide": 10, "unalive": 10, "killing myself": 10, 
    "end it all": 10, "no way out": 9, "die": 9, "goodbye forever": 9, 
    "hopeless": 8, "self-harm": 8, "cutting": 8, "jump off": 8, 
    "pills": 7, "overdose": 7, "hang myself": 10, "drown myself": 9, 
    "bridge": 8, "gun": 9, "razor": 7, "pain is unbearable": 8, 
    "nothing matters": 7, "no point in living": 9, "life is pointless": 8, 
    "I give up": 8, "can't take this anymore": 9, "don't want to be here": 9, 
    "I'm done": 8, "no future": 7, "nobody cares": 6, "I'm exhausted": 6,
    "can't stop crying": 6, "broken beyond repair": 7, "hate myself": 7,
    "failure": 6, "worthless": 7, "empty": 6, "lost all hope": 9, "drowning": 8,
    "dark thoughts": 8, "feeling trapped": 7, "no escape": 8,
    "can't breathe": 7, "suffocating": 8, "pain never ends": 8,
    "stuck in my head": 7, "voices won't stop": 8, "can't sleep": 7,
    "nightmares": 6, "crying all the time": 7, "feeling numb": 8,
    "can't focus": 6, "lost interest": 7, "everything is hard": 8,
    "can't get out of bed": 8, "no energy": 7, "feeling heavy": 8,
    "can't eat": 7, "lost appetite": 6, "feeling sick": 7,
}


temporal_lexicon = {
    "now": 10, "immediately": 10, "tonight": 9, "this moment": 9, 
    "right now": 10, "soon": 8, "today": 8, "can't wait": 9, 
    "urgent": 9, "this second": 10, "tomorrow": 7, "before it's too late": 9, 
    "running out of time": 9, "this week": 6, "eventually": 5, 
    "someday": 4, "whenever": 3, "later": 2
}


severity_lexicon = {
    "extremely": 10, "desperately": 10, "completely": 9, "totally": 9, 
    "absolutely": 9, "cannot handle": 9, "falling apart": 8, "breaking down": 8, 
    "barely holding on": 9, "too much to bear": 9, "can't take anymore": 10, 
    "beyond saving": 9, "drowning": 8, "collapsing": 8, "out of control": 8, 
    "spiraling": 7, "barely surviving": 7, "giving up": 8, "crushed": 7, 
    "overwhelmed": 7, "anxious": 6, "stressed": 5, "worried": 4, "sad": 3, 
    "upset": 2, "unhappy": 2
}


#Serverity risk sentences
high_risk = [
    "I don't want to live anymore", "I feel like ending it all", "No point in going on",
    "I need to disappear", "I wish I never woke up", "It hurts too much to stay",
    "I'm ready to give up", "I need a way to make it stop", "I keep thinking about ending things",
    "I just want the pain to end", "Nobody would miss me if I were gone", 
    "I'm tired of pretending everything is okay", "I can't do this anymore",
    "Everything is too much", "I feel trapped in my own mind", "I see no future for myself",
    "No one understands what I'm going through", "I can't escape my thoughts",
    "I don't deserve to live", "I just want to disappear forever",
    "I wish I could sleep and never wake up", "I feel like I'm drowning and no one notices",
    "I hate myself so much", "I can't stop thinking about how to end it",
    "I've been planning my way out", "I hope this is the last time I feel this way",
    "I feel like a burden to everyone", "Nothing makes sense anymore",
    "Every day is unbearable", "The pain never stops", "I want to turn everything off",
    "I just need a way out", "I'm done with everything", "I've lost all hope",
    "No one would care if I was gone", "I feel completely empty inside",
    "I have no energy left to fight", "I'm exhausted from pretending I'm okay",
    "I wish I had the courage to do it", "I don't belong in this world",
    "Why am I still here?", "I'm scared of myself", "I can't keep fighting these thoughts",
    "I don't want to feel this anymore", "I feel like I'm suffocating",
    "The voices in my head won't stop", "I'm tired of struggling", "I don't see the point anymore",
    "I just need the pain to end", "I feel so hopeless", "This is my last goodbye"
]

low_risk = [
    "Mental health is important", "Therapy has helped me a lot", 
    "We need to talk about depression more", "Journaling has really helped my anxiety",
    "It's okay to ask for help", "Taking a break for my mental health",
    "Finding the right medication changed my life", "Self-care is so important",
    "It's okay to not be okay", "Healing takes time", "Learning to set boundaries is hard",
    "Exercise really helps my mood", "Meditation helps me stay grounded",
    "Talking to friends makes a big difference", "Therapy isn't just for when you're struggling",
    "Getting enough sleep is key to my mental health",
    "Protecting my peace at all costs", "Gotta focus on my mental health today",
    "Normalize taking mental health days", "I need to touch grass",
    "Sending good vibes to everyone struggling", "Being mindful helps me stay present",
    "A good routine helps my mental health", "Music is my therapy",
    "I'm finally learning to love myself", "Having a support system is everything",
    "Deep breathing really helps my anxiety", "Taking time for myself feels so good",
    "Trying to stay positive every day", "Therapy has changed my perspective",
    "I'm working on improving my mindset", "Setting boundaries has been life-changing",
    "Talking about mental health should be normal", "Happiness is a journey, not a destination",
    "Being kind to yourself is the first step", "Gratitude helps shift my perspective",
    "Mental health days should be mandatory", "I'm finally prioritizing myself",
    "Healing isn't linear, and that's okay", "Sleep is my best coping mechanism",
    "Sometimes all you need is a deep breath", "I'm learning to forgive myself",
    "Fresh air and a walk always help", "I try to focus on the little things",
    "Checking in with yourself is important", "Mental health matters more than productivity",
    "Journaling my thoughts helps me process emotions", "Being in nature helps clear my mind",
    "Meditation is a game-changer", "Prioritizing my mental well-being every day"
]


# high_risk_avg = calculate_mean_pool(high_risk)
# low_risk_avg = calculate_mean_pool(low_risk)
# sim_diff = high_risk_avg - low_risk_avg
# sim_diff = sim_diff / np.linalg.norm(sim_diff)

sim_diff = pickle.load(open("./sim_diff.pkl", "rb"))






#################### Inference ####################
###################################################





def inference_basic_score (inference_text):
    text_embedding = get_response(inference_text)
    # return sigmoid_function(cosine_similarity_with_sim_diff(text_embedding, sim_diff))
    similarity = calculate_risk_projection(text_embedding, sim_diff)
    return map_similarity_to_risk_range(similarity)

def final_score(inference_text):


    crisis_words = check_crisis_lexicon(inference_text, crisis_lexicon)
    # print(f"Crisis words: {crisis_words}")
    crisis_score = sum(return_lexicon_score(word, crisis_lexicon) 
                      for word in crisis_words)
    # print(f"Crisis score: {crisis_score}")
    



    temporal_dependencies = check_dependency_lexicons(inference_text, temporal_lexicon, 
                                                    model, crisis_words)
    # print(f"Temporal dependencies: {temporal_dependencies}")
    temporal_score = sum(return_lexicon_score(word, temporal_lexicon) 
                        for word in temporal_dependencies)
    # print(f"Temporal score: {temporal_score}")
    

    
    severity_dependencies = check_dependency_lexicons(inference_text, severity_lexicon, 
                                                    model, crisis_words)
    # print(f"Severity dependencies: {severity_dependencies}")
    severity_score = np.reciprocal(sum(return_lexicon_score(word, severity_lexicon) 
                        for word in severity_dependencies)) if severity_dependencies else 1
    # print(f"Severity score: {severity_score}")


    inference_score = inference_basic_score(inference_text)
    # print(f"Inference score: {inference_score}")
    
    return (inference_score * severity_score) + temporal_score + crisis_score 

def risk_classification(final_score, Threshold_up = 5, Threshold_down = 1.8):
    if final_score >= Threshold_up:
        return "High Risk"
    elif final_score <= Threshold_down:
        return "Low Risk"
    else:
        return "Medium Risk"
    

def final_risk_classification(text):
    score = final_score(text)
    classification = risk_classification(score)
    print(f"Final Score: {score}")
    print(f"Classification: {classification}")

if __name__ == "__main__":
    
    #Import the data
    df = pd.read_csv("./../../data/reddit_data.csv", encoding='utf-8')
    df.created = pd.to_datetime(df.created)
    df.title = df.title.astype(str)
    df.selftext = df.selftext.astype(str)

    #Preprocess the data
    df["title"] = df["title"].apply(lambda x : re.sub(r"http\S+|www\S+|https\S+", "", x))
    df["selftext"] = df["selftext"].apply(lambda x : re.sub(r"http\S+|www\S+|https\S+", "", x))

    #VADER sentiment analysis
    
    vader = SentimentIntensityAnalyzer()
    df["vader_title_sentiment"] = df["title"].apply(lambda x : vader.polarity_scores(x)["compound"])
    df["vader_selftext_sentiment"] = df["selftext"].apply(lambda x : vader.polarity_scores(x)["compound"])
    df["vader_sentiment"] = df["selftext"].apply(
        lambda x : score_to_sentiment(vader.polarity_scores(x)["compound"])
        )
    
    #Applying the final risk classification
    df["final_risk_score"] = df["selftext"].apply(lambda x : final_score(x))
    df["risk_classification"] = df["final_risk_score"].apply(lambda x : risk_classification(x))
    df["risk_classification"] = df["risk_classification"].astype(str)
    df["final_risk_score"] = df["final_risk_score"].astype(float)
    df["final_risk_score"] = df["final_risk_score"].round(2)

    #Save the final dataframe
    df.to_csv("./../../data/final_reddit_data.csv", index = False)
    print("Final dataframe saved as final_reddit_data.csv")
    