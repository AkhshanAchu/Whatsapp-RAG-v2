import pandas as pd
import numpy as np
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import emoji

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

def caps_ratio(text):
    caps = sum(1 for c in text if c.isupper())
    return caps / len(text) if len(text) > 0 else 0

def generate_chat_stats(file_path):
    df = pd.read_csv(file_path, parse_dates=['date_time'])

    # Preprocessing
    df['date'] = df['date_time'].dt.date
    df['hour'] = df['date_time'].dt.hour
    df['weekday'] = df['date_time'].dt.day_name()
    df['day'] = df['date_time'].dt.normalize()

    df = df.dropna(subset=['message'])
    df = df[df['message'].str.strip().astype(bool)]

    # --- Word/Char/Sentence Stats ---
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    df['char_count'] = df['message'].apply(len)
    df['caps_ratio'] = df['message'].apply(caps_ratio)

    # --- Per User Stats ---
    user_stats = df.groupby('user').agg(
        total_messages=('message', 'count'),
        total_words=('word_count', 'sum'),
        avg_words=('word_count', 'mean'),
        avg_chars=('char_count', 'mean'),
        avg_caps_ratio=('caps_ratio', 'mean')
    ).to_dict(orient='index')

    # --- Emoji Stats ---
    df['emojis'] = df['message'].apply(extract_emojis)
    emoji_list = [e for sublist in df['emojis'] for e in sublist]
    emoji_counts = dict(Counter(emoji_list).most_common(10))

    # --- Sentiment ---
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df['message'].apply(lambda x: sia.polarity_scores(x)['compound'])
    sentiment_per_user = df.groupby('user')['sentiment'].mean().to_dict()

    # --- Response Time ---
    df = df.sort_values(by='date_time')
    df['next_user'] = df['user'].shift(-1)
    df['next_time'] = df['date_time'].shift(-1)
    df['response_time_sec'] = (df['next_time'] - df['date_time']).dt.total_seconds()
    resp_df = df[df['user'] != df['next_user']]
    avg_resp_time = resp_df.groupby('user')['response_time_sec'].mean().fillna(0).to_dict()

    # --- Keyword Frequency ---
    cv = CountVectorizer(stop_words='english', max_features=100)
    X = cv.fit_transform(df['message'])
    top_words = pd.Series(X.toarray().sum(axis=0), index=cv.get_feature_names_out()).nlargest(20).to_dict()

    # --- Wake/Sleep Patterns ---
    first_msg_hour = df.groupby(['date', 'user'])['hour'].min().reset_index().groupby('user')['hour'].mean().to_dict()
    last_msg_hour = df.groupby(['date', 'user'])['hour'].max().reset_index().groupby('user')['hour'].mean().to_dict()

    # --- Dominance Score ---
    msg_count = df['user'].value_counts(normalize=True).to_dict()

    # --- Longest Silence ---
    df['gap_to_next'] = df['date_time'].shift(-1) - df['date_time']
    longest_silence = df[df['user'] != df['next_user']].groupby('user')['gap_to_next'].max().dt.total_seconds().fillna(0).to_dict()

    # Final dictionary
    chat_stats = {
        'per_user_stats': user_stats,
        'emoji_counts': emoji_counts,
        'avg_sentiment': sentiment_per_user,
        'avg_response_time_sec': avg_resp_time,
        'top_words': top_words,
        'wake_sleep_pattern': {
            'avg_first_msg_hour': first_msg_hour,
            'avg_last_msg_hour': last_msg_hour,
        },
        'dominance_score': msg_count,
        'longest_silence_seconds': longest_silence
    }

    return chat_stats
