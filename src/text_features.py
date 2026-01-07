import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_lyrics_embeddings(csv_path):
    df = pd.read_csv(csv_path)
    df["lyrics"] = df["lyrics"].fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        max_features=300,
        stop_words="english"
    )

    return vectorizer.fit_transform(df["lyrics"]).toarray()
