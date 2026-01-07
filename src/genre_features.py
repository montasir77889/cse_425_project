import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def extract_genre_features(csv_path):
    df = pd.read_csv(csv_path)
    enc = OneHotEncoder(sparse_output=False)
    genre_vec = enc.fit_transform(df[["genre"]])
    return genre_vec, enc.categories_
