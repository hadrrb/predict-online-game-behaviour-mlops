import pandas as pd
import requests
import ast
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import FeatureHasher

df = pd.read_csv("../data/online_gaming_behavior_dataset.csv")

enc = LabelEncoder()
hasher = FeatureHasher(n_features=2, input_type="string")
hashed_features = hasher.transform(df["Gender"].apply(lambda df: [df]))
hashed_df = pd.DataFrame(
    hashed_features.toarray(), columns=[f"gender_{i}" for i in range(2)]
)
df = pd.concat([df, hashed_df], axis=1)
df["Location"] = enc.fit_transform(df["Location"])
df["GameGenre"] = enc.fit_transform(df["GameGenre"])
df["GameDifficulty"] = enc.fit_transform(df["GameDifficulty"])
df["PlayTimePerWeek"] = df["SessionsPerWeek"] * df["AvgSessionDurationMinutes"] / 60
y_true = enc.fit_transform(df["EngagementLevel"])
df = df.drop(
    [
        "SessionsPerWeek",
        "AvgSessionDurationMinutes",
        "Gender",
        "PlayerID",
        "EngagementLevel",
    ],
    axis=1,
)

url = "http://127.0.0.1:7777/predict"
headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, json=df.to_dict())

y_predict = ast.literal_eval(response.text)

print(accuracy_score(y_true, y_predict))
