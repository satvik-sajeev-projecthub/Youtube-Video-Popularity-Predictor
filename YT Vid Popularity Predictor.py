from googleapiclient.discovery import build
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("API_KEY")

youtube = build('youtube', 'v3', developerKey=api_key)

videos = []

request = youtube.search().list(
    part="snippet",
    q="Football",  
    maxResults=50,
    type="video"
)

response = request.execute()

for item in response['items']:
    video_id = item['id']['videoId']
    title = item['snippet']['title']
    channel = item['snippet']['channelTitle']
    publish_date = item['snippet']['publishedAt']
    videos.append([video_id, title, channel, publish_date])

# Save to CSV
df = pd.DataFrame(videos, columns=["video_id", "title", "channel", "publish_date"])
df.to_csv("youtube_videos.csv", index=False)
print("✅ Data saved as youtube_videos.csv")



from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import pandas as pd

# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")

youtube = build('youtube', 'v3', developerKey=api_key)

# Load your CSV with video_ids
df = pd.read_csv("youtube_videos.csv")

# Prepare lists to store extra features
views_list = []
likes_list = []
comments_list = []
duration_list = []

for video_id in df['video_id']:
    request = youtube.videos().list(
        part="statistics,contentDetails",
        id=video_id
    )
    response = request.execute()
    stats = response['items'][0]['statistics']
    content = response['items'][0]['contentDetails']

    views_list.append(int(stats.get('viewCount', 0)))
    likes_list.append(int(stats.get('likeCount', 0)))
    comments_list.append(int(stats.get('commentCount', 0)))
    duration_list.append(content.get('duration', 'PT0S'))  # ISO 8601 format


df['views'] = views_list
df['likes'] = likes_list
df['comments'] = comments_list
df['duration'] = duration_list


df.to_csv("youtube_videos_enriched.csv", index=False)
print("✅ Data saved as youtube_videos_enriched.csv")

import isodate
import pandas as pd

df = pd.read_csv("youtube_videos_enriched.csv")

# Convert ISO 8601 duration to seconds
df['duration_seconds'] = df['duration'].apply(lambda x: int(isodate.parse_duration(x).total_seconds()))

# Save updated CSV
df.to_csv("youtube_videos_final.csv", index=False)
print("✅ Durations converted and saved as youtube_videos_final.csv")

import pandas as pd


df = pd.read_csv("youtube_videos_final.csv")

# Create a label: popular if views > 1,000,000
df['is_popular'] = df['views'].apply(lambda x: 1 if x > 1000000 else 0)


df = df[['title', 'duration_seconds', 'likes', 'comments', 'is_popular']]
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english', max_features=500)  # top 500 words
title_features = tfidf.fit_transform(df['title'])
from scipy.sparse import hstack

# Numeric features: duration, likes, comments
numeric_features = df[['duration_seconds', 'likes', 'comments']].values

# Combine numeric + text features
X = hstack([title_features, numeric_features])
y = df['is_popular']
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
def predict_popularity(title, duration, likes, comments):
    title_vec = tfidf.transform([title])
    features = hstack([title_vec, [[duration, likes, comments]]])
    return model.predict(features)[0]


print(predict_popularity("Football Games Highlights", 600, 5000, 300))




