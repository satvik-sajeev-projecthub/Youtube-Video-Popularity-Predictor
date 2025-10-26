YouTube Popularity Predictor

📌 Project Overview

This project predicts whether a YouTube video will be popular based on its title, duration, likes, and comments.
It demonstrates end-to-end machine learning workflow: data collection via YouTube API, data cleaning, feature engineering, model training, and prediction.

🛠 Technologies Used

Python 3.12

Pandas, NumPy

scikit-learn (ML models)

Google API Client (google-api-python-client)

python-dotenv (secure API key handling)

isodate (duration parsing)

🔹 Dataset

Collected using the YouTube Data API v3.

Features include:

video_id

title

channel

publish_date

views, likes, comments

duration_seconds

Label (is_popular) is 1 if views > 1M, otherwise 0.

⚙️ How to Run

Clone the repository:

git clone <your-repo-url>
cd <project-folder>


Install dependencies:

pip install -r requirements.txt


Add a .env file with your YouTube API key:

GOOGLE_API_KEY=YOUR_API_KEY_HERE


Run the main script:

python YT_Popularity_Predictor.py

📈 Model

Random Forest Classifier trained on video features.

Test accuracy: ~0.9 (varies by dataset).

Predict popularity for new videos with:

predict_popularity(title, duration, likes, comments)


Output: 1 = popular, 0 = not popular

💡 Key Learnings

Collecting real-world data from APIs

Safe handling of API keys with .env

Text preprocessing using TF-IDF

Combining text and numeric features for ML

End-to-end ML workflow (train → test → predict)

📂 Project Structure
YT_Popularity_Predictor/
│
├─ YT_Popularity_Predictor.py  # Main code
├─ youtube_videos.csv           # Raw dataset
├─ youtube_videos_enriched.csv  # Dataset with stats
├─ youtube_videos_final.csv     # Final dataset with numeric duration
├─ .env                        # API key (not uploaded)
└─ README.md
