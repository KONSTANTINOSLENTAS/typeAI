import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager
from bson import ObjectId 

# --- 1. Configuration ---
app = Flask(__name__)
CORS(app)

# --- NEW: Explicit MongoDB Configuration ---
# 1. Get the full connection string from environment variables
MONGO_URI = os.environ.get('DATABASE_URL', 'mongodb://localhost:27017/typing_db')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'default-fallback-secret-key')

# 2. Extract the database name from the URL if possible, or set a default
# We need the name 'typing_db' explicitly for the client object.
DB_NAME = 'typing_db' 

# 3. Initialize the MongoClient with the full URI
client = MongoClient(MONGO_URI)

# 4. Explicitly select the database using the name
# This line bypasses the complex URI parsing that was failing.
db = client[DB_NAME] 
# --- END NEW BLOCK ---

bcrypt = Bcrypt(app)
jwt = JWTManager(app)
# (rest of the file is unchanged)

# Collections (MongoDB "tables")
users_collection = db.users
samples_collection = db.samples

MODEL_FILE = "multi_user_model.joblib"
MIN_SAMPLES_TO_TRAIN = 10

model = None
label_encoder = LabelEncoder()
model_columns = []

# --- 2. Feature Engineering ---
def calculate_global_features(events_list):
    df = pd.DataFrame(events_list)
    release_events = df[df['event'] == 'release'].dropna(subset=['hold_time_ms'])
    press_events = df[df['event'] == 'press'].copy()
    if press_events.empty or release_events.empty: return {}
    press_events['pp_latency'] = press_events['time_ms'].diff()
    features = {
        'mean_hold_time': release_events['hold_time_ms'].mean(),
        'std_hold_time': release_events['std_hold_time'].std(),
        'mean_pp_latency': press_events['pp_latency'].mean(),
        'std_pp_latency': press_events['pp_latency'].std(),
        'typing_speed_kps': len(press_events) / (df['time_ms'].max() / 1000)
    }
    return {k: v if pd.notna(v) else 0 for k, v in features.items()}

# --- 3. API Routes ---
@app.route("/status", methods=["GET"])
def get_status():
    try:
        model_ready = samples_collection.count_documents({}) >= MIN_SAMPLES_TO_TRAIN
        return jsonify({"model_ready": model_ready})
    except Exception:
        return jsonify({"model_ready": False})

@app.route("/users", methods=["GET"])
def get_user_stats():
    try:
        pipeline = [
            {"$group": {"_id": "$username", "samples": {"$sum": 1}}}
        ]
        stats = list(samples_collection.aggregate(pipeline))
        stats = [{"username": s["_id"], "samples": s["samples"]} for s in stats]
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username, password = data.get('username'), data.get('password')
    if not username or not password: return jsonify({"error": "Missing username or password"}), 400
    if users_collection.find_one({"username": username}): return jsonify({"error": "Username already exists"}), 400
    
    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    users_collection.insert_one({"username": username, "password_hash": password_hash})
    return jsonify({"status": "user created", "username": username})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username, password = data.get('username'), data.get('password')
    user = users_collection.find_one({"username": username})
    if user and bcrypt.check_password_hash(user['password_hash'], password):
        access_token = create_access_token(identity=user['username'])
        return jsonify(access_token=access_token)
    return jsonify({"error": "Invalid username or password"}), 401

@app.route("/add_sample", methods=["POST"])
@jwt_required()
def add_sample():
    username = get_jwt_identity()
    user = users_collection.find_one({"username": username})
    if not user: return jsonify({"error": "User not found"}), 404
            
    data = request.json
    events, accuracy, sample_type = data.get('events'), data.get('accuracy'), data.get('sample_type')
    if not events: return jsonify({"error": "Missing 'events' data"}), 400
    
    features = calculate_global_features(events)
    
    sample_doc = {
        "username": username,
        "sample_type": sample_type,
        "accuracy": accuracy if sample_type == 'diverse' else None,
        "mean_hold_time": features.get('mean_hold_time'),
        "std_hold_time": features.get('std_hold_time'),
        "mean_pp_latency": features.get('mean_pp_latency'),
        "std_pp_latency": features.get('std_pp_latency'),
        "typing_speed_kps": features.get('typing_speed_kps'),
        "user_id": user['_id']
    }
    samples_collection.insert_one(sample_doc)
    return jsonify({"status": "sample added"})

@app.route("/train", methods=["POST"])
@jwt_required()
def train_model():
    global model, label_encoder, model_columns
    
    try:
        all_samples = list(samples_collection.find({}))
        if not all_samples: return jsonify({"error": "No data to train on."}), 400
        df = pd.DataFrame(all_samples)
    except Exception as e:
        return jsonify({"error": f"Failed to read data from database: {e}"}), 500

    if len(df) < MIN_SAMPLES_TO_TRAIN: return jsonify({"error": f"Need at least {MIN_SAMPLES_TO_TRAIN} samples. You have {len(df)}."}), 400

    X = df.drop(columns=['_id', 'user_id', 'username', 'sample_type'])
    y_labels = df['username']
    X['accuracy'] = X['accuracy'].fillna(1.0)
    
    model_columns = X.columns.tolist()
    y = label_encoder.fit_transform(y_labels)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print(f"Model trained on {len(X)} samples for users: {label_encoder.classes_}")
    return jsonify({"status": "model trained", "users": label_encoder.classes_.tolist()})

@app.route("/predict", methods=["POST"])
def predict():
    global model, label_encoder, model_columns
    
    if model is None: 
        if not load_model_from_db():
            return jsonify({"error": "Model is not trained."}), 400

    data = request.json
    events = data.get('events')
    if not events: return jsonify({"error": "No 'events' data provided"}), 400

    features = calculate_global_features(events)
    features['accuracy'] = 1.0
    
    live_df = pd.DataFrame([features])
    live_df = live_df.reindex(columns=model_columns).fillna(0)
    
    probabilities = model.predict_proba(live_df)
    confidence = np.max(probabilities)
    prediction_index = np.argmax(probabilities, axis=1)
    predicted_user = label_encoder.inverse_transform(prediction_index)
    
    return jsonify({"predicted_user": predicted_user[0], "confidence": float(confidence)})

@app.route("/accuracy_leaderboard", methods=["GET"])
def get_accuracy_leaderboard():
    try:
        pipeline = [
            {"$match": {"sample_type": "diverse", "accuracy": {"$ne": None}}},
            {"$group": {"_id": "$username", "avg_accuracy": {"$avg": "$accuracy"}}},
            {"$sort": {"avg_accuracy": -1}}
        ]
        stats = list(samples_collection.aggregate(pipeline))
        stats = [{"username": s["_id"], "avg_accuracy": avg_acc} for s, avg_acc in [(d["_id"], d["avg_accuracy"]) for d in stats]]
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/user_stats/<username>", methods=["GET"])
def get_user_detail_stats(username):
    try:
        user_samples = list(samples_collection.find({"username": {"$regex": f"^{username}$", "$options": "i"}}))
        if not user_samples: return jsonify({"error": "User not found"}), 404
        
        df = pd.DataFrame(user_samples)
        
        avg_kps = df['typing_speed_kps'].mean()
        wpm = (avg_kps * 60) / 5
        
        diverse_samples = df.dropna(subset=['accuracy'])
        avg_accuracy = diverse_samples['accuracy'].mean() if not diverse_samples.empty else None

        stats = {
            "username": username,
            "total_samples": len(df),
            "avg_wpm": wpm,
            "avg_hold_time_ms": df['mean_hold_time'].mean(),
            "avg_hold_std_ms": df['std_hold_time'].mean(),
            "avg_latency_ms": df['mean_pp_latency'].mean(),
            "avg_latency_std_ms": df['std_pp_latency'].mean(),
            "avg_accuracy": avg_accuracy
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def load_model_from_db():
    """Reads all samples from DB and rebuilds the AI model in memory."""
    global model, label_encoder, model_columns
    
    try:
        all_samples = list(samples_collection.find({}))
        if not all_samples: return False
        df = pd.DataFrame(all_samples)
        
        if len(df) < MIN_SAMPLES_TO_TRAIN: return False

        X = df.drop(columns=['_id', 'user_id', 'username', 'sample_type'])
        y_labels = df['username']
        X['accuracy'] = X['accuracy'].fillna(1.0)

        model_columns = X.columns.tolist()
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_labels)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        print("Model rebuilt successfully from MongoDB.")
        return True

    except Exception as e:
        print(f"Error during on-the-fly model rebuild: {e}")
        return False
        
if __name__ == "__main__":
    print(f"Starting Flask server at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))