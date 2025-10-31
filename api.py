import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import create_access_token, get_jwt_identity, jwt_required, JWTManager

# --- 1. Configuration (Unchanged) ---
app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'your-super-secret-random-key'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

DB_FILE = "master_features.csv"
MODEL_FILE = "multi_user_model.joblib"
MIN_SAMPLES_TO_TRAIN = 10

model = None
label_encoder = LabelEncoder()
model_columns = []

# --- 2. Database Model (Unchanged) ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

# --- 3. Feature Engineering (Unchanged) ---
def calculate_global_features(events_list):
    df = pd.DataFrame(events_list)
    release_events = df[df['event'] == 'release'].dropna(subset=['hold_time_ms'])
    press_events = df[df['event'] == 'press'].copy()
    if press_events.empty or release_events.empty: return {}
    press_events['pp_latency'] = press_events['time_ms'].diff()
    features = {
        'mean_hold_time': release_events['hold_time_ms'].mean(),
        'std_hold_time': release_events['hold_time_ms'].std(),
        'mean_pp_latency': press_events['pp_latency'].mean(),
        'std_pp_latency': press_events['pp_latency'].std(),
        'typing_speed_kps': len(press_events) / (df['time_ms'].max() / 1000)
    }
    return {k: v if pd.notna(v) else 0 for k, v in features.items()}

# --- 4. API Routes (Status, Users, Security) (Unchanged) ---
@app.route("/status", methods=["GET"])
def get_status():
    model_exists = os.path.exists(MODEL_FILE)
    return jsonify({"model_ready": model_exists})

@app.route("/users", methods=["GET"])
def get_user_stats():
    if not os.path.exists(DB_FILE): return jsonify([])
    try:
        df = pd.read_csv(DB_FILE)
        stats = df.groupby('username').size().reset_index(name='samples')
        return jsonify(stats.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    username, password = data.get('username'), data.get('password')
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password_hash=password_hash)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({"status": "user created", "username": username})

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username, password = data.get('username'), data.get('password')
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        access_token = create_access_token(identity=user.username)
        return jsonify(access_token=access_token)
    return jsonify({"error": "Invalid username or password"}), 401

# --- 5. UPDATED API Route: Add Sample ---
@app.route("/add_sample", methods=["POST"])
@jwt_required()
def add_sample():
    username = get_jwt_identity()
    data = request.json
    events = data.get('events')
    accuracy = data.get('accuracy')
    # NEW: Get the sample type
    sample_type = data.get('sample_type') # 'diverse' or 'free'

    if not events:
        return jsonify({"error": "Missing 'events' data"}), 400
    
    features = calculate_global_features(events)
    features['username'] = username
    features['sample_type'] = sample_type
    # NEW: Only store accuracy if it's a diverse sample
    features['accuracy'] = accuracy if sample_type == 'diverse' else np.nan
    
    df = pd.DataFrame([features])
    file_exists = os.path.exists(DB_FILE)
    df.to_csv(DB_FILE, mode='a', header=not file_exists, index=False)
    
    return jsonify({"status": "sample added"})

# --- 6. API Route: Train the Model (UPDATED) ---
@app.route("/train", methods=["POST"])
@jwt_required()
def train_model():
    global model, label_encoder, model_columns
    if not os.path.exists(DB_FILE):
        return jsonify({"error": "No data to train on. Please add samples."}), 400

    # --- NEW: Add error handling for corrupt CSV ---
    try:
        df = pd.read_csv(DB_FILE)
    except pd.errors.EmptyDataError:
        return jsonify({"error": "Data file is empty or corrupt. Please delete 'master_features.csv' and restart."}), 500
    except Exception as e:
        return jsonify({"error": f"Error reading data file: {e}"}), 500
    # --- End of new code ---

    if len(df) < MIN_SAMPLES_TO_TRAIN:
        return jsonify({"error": f"Not enough data. Need at least {MIN_SAMPLES_TO_TRAIN} samples. You have {len(df)}."}), 400

    # NEW: We drop 'sample_type' but keep 'accuracy'
    X = df.drop(columns=['username', 'sample_type'])
    y_labels = df['username']
    
    # NEW: Fill NaN (from free text) with 1.0 (100%)
    X['accuracy'] = X['accuracy'].fillna(1.0)
    
    model_columns = X.columns.tolist()
    y = label_encoder.fit_transform(y_labels)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, MODEL_FILE)
    print(f"Model trained on {len(X)} samples for users: {label_encoder.classes_}")
    return jsonify({"status": "model trained", "users": label_encoder.classes_.tolist()})

# --- 7. UPDATED API Route: Predict ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        load_model_on_startup()
        if model is None: return jsonify({"error": "Model is not trained."}), 400

    data = request.json
    events = data.get('events')
    if not events: return jsonify({"error": "No 'events' data provided"}), 400

    features = calculate_global_features(events)
    features['accuracy'] = 1.0
    
    live_df = pd.DataFrame([features])
    live_df = live_df.reindex(columns=model_columns).fillna(0)
    
    # --- NEW: Get probabilities instead of just a single prediction ---
    # probabilities will be an array like: [[0.1, 0.8, 0.1]]
    probabilities = model.predict_proba(live_df)
    
    # Get the highest probability (e.g., 0.8)
    confidence = np.max(probabilities)
    
    # Get the *index* of the highest probability (e.g., 1)
    prediction_index = np.argmax(probabilities, axis=1)
    
    # Convert the index back to a username (e.g., 'netix')
    predicted_user = label_encoder.inverse_transform(prediction_index)
    
    # Return both the user AND the confidence score
    # We must convert confidence from a NumPy float to a standard Python float for JSON
    return jsonify({
        "predicted_user": predicted_user[0],
        "confidence": float(confidence) 
    })

# --- 8. NEW API Route: Accuracy Leaderboard ---
@app.route("/accuracy_leaderboard", methods=["GET"])
def get_accuracy_leaderboard():
    if not os.path.exists(DB_FILE): return jsonify([])
    try:
        df = pd.read_csv(DB_FILE)
        # Drop all rows that don't have an accuracy (i.e., 'free' samples)
        df = df.dropna(subset=['accuracy'])
        if df.empty: return jsonify([])
        
        # Get the mean accuracy for each user
        stats = df.groupby('username')['accuracy'].mean().reset_index(name='avg_accuracy')
        stats = stats.sort_values(by='avg_accuracy', ascending=False)
        return jsonify(stats.to_dict('records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 9. NEW API Route: User Detail Stats ---
@app.route("/user_stats/<username>", methods=["GET"])
def get_user_detail_stats(username):
    if not os.path.exists(DB_FILE):
        return jsonify({"error": "No data file found"}), 404
    try:
        df = pd.read_csv(DB_FILE)
        user_df = df[df['username'].str.lower() == username.lower()]
        
        if user_df.empty:
            return jsonify({"error": "User not found"}), 404

        # Calculate Words Per Minute (WPM)
        # 1 KPS = 12 WPM (based on 5 chars/word)
        avg_kps = user_df['typing_speed_kps'].mean()
        wpm = (avg_kps * 60) / 5

        # Calculate overall accuracy
        diverse_samples = user_df.dropna(subset=['accuracy'])
        avg_accuracy = diverse_samples['accuracy'].mean() if not diverse_samples.empty else None

        stats = {
            "username": username,
            "total_samples": len(user_df),
            "avg_wpm": wpm,
            "avg_hold_time_ms": user_df['mean_hold_time'].mean(),
            "avg_hold_std_ms": user_df['std_hold_time'].mean(),
            "avg_latency_ms": user_df['mean_pp_latency'].mean(),
            "avg_latency_std_ms": user_df['std_pp_latency'].mean(),
            "avg_accuracy": avg_accuracy
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 10. Helper Function (UPDATED) ---
def load_model_on_startup():
    global model, label_encoder, model_columns
    if os.path.exists(MODEL_FILE) and os.path.exists(DB_FILE):
        print("Loading existing model...")
        model = joblib.load(MODEL_FILE)
        df = pd.read_csv(DB_FILE)
        label_encoder.fit(df['username'])
        # UPDATED: Make sure we get all columns
        model_columns = [col for col in df.columns if col not in ['username', 'sample_type']]
        print(f"Model loaded. Features: {model_columns}")

# --- 11. Run the Server ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    load_model_on_startup()
    print(f"Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)