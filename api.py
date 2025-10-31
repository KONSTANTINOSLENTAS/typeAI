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

# --- 1. Configuration ---
app = Flask(__name__)
CORS(app)

# --- NEW: PostgreSQL Database Configuration ---
# Render will provide this URL. For local testing, it falls back to a local 'users.db'
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
# Fix for Render's PostgreSQL URL
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'default-fallback-secret-key') # Use env var

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

MODEL_FILE = "multi_user_model.joblib"
MIN_SAMPLES_TO_TRAIN = 10

model = None
label_encoder = LabelEncoder()
model_columns = []

# --- 2. Database Models ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    # Relationship: A user can have many samples
    samples = db.relationship('TypingSample', backref='user', lazy=True)

# NEW: Table for all typing data
class TypingSample(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    sample_type = db.Column(db.String(20)) # 'diverse' or 'free'
    accuracy = db.Column(db.Float, nullable=True) # Will be null for 'free'
    mean_hold_time = db.Column(db.Float)
    std_hold_time = db.Column(db.Float)
    mean_pp_latency = db.Column(db.Float)
    std_pp_latency = db.Column(db.Float)
    typing_speed_kps = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

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

# --- 4. API Routes ---
@app.route("/status", methods=["GET"])
def get_status():
    # Status now depends on if the model file exists (which it won't yet on a server)
    # A better check is if we have enough samples in the DB to train one
    try:
        model_ready = TypingSample.query.count() >= MIN_SAMPLES_TO_TRAIN
        return jsonify({"model_ready": model_ready})
    except Exception:
        return jsonify({"model_ready": False})


@app.route("/users", methods=["GET"])
def get_user_stats():
    try:
        # Query the database directly
        samples = db.session.query(TypingSample.username, db.func.count(TypingSample.id)).group_by(TypingSample.username).all()
        stats = [{"username": username, "samples": count} for username, count in samples]
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/signup", methods=["POST"])
def signup():
    # (This route is unchanged)
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
    # (This route is unchanged)
    data = request.json
    username, password = data.get('username'), data.get('password')
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        access_token = create_access_token(identity=user.username)
        return jsonify(access_token=access_token)
    return jsonify({"error": "Invalid username or password"}), 401

@app.route("/add_sample", methods=["POST"])
@jwt_required()
def add_sample():
    username = get_jwt_identity()
    user = User.query.filter_by(username=username).first()
    if not user:
        return jsonify({"error": "User not found"}), 404

    data = request.json
    events, accuracy, sample_type = data.get('events'), data.get('accuracy'), data.get('sample_type')
    if not events: return jsonify({"error": "Missing 'events' data"}), 400

    features = calculate_global_features(events)

    # Create new sample row in the DB
    new_sample = TypingSample(
        username=username,
        sample_type=sample_type,
        accuracy=accuracy if sample_type == 'diverse' else None,
        mean_hold_time=features.get('mean_hold_time'),
        std_hold_time=features.get('std_hold_time'),
        mean_pp_latency=features.get('mean_pp_latency'),
        std_pp_latency=features.get('std_pp_latency'),
        typing_speed_kps=features.get('typing_speed_kps'),
        user_id=user.id
    )
    db.session.add(new_sample)
    db.session.commit()
    return jsonify({"status": "sample added"})

@app.route("/train", methods=["POST"])
@jwt_required()
def train_model():
    global model, label_encoder, model_columns

    try:
        # Read all data from the SQL database into a pandas DataFrame
        query = db.session.query(TypingSample)
        df = pd.read_sql(query.statement, db.session.bind)
    except Exception as e:
        return jsonify({"error": f"Failed to read data from database: {e}"}), 500

    if len(df) < MIN_SAMPLES_TO_TRAIN:
        return jsonify({"error": f"Need at least {MIN_SAMPLES_TO_TRAIN} samples. You have {len(df)}."}), 400

    X = df.drop(columns=['id', 'user_id', 'username', 'sample_type'])
    y_labels = df['username']
    X['accuracy'] = X['accuracy'].fillna(1.0) # Fill free-text NaNs

    model_columns = X.columns.tolist()
    y = label_encoder.fit_transform(y_labels)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # We can't save the model to a file on Render's free tier.
    # So, we just keep it in memory. It will be lost on restart.
    # For a portfolio, this is fine. The model will just retrain
    # when a user clicks "Train" again.

    print(f"Model trained on {len(X)} samples for users: {label_encoder.classes_}")
    return jsonify({"status": "model trained", "users": label_encoder.classes_.tolist()})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None: # If server restarted, model is gone
        try:
            # Attempt to load data and retrain the model quickly
            query = db.session.query(TypingSample)
            df = pd.read_sql(query.statement, db.session.bind)
            if len(df) >= MIN_SAMPLES_TO_TRAIN:
                X = df.drop(columns=['id', 'user_id', 'username', 'sample_type'])
                y_labels = df['username']
                X['accuracy'] = X['accuracy'].fillna(1.0)
                global model_columns, label_encoder, model
                model_columns = X.columns.tolist()
                y = label_encoder.fit_transform(y_labels)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)
                print("Model re-trained on-the-fly.")
            else:
                return jsonify({"error": "Model is not trained."}), 400
        except Exception as e:
            return jsonify({"error": f"Failed to auto-train: {e}"}), 500

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
        # Query for users and their average accuracy
        query = db.session.query(
            TypingSample.username, 
            db.func.avg(TypingSample.accuracy)
        ).filter(TypingSample.sample_type == 'diverse').group_by(TypingSample.username).order_by(db.func.avg(TypingSample.accuracy).desc()).all()

        stats = [{"username": username, "avg_accuracy": avg_acc} for username, avg_acc in query if avg_acc is not None]
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/user_stats/<username>", methods=["GET"])
def get_user_detail_stats(username):
    try:
        # Query all samples for the user
        user_samples = TypingSample.query.filter(db.func.lower(TypingSample.username) == username.lower()).all()
        if not user_samples:
            return jsonify({"error": "User not found"}), 404

        # Use pandas to easily calculate stats from the query results
        df = pd.DataFrame([s.__dict__ for s in user_samples])

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

# --- 9. Run the Server ---
if __name__ == "__main__":
    # Create the database tables if they don't exist
    with app.app_context():
        db.create_all()

    print(f"Starting Flask server at http://127.0.0.1:5000")
    # Use Gunicorn-friendly host/port for Render
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))