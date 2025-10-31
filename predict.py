import time
import pandas as pd
import numpy as np
import joblib
import os
from pynput import keyboard

# --- Configuration ---
TARGET_PHRASE = "The quick brown fox jumps over the lazy dog."
MODEL_FILE = "typing_model.joblib"
FEATURES_FILE = "features.csv" # We need this to get the column order

# --- Global variables for capturing ---
session_events = []
key_press_times = {}
start_time = None

# --- Capture Functions (from capture.py) ---

def on_press(key):
    global start_time, session_events, key_press_times
    if start_time is None:
        start_time = time.perf_counter()
    current_time = time.perf_counter()
    relative_time_ms = (current_time - start_time) * 1000
    try:
        key_char = key.char
    except AttributeError:
        key_char = str(key)
    event_data = {"event": "press", "key": key_char, "time_ms": relative_time_ms}
    session_events.append(event_data)
    if key not in key_press_times:
        key_press_times[key] = current_time
    # Feedback for typing
    if key == keyboard.Key.space:
        print(' ', end='', flush=True)
    elif key == keyboard.Key.backspace:
        print('\b \b', end='', flush=True) 
    elif key_char is not None and len(key_char) == 1:
        print(key_char, end='', flush=True)

def on_release(key):
    global start_time, session_events, key_press_times
    if start_time is None: return
    current_time = time.perf_counter()
    relative_time_ms = (current_time - start_time) * 1000
    try:
        key_char = key.char
    except AttributeError:
        key_char = str(key)
    hold_time_ms = None
    if key in key_press_times:
        hold_time_ms = (current_time - key_press_times[key]) * 1000
        del key_press_times[key]
    event_data = {
        "event": "release",
        "key": key_char,
        "time_ms": relative_time_ms,
        "hold_time_ms": hold_time_ms
    }
    session_events.append(event_data)
    if key == keyboard.Key.enter:
        print("\n--- Capturing Complete ---")
        return False  # Stops the listener

# --- Processing Function (from process.py) ---
def process_live_events(events_list):
    """
    Processes a live list of event dictionaries into a feature dictionary.
    """
    df = pd.DataFrame(events_list)
    features = {}
    
    # 1. Hold Time Features
    release_events = df[df['event'] == 'release'].dropna(subset=['hold_time_ms'])
    hold_stats = release_events.groupby('key')['hold_time_ms'].agg(['mean', 'std']).fillna(0)
    for key, row in hold_stats.iterrows():
        features[f'H.mean.{key}'] = row['mean']
        features[f'H.std.{key}'] = row['std']
        
    # 2. Press-to-Press (P-P) Latency Features
    press_events = df[df['event'] == 'press'].copy()
    press_events['pp_latency'] = press_events['time_ms'].diff()
    press_events['prev_key'] = press_events['key'].shift(1)
    press_events.dropna(subset=['prev_key', 'pp_latency'], inplace=True)
    press_events['digraph'] = press_events['prev_key'] + '-' + press_events['key']
    pp_stats = press_events.groupby('digraph')['pp_latency'].agg(['mean', 'std']).fillna(0)
    for digraph, row in pp_stats.iterrows():
        features[f'P-P.mean.{digraph}'] = row['mean']
        features[f'P-P.std.{digraph}'] = row['std']
        
    return features

# --- Main Prediction Logic ---
def main():
    # --- 1. Load the Model and Feature Columns ---
    print(f"Loading your saved AI model from {MODEL_FILE}...")
    try:
        model = joblib.load(MODEL_FILE)
        # We MUST load the original columns to ensure the new data has the same "shape"
        train_df = pd.read_csv(FEATURES_FILE)
        # Get the list of all feature columns the model was trained on
        model_columns = train_df.drop(columns=['source_file'], errors='ignore').columns
    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(f"Could not load {e.filename}.")
        print("Please run 'train.py' to create the model file first.")
        return
    except Exception as e:
        print(f"An error occurred loading the model: {e}")
        return
        
    print("Model loaded.")
    
    # --- 2. Capture a new typing sample ---
    print("\n--- User Authentication ---")
    print("Please type the following phrase to authenticate:")
    print(f"'{TARGET_PHRASE}'")
    print("\n>", end=' ', flush=True)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
        
    if not session_events:
        print("No typing detected. Exiting.")
        return

    # --- 3. Process the new sample ---
    print("Processing your typing pattern...")
    live_features_dict = process_live_events(session_events)
    
    # Convert the single dictionary of features into a 1-row DataFrame
    live_df = pd.DataFrame([live_features_dict])
    
    # *** THIS IS THE MOST IMPORTANT STEP ***
    # Reindex the live data to match the model's training columns.
    # This adds all the columns the model expects (e.g., 'P-P.mean.x-y')
    # and fills them with 0 if you didn't type that digraph.
    live_features_prepared = live_df.reindex(columns=model_columns).fillna(0)
    
    # --- 4. Make the Prediction ---
    print("Authenticating...")
    
    # The model predicts: [1] for inlier (you) or [-1] for outlier (not you)
    prediction = model.predict(live_features_prepared)
    
    # --- 5. Show the Result ---
    if prediction[0] == 1:
        print("\n" + "="*30)
        print("  ✅ AUTHENTICATION SUCCESSFUL")
        print("   Welcome back, user!")
        print("="*30)
    else:
        print("\n" + "!"*30)
        print("  ❌ AUTHENTICATION FAILED")
        print("   Typing pattern does not match.")
        print("!"*30)

if __name__ == "__main__":
    main()