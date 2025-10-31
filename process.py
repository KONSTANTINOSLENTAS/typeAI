import pandas as pd
import numpy as np
import json
import os

DATA_DIR = "keystroke_data"
OUTPUT_FILE = "features.csv" # The final training set

def process_file(filepath):
    """
    Processes a single session JSON file and calculates its 
    feature vector (mean and std for hold times and P-P latencies).
    """
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Load the raw event data into a pandas DataFrame
    df = pd.DataFrame(data)
    
    features = {}
    
    # --- 1. Hold Time (Duration) Features ---
    # Filter for 'release' events which contain the hold_time_ms
    release_events = df[df['event'] == 'release'].dropna(subset=['hold_time_ms'])
    
    # Group by key and calculate the mean and standard deviation of hold time
    # .fillna(0) is important: if a key is pressed only once, its std is NaN (Not a Number)
    hold_stats = release_events.groupby('key')['hold_time_ms'].agg(['mean', 'std']).fillna(0)
    
    # Flatten this data into our features dictionary
    for key, row in hold_stats.iterrows():
        features[f'H.mean.{key}'] = row['mean']
        features[f'H.std.{key}'] = row['std']
        
    # --- 2. Press-to-Press (P-P) Latency Features ---
    # Filter for 'press' events
    # We use .copy() to avoid a common pandas warning
    press_events = df[df['event'] == 'press'].copy() 
    
    # Calculate the time difference between consecutive key presses
    # This is the "Press-to-Press" (P-P) latency
    press_events['pp_latency'] = press_events['time_ms'].diff()
    
    # Get the previous key to create the digraph (e.g., 'T-h')
    press_events['prev_key'] = press_events['key'].shift(1)
    
    # .dropna() removes the very first key press, which has no 'prev_key' or 'pp_latency'
    press_events.dropna(subset=['prev_key', 'pp_latency'], inplace=True)
    
    # Create the digraph label (e.g., "T-h")
    press_events['digraph'] = press_events['prev_key'] + '-' + press_events['key']
    
    # Group by the digraph and get the mean/std of the P-P latency
    pp_stats = press_events.groupby('digraph')['pp_latency'].agg(['mean', 'std']).fillna(0)

    # Flatten this data into our features dictionary
    for digraph, row in pp_stats.iterrows():
        features[f'P-P.mean.{digraph}'] = row['mean']
        features[f'P-P.std.{digraph}'] = row['std']
        
    return features

# --- Main execution ---
def main():
    all_features = [] # This list will hold one feature dictionary per file
    
    print(f"Looking for data in '{DATA_DIR}' directory...")
    
    # Loop through every file in the data directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            print(f"Processing {filename}...")
            
            try:
                # Calculate the features for this one file
                features = process_file(filepath)
                
                # Add the filename for reference
                features['source_file'] = filename
                
                all_features.append(features)
            except Exception as e:
                print(f"  Could not process {filename}: {e}")

    if not all_features:
        print("\nNo JSON files found to process.")
        print("Please run 'capture.py' a few times to create some data.")
        return

    # Convert the list of dictionaries into a single, large DataFrame
    # .fillna(0) is CRITICAL. It ensures that if one session missed a key
    # (e.g., no 'a-b' digraph), it gets a 0 instead of NaN.
    final_df = pd.DataFrame(all_features).fillna(0)
    
    # Reorder columns to put 'source_file' first (optional, but cleaner)
    cols = ['source_file'] + [col for col in final_df if col != 'source_file']
    final_df = final_df[cols]
    
    # Save the final, complete dataset to a CSV file
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n--- Feature extraction complete! ---")
    print(f"Processed {len(all_features)} files.")
    print(f"Training data table saved to: {OUTPUT_FILE}")
    print("\n--- Sample of Your New Data (first 5 rows) ---")
    print(final_df.head())

if __name__ == "__main__":
    main()