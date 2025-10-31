import pandas as pd
import joblib
import os
from sklearn.svm import OneClassSVM

# --- Configuration ---
FEATURES_FILE = "features.csv"
MODEL_FILE = "typing_model.joblib" # The final, saved AI model

def main():
    # --- 1. Load the Training Data ---
    print(f"Loading training data from {FEATURES_FILE}...")
    try:
        df = pd.read_csv(FEATURES_FILE)
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"File not found: '{FEATURES_FILE}'")
        print("Please run 'process.py' first to create your feature file.")
        return

    if df.empty or len(df) < 2:
        print(f"\n--- ERROR ---")
        print(f"Your '{FEATURES_FILE}' file doesn't have enough data.")
        print("Please run 'capture.py' at least 5-10 times, then run 'process.py' again.")
        return

    # Prepare data for Scikit-learn:
    # We drop 'source_file' because it's a label, not a numeric feature
    X_train = df.drop(columns=['source_file'])

    # --- 2. Train the AI Model ---
    print(f"Training AI model (One-Class SVM) on {len(X_train)} samples...")

    # We use a One-Class SVM. This is an "anomaly detection" model.
    # It learns the "shape" of your normal typing pattern.
    #
    # Key Parameters:
    # kernel='rbf': The best kernel for complex, non-linear patterns.
    # nu=0.2: This is the most important setting. It's an "upper bound"
    #         on the fraction of training errors. Setting it to 0.2
    #         gives the model flexibility, assuming ~20% of your
    #         samples might be a bit "weird" or different.
    # gamma='scale': A smart default that adjusts to your data.
    
    model = OneClassSVM(nu=0.2, kernel='rbf', gamma='scale')
    
    # Train the model on your feature data
    model.fit(X_train)
    
    print("Model training complete.")

    # --- 3. Save the Trained Model ---
    print(f"Saving trained model to {MODEL_FILE}...")
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved successfully!")
    
    # --- 4. Test the Model on Your Data ---
    # Let's see what the model thinks of the data it just trained on.
    # It will output:
    #  1 = "Inlier" (This is you)
    # -1 = "Outlier" (This is an anomaly / not you)
    
    print("\n--- Testing model on your training data ---")
    predictions = model.predict(X_train)
    
    # Create a nice report
    results = pd.DataFrame({
        'source_file': df['source_file'],
        'prediction': predictions
    })
    
    # Map the 1/-1 to human-readable labels
    results['prediction_label'] = results['prediction'].map({
        1: "This looks like YOU",
        -1: "This looks like an ANOMALY"
    })
    
    print(results[['source_file', 'prediction_label']])
    
    # Print a summary
    inliers = (predictions == 1).sum()
    outliers = (predictions == -1).sum()
    
    print(f"\nSummary: {inliers} samples identified as you, {outliers} samples flagged as anomalies.")
    if outliers > 0:
        print("This is normal! The model is learning to be strict.")

if __name__ == "__main__":
    main()