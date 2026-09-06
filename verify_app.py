# verify_app.py

import os
import json
import joblib
import pandas as pd

def verify():
    print("[Verify] Checking precomputed files...")
    
    files = [
        "processed_data/balls_df.parquet",
        "processed_data/matches_df.csv",
        "processed_data/bat_wide.csv",
        "processed_data/bowl_wide.csv",
        "processed_data/venue_stats.csv",
        "processed_data/match_ups.csv",
        "processed_data/model_rf.joblib",
        "processed_data/model_win.joblib",
        "processed_data/classifier_career.joblib",
        "processed_data/classifier_phase.joblib",
        "processed_data/classifier_metrics.json"
    ]
    
    missing_files = []
    for f in files:
        if not os.path.exists(f):
            missing_files.append(f)
            
    if missing_files:
        print(f"[Verify] Missing files: {missing_files}")
        return False
        
    print("[Verify] All precomputed files exist.")
    
    # Test loading
    try:
        balls = pd.read_parquet("processed_data/balls_df.parquet")
        matches = pd.read_csv("processed_data/matches_df.csv")
        bat_wide = pd.read_csv("processed_data/bat_wide.csv")
        bowl_wide = pd.read_csv("processed_data/bowl_wide.csv")
        venue_stats = pd.read_csv("processed_data/venue_stats.csv")
        matchups = pd.read_csv("processed_data/match_ups.csv")
        
        with open("processed_data/classifier_metrics.json") as f:
            metrics = json.load(f)
            
        model_rf = joblib.load("processed_data/model_rf.joblib")
        model_win = joblib.load("processed_data/model_win.joblib")
        clf_career = joblib.load("processed_data/classifier_career.joblib")
        clf_phase = joblib.load("processed_data/classifier_phase.joblib")
        
        print("[Verify] All data and models loaded successfully!")
        print(f"[Verify] Total deliveries: {len(balls):,}")
        print(f"[Verify] Total matches: {len(matches):,}")
        print(f"[Verify] Total batsmen: {len(bat_wide)}")
        print(f"[Verify] Total bowlers: {len(bowl_wide)}")
        print(f"[Verify] Classifier metrics: Accuracy: {metrics['phase']['accuracy']}, Recall: {metrics['phase']['recall']}")
        
        return True
    except Exception as e:
        print(f"[Verify] Error loading data or models: {e}")
        return False

if __name__ == '__main__':
    verify()
