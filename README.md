# 🏏 IPL Phase-Specialist Engine Dashboard
### Universität des Saarlandes | Sports Analytics Course Project

An interactive, machine-learning-powered sports analytics dashboard that demonstrates why **phase-wise segregation** is superior to overall career averages for player scouting and matchup optimization in T20 Cricket.

This project is built using Cricsheet ball-by-ball data and features custom parsing pipelines, regression predictors, unsupervised player clustering, win-probability estimators, and side-by-side binary classifiers to mathematically prove the value of phase segregation.

By: Nareshsarathy Sambath(7089589), Gauhar Ayub Khan(7076413)

There are few data files which weren't be able to upload due to size truncating in GitHub. There are few data files missing in the data folder. And there another config file which is available, download the file and add it to a folder named .streamlit and you can run the file.

---

## 📂 Project Structure

```text
moneyball_cricket/
├── data/                       # Raw Cricsheet T20 JSON match files
├── processed_data/             # Cached compiled Parquets, CSVs & Joblib models
│   ├── balls_df.parquet        # Compressed delivery-level dataframe
│   ├── matches_df.csv          # Match metadata (venues, toss, seasons, winners)
│   ├── bat_wide.csv            # Wide format batting phase stats
│   ├── bowl_wide.csv           # Wide format bowling phase stats
│   ├── venue_stats.csv         # Phase-wise metrics per stadium
│   ├── match_ups.csv           # Head-to-head records between all batters/bowlers
│   ├── model_rf.joblib         # Random Forest regressor (predicts Death SR)
│   ├── model_win.joblib        # Logistic Regression (predicts Win Probability)
│   ├── classifier_career.joblib# Baseline classifier (trained on Career SR only)
│   ├── classifier_phase.joblib # Proposed classifier (trained on Phase-wise SRs)
│   └── classifier_metrics.json # Precision, Recall, and Confusion Matrices metrics
├── src/
│   └── pipeline.py             # Precomputes raw JSON files, generates CSVs & trains models
├── app.py                      # Interactive Streamlit dashboard web application
├── requirements.txt            # Python dependencies (Streamlit, Plotly, PyArrow, scikit-learn)
└── README.md                   # Project documentation
```

---

## ⚡ How to Setup and Run From Scratch

Follow these 4 steps to set up, precompute the data, and run the dashboard locally.

### Step 1: Install Dependencies
Open your terminal/command prompt and run the following command to install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Download Cricsheet JSON Data
Ensure your ball-by-ball IPL JSON match files are placed inside the `data/` directory. The pipeline expects files named as standard integers (e.g. `1082591.json`).

### Step 3: Run the Precomputation Pipeline
Before running the dashboard, execute the precomputation pipeline script. This script parses all JSON files, computes direct player matchups, stadium conditions, and trains the machine learning models (saving everything to `processed_data/` so the app loads instantly):
```bash
python src/pipeline.py
```
*(This will print parsing progress and save trained `.joblib` and `.csv` assets on success).*

### Step 4: Run the Streamlit Dashboard
Launch the web server locally:
```bash
streamlit run app.py
```
Once run, your browser will open the interactive dashboard automatically at **[http://localhost:8501](http://localhost:8501)**.

---

## 📊 Core Features & Analytics Sections

### 1. Presentation Background & Context (Tab 1)
- Recreates academic slides detailing T20 playing conditions (ICC fielding restrictions: max 2 fielders outside circle in overs 1-6, vs. 5 fielders outside in overs 7-20).
- **The Core Problem**: Visualizes career strike rate vs. death strike rate in an interactive Plotly scatter plot with a $Y=X$ line. It shows how career averages wash out phase specialization.

### 2. Interactive Player Explorer (Tab 2)
- Select any player (batsman or bowler).
- Generates an interactive **Plotly Radar Chart** comparing the player's phase metrics against the tournament average.
- Calculates and lists their **Composite Specialist Score** and **Phase Gap** (Best vs. Worst phase).

### 3. Dynamic Leaderboards (Tab 3)
- Ranks top batters (by Strike Rate) and bowlers (by Economy) for each phase (Powerplay, Middle, Death).
- Interactive sliders let users adjust the **minimum legal balls faced/bowled** (outlier filters) and toggle between **All Seasons** and **Last 4 Seasons** (active squad view).

### 4. Venue-Specific Phase Analyzer (Tab 4)
- Characterizes stadium pitches into *Batting Paradises* (high run rates, small boundaries), *Slower/Turning Pitches* (spinner friendly, low run rates), and *Balanced Tracks*.
- Charts phase run rates and boundary percentages for different venues.

### 5. Bowler-vs-Batsman Phase matchup Simulator (Tab 5)
- Select a batter and a bowler to view their direct head-to-head records.
- Uses a **Bayesian shrinkage model** (combining direct head-to-head records with general phase strengths and tournament baselines) to predict the outcome probability of their face-off (Dot %, Single %, Boundary %, Wicket %) in a chosen phase.

### 6. Era Progression Timeline (Tab 6)
- Evaluates how T20 batting intent has evolved from 2008 to 2026, mapping how rules like the Impact Player rule pushed run rates to historical highs.

### 7. Classifier Validation (Tab 7)
- Displays side-by-side **Confusion Matrices** comparing Model A (baseline career only classifier) vs. Model B (proposed Phase-wise classifier) in identifying death finishers.
- Mathematically proves that the career average baseline suffers from high **False Negatives** (low recall), while phase-wise analytics correctly captures specialized hitters.

### 8. Machine Learning Playground (Tab 8)
- **Death SR Predictor**: Random Forest model predicting a batter's Death SR based on other phases. Input hypothetical stats to get predicted values.
- **PCA Cluster Map**: 2D scatter plot of K-Means clusters representing player archetypes (*Powerplay Strikers*, *Death Finishers*, *All-Phase Dominators*, *Anchors*).
- **Win Probability**: Simulates the win percentage of Team 1 based on relative team strength and toss decisions.
