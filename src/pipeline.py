# src/pipeline.py

import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

warnings.filterwarnings('ignore')

# Config parameters (minimum balls faced per phase for outlier filtering)
class CFG:
    min_bat_balls = {'Powerplay': 222, 'Middle': 294, 'Death': 162}
    min_bowl_balls = {'Powerplay': 210, 'Middle': 288, 'Death': 144}
    
    # Weights for composite specialist score
    bat_weights = {'pp': 0.35, 'mid': 0.30, 'dth': 0.35}
    bowl_weights = {'pp': 0.30, 'mid': 0.25, 'dth': 0.45}

def parse_match(fpath):
    """
    Parse one Cricsheet JSON match file into ball-by-ball deliveries
    and match-level metadata.
    """
    with open(fpath, encoding='utf-8') as f:
        d = json.load(f)

    info = d['info']
    match_id = os.path.splitext(os.path.basename(fpath))[0]
    season = info.get('season', None)
    date = info['dates'][0] if info.get('dates') else None
    venue = info.get('venue', '')
    teams = info.get('teams', [None, None])
    outcome = info.get('outcome', {})
    toss = info.get('toss', {})
    registry = info.get('registry', {}).get('people', {})

    winner = outcome.get('winner', None)
    by_dict = outcome.get('by', {})
    margin_type = list(by_dict.keys())[0] if by_dict else None
    margin_val = list(by_dict.values())[0] if by_dict else None
    potm_list = info.get('player_of_match', [])
    potm = potm_list[0] if potm_list else None
    toss_winner = toss.get('winner')
    toss_decision = toss.get('decision')

    match_row = dict(
        match_id=match_id, season=season, date=date, venue=venue,
        team1=teams[0], team2=teams[1], winner=winner,
        margin_type=margin_type, margin_val=margin_val,
        toss_winner=toss_winner, toss_decision=toss_decision, potm=potm,
    )

    ball_rows = []
    for inn_i, inn in enumerate(d.get('innings', [])):
        batting_team = inn.get('team')
        bowling_team = next((t for t in teams if t != batting_team), None)
        innings_num = inn_i + 1

        for ov in inn.get('overs', []):
            over_num = ov['over']  # 0-indexed
            phase = ('Powerplay' if over_num <= 5 else
                     'Middle' if over_num <= 14 else
                     'Death')

            for ball in ov.get('deliveries', []):
                batter = ball['batter']
                bowler = ball['bowler']
                non_striker = ball['non_striker']
                runs_bat = ball['runs']['batter']
                runs_extra = ball['runs']['extras']
                runs_total = ball['runs']['total']

                extras = ball.get('extras', {})
                is_wide = 'wides' in extras
                is_noball = 'noballs' in extras
                is_bye = 'byes' in extras
                is_legbye = 'legbyes' in extras
                is_legal = not (is_wide or is_noball)

                is_dot = (runs_total == 0 and is_legal)
                is_four = (runs_bat == 4 and not is_wide)
                is_six = (runs_bat == 6)

                wickets = ball.get('wickets', [])
                is_wicket = len(wickets) > 0
                wicket_kind = wickets[0]['kind'] if wickets else None
                player_out = wickets[0].get('player_out', batter) if wickets else None
                fielders = wickets[0].get('fielders', []) if wickets else []
                fielder = fielders[0]['name'] if fielders else None
                bowler_wkt = is_wicket and wicket_kind not in (
                    'run out', 'retired hurt', 'retired out', 'obstructing the field')

                ball_rows.append(dict(
                    match_id=match_id, season=season, date=date,
                    venue=venue, innings=innings_num,
                    batting_team=batting_team, bowling_team=bowling_team,
                    over=over_num, phase=phase,
                    batter=batter, bowler=bowler, non_striker=non_striker,
                    runs_bat=runs_bat, runs_extra=runs_extra, runs_total=runs_total,
                    is_wide=is_wide, is_noball=is_noball,
                    is_bye=is_bye, is_legbye=is_legbye,
                    is_legal=is_legal, is_dot=is_dot,
                    is_four=is_four, is_six=is_six,
                    is_wicket=is_wicket, wicket_kind=wicket_kind,
                    player_out=player_out, bowler_wkt=bowler_wkt,
                    fielder=fielder, winner=winner, toss_winner=toss_winner,
                    toss_decision=toss_decision,
                ))

    return ball_rows, match_row

def build_phase_tables(balls_df):
    """
    Computes player aggregates, normalizes them, and calculates
    composite performance scores and phase gaps.
    """
    legal = balls_df[balls_df['is_legal']].copy()

    # Career aggregates
    career_bat = (legal.groupby('batter')
                  .agg(runs=('runs_bat', 'sum'), balls=('is_legal', 'sum'),
                       fours=('is_four', 'sum'), sixes=('is_six', 'sum'),
                       matches=('match_id', 'nunique')).reset_index())
    career_bat['career_sr'] = (career_bat['runs'] / career_bat['balls'] * 100).round(2)

    career_bowl = (legal.groupby('bowler')
                   .agg(runs_c=('runs_total', 'sum'), balls=('is_legal', 'sum'),
                        wickets=('bowler_wkt', 'sum'), matches=('match_id', 'nunique')).reset_index())
    career_bowl['career_econ'] = (career_bowl['runs_c'] / career_bowl['balls'] * 6).round(2)

    # Batting phase aggregates
    bat_phase = (legal.groupby(['batter', 'phase'])
                 .agg(runs=('runs_bat', 'sum'), balls=('is_legal', 'sum'),
                      fours=('is_four', 'sum'), sixes=('is_six', 'sum'),
                      dots=('is_dot', 'sum'), innings=('match_id', 'nunique')).reset_index())
    
    dismissals = (balls_df[balls_df['is_wicket'] & balls_df['player_out'].notna()]
                  .groupby(['player_out', 'phase']).size().reset_index(name='dismissals')
                  .rename(columns={'player_out': 'batter'}))
    
    bat_phase = bat_phase.merge(dismissals, on=['batter', 'phase'], how='left')
    bat_phase['dismissals'] = bat_phase['dismissals'].fillna(0)
    bat_phase['strike_rate'] = (bat_phase['runs'] / bat_phase['balls'] * 100).round(2)
    bat_phase['boundary_pct'] = ((bat_phase['fours'] * 4 + bat_phase['sixes'] * 6) /
                                 bat_phase['runs'].clip(lower=1) * 100).round(1)

    # Bowling phase aggregates
    bowl_phase = (legal.groupby(['bowler', 'phase'])
                  .agg(runs_c=('runs_total', 'sum'), balls=('is_legal', 'sum'),
                       wickets=('bowler_wkt', 'sum'), dots=('is_dot', 'sum'),
                       innings=('match_id', 'nunique')).reset_index())
    bowl_phase['economy'] = (bowl_phase['runs_c'] / bowl_phase['balls'] * 6).round(2)
    bowl_phase['dot_pct'] = (bowl_phase['dots'] / bowl_phase['balls'] * 100).round(1)

    # Apply outlier minimum legal-balls filters
    bat_filtered = bat_phase[bat_phase.apply(
        lambda r: r['balls'] >= CFG.min_bat_balls.get(r['phase'], 0), axis=1)].copy()
    bowl_filtered = bowl_phase[bowl_phase.apply(
        lambda r: r['balls'] >= CFG.min_bowl_balls.get(r['phase'], 0), axis=1)].copy()

    # Pivot batting to wide format
    bat_wide = (bat_filtered.pivot_table(index='batter', columns='phase',
                                         values='strike_rate', aggfunc='first').reset_index())
    bat_wide.columns.name = None
    bat_wide = bat_wide.rename(columns={'Powerplay': 'sr_pp', 'Middle': 'sr_mid', 'Death': 'sr_dth'})
    for c in ['sr_pp', 'sr_mid', 'sr_dth']:
        if c not in bat_wide.columns:
            bat_wide[c] = np.nan
    bat_wide = bat_wide.merge(career_bat[['batter', 'runs', 'career_sr', 'matches']],
                              on='batter', how='left')

    # Pivot bowling to wide format
    bowl_wide = (bowl_filtered.pivot_table(index='bowler', columns='phase',
                                           values='economy', aggfunc='first').reset_index())
    bowl_wide.columns.name = None
    bowl_wide = bowl_wide.rename(columns={'Powerplay': 'econ_pp', 'Middle': 'econ_mid', 'Death': 'econ_dth'})
    for c in ['econ_pp', 'econ_mid', 'econ_dth']:
        if c not in bowl_wide.columns:
            bowl_wide[c] = np.nan
    bowl_wide = bowl_wide.merge(career_bowl[['bowler', 'wickets', 'career_econ', 'matches']],
                                on='bowler', how='left')

    # Normalization (0-100)
    scaler = MinMaxScaler(feature_range=(0, 100))
    for col in ['sr_pp', 'sr_mid', 'sr_dth', 'career_sr']:
        valid = bat_wide[col].notna()
        if valid.sum() >= 2:
            bat_wide.loc[valid, f'{col}_n'] = scaler.fit_transform(bat_wide.loc[valid, [col]])
            
    for col in ['econ_pp', 'econ_mid', 'econ_dth', 'career_econ']:
        valid = bowl_wide[col].notna()
        if valid.sum() >= 2:
            inv = -bowl_wide.loc[valid, col]
            bowl_wide.loc[valid, f'{col}_n'] = scaler.fit_transform(inv.values.reshape(-1, 1))

    # Composite scores
    def bat_perf(row):
        car = row.get('career_sr_n', 50.0); car = 50.0 if pd.isna(car) else car
        pp  = row.get('sr_pp_n',  np.nan); pp  = car if pd.isna(pp)  else pp
        mid = row.get('sr_mid_n', np.nan); mid = car if pd.isna(mid) else mid
        dth = row.get('sr_dth_n', np.nan); dth = car if pd.isna(dth) else dth
        return round(CFG.bat_weights['pp'] * pp + CFG.bat_weights['mid'] * mid + CFG.bat_weights['dth'] * dth, 2)

    def bowl_perf(row):
        car = row.get('career_econ_n', 50.0); car = 50.0 if pd.isna(car) else car
        pp  = row.get('econ_pp_n',  np.nan); pp  = car if pd.isna(pp)  else pp
        mid = row.get('econ_mid_n', np.nan); mid = car if pd.isna(mid) else mid
        dth = row.get('econ_dth_n', np.nan); dth = car if pd.isna(dth) else dth
        return round(CFG.bowl_weights['pp'] * pp + CFG.bowl_weights['mid'] * mid + CFG.bowl_weights['dth'] * dth, 2)

    bat_wide['perf_score'] = bat_wide.apply(bat_perf, axis=1)
    bowl_wide['perf_score'] = bowl_wide.apply(bowl_perf, axis=1)

    # Phase gaps
    bat_wide['sr_best'] = bat_wide[['sr_pp', 'sr_mid', 'sr_dth']].max(axis=1)
    bat_wide['sr_worst'] = bat_wide[['sr_pp', 'sr_mid', 'sr_dth']].min(axis=1)
    bat_wide['phase_gap'] = (bat_wide['sr_best'] - bat_wide['sr_worst']).round(1)
    
    bowl_wide['econ_best'] = bowl_wide[['econ_pp', 'econ_mid', 'econ_dth']].min(axis=1)
    bowl_wide['econ_worst'] = bowl_wide[['econ_pp', 'econ_mid', 'econ_dth']].max(axis=1)
    bowl_wide['phase_gap'] = (bowl_wide['econ_worst'] - bowl_wide['econ_best']).round(1)

    return bat_wide, bowl_wide, bat_filtered, bowl_filtered

def run_pipeline():
    print("[Pipeline] Starting Precomputation Pipeline...")
    os.makedirs("processed_data", exist_ok=True)

    DATA_DIR = 'data'
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))
    print(f"Found {len(json_files)} raw match files in '{DATA_DIR}'")

    all_balls = []
    all_matches = []

    for fpath in tqdm(json_files, desc='Parsing JSON files'):
        try:
            balls, match = parse_match(fpath)
            all_balls.extend(balls)
            all_matches.append(match)
        except Exception as e:
            print(f"  [Error] Error parsing {os.path.basename(fpath)}: {e}")

    balls_df = pd.DataFrame(all_balls)
    matches_df = pd.DataFrame(all_matches)

    # Clean date and seasons
    balls_df['date'] = pd.to_datetime(balls_df['date'], errors='coerce')
    balls_df['season'] = pd.to_numeric(balls_df['season'], errors='coerce').astype('Int64')
    matches_df['season'] = pd.to_numeric(matches_df['season'], errors='coerce').astype('Int64')

    # Save compile-ready parquet/csv files
    balls_df.to_parquet("processed_data/balls_df.parquet", index=False)
    matches_df.to_csv("processed_data/matches_df.csv", index=False)
    print("[Pipeline] Parsed deliveries and matches saved.")

    # Build phase wide structures
    bat_wide, bowl_wide, bat_filtered, bowl_filtered = build_phase_tables(balls_df)
    bat_wide.to_csv("processed_data/bat_wide.csv", index=False)
    bowl_wide.to_csv("processed_data/bowl_wide.csv", index=False)
    print("[Pipeline] Phase pivot structures generated.")

    # Venue Stats Precomputation
    # Grouping by stadium and phase to compile pitching metrics
    legal_balls = balls_df[balls_df['is_legal']].copy()
    venue_phase = (legal_balls.groupby(['venue', 'phase'])
                   .agg(runs_c=('runs_total', 'sum'),
                        balls=('is_legal', 'sum'),
                        fours=('is_four', 'sum'),
                        sixes=('is_six', 'sum'),
                        wickets=('is_wicket', 'sum'))
                   .reset_index())
    venue_phase['run_rate'] = (venue_phase['runs_c'] / venue_phase['balls'] * 6).round(2)
    venue_phase['boundary_rate'] = (((venue_phase['fours'] + venue_phase['sixes']) / venue_phase['balls']) * 100).round(2)

    # Pitch characterizations
    venues_mapped = []
    for venue, group in venue_phase.groupby('venue'):
        # Aggregate run rate in Powerplay & Death
        pp_rr = group[group['phase'] == 'Powerplay']['run_rate'].values
        mid_rr = group[group['phase'] == 'Middle']['run_rate'].values
        pp_rr_val = pp_rr[0] if len(pp_rr) > 0 else 7.5
        mid_rr_val = mid_rr[0] if len(mid_rr) > 0 else 7.0
        
        if pp_rr_val >= 8.2 or mid_rr_val >= 7.8:
            pitch_type = "🏏 Batting Paradise (Fast outfield, small boundaries)"
        elif pp_rr_val <= 7.2 or mid_rr_val <= 6.8:
            pitch_type = "🌀 Slower Pitch (Spinner friendly, low bounce)"
        else:
            pitch_type = "⚖️ Balanced Pitch (Standard bounce & boundaries)"
        
        for _, row in group.iterrows():
            venues_mapped.append({
                'venue': row['venue'],
                'phase': row['phase'],
                'runs_c': row['runs_c'],
                'balls': row['balls'],
                'run_rate': row['run_rate'],
                'boundary_rate': row['boundary_rate'],
                'wickets': row['wickets'],
                'pitch_type': pitch_type
            })
    pd.DataFrame(venues_mapped).to_csv("processed_data/venue_stats.csv", index=False)
    print("[Pipeline] Venue-specific phase stats completed.")

    # Matchups Precomputation
    # Computes direct batsman-bowler head-to-head records
    matchups = (legal_balls.groupby(['batter', 'bowler'])
                .agg(runs=('runs_bat', 'sum'),
                     balls=('is_legal', 'sum'),
                     fours=('is_four', 'sum'),
                     sixes=('is_six', 'sum'),
                     dismissals=('bowler_wkt', 'sum'))
                .reset_index())
    matchups.to_csv("processed_data/match_ups.csv", index=False)
    print("[Pipeline] Direct head-to-head matchups saved.")

    # Model Training 1: Death Over SR Predictor
    ml_bat = bat_wide.dropna(subset=['sr_pp', 'sr_mid', 'sr_dth', 'career_sr']).copy()
    ml_bat['log_runs'] = np.log1p(ml_bat['runs'])
    X = ml_bat[['sr_pp', 'sr_mid', 'career_sr', 'log_runs', 'matches']]
    y = ml_bat['sr_dth']

    rf_reg = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf_reg.fit(X, y)
    joblib.dump(rf_reg, "processed_data/model_rf.joblib")
    print("[Pipeline] Random Forest Death Over SR regressor saved.")

    # Model Training 2: Win Probability Classifier
    matches_ml = matches_df.dropna(subset=['winner', 'toss_decision']).copy()
    matches_ml['team1_wins'] = (matches_ml['winner'] == matches_ml['team1']).astype(int)
    matches_ml['toss_bat'] = (matches_ml['toss_decision'] == 'bat').astype(int)
    matches_ml['toss_is_t1'] = (matches_ml['toss_winner'] == matches_ml['team1']).astype(int)

    # Average run rate strength proxy
    team_rr = (balls_df[balls_df['is_legal']]
               .groupby(['match_id', 'batting_team'])
               .agg(runs=('runs_total', 'sum'), balls=('is_legal', 'sum'))
               .reset_index())
    team_rr['rr'] = team_rr['runs'] / team_rr['balls'] * 6
    avg_rr = team_rr.groupby('batting_team')['rr'].mean().reset_index(name='avg_rr')
    matches_ml = matches_ml.merge(avg_rr.rename(columns={'batting_team': 'team1', 'avg_rr': 't1_rr'}), on='team1', how='left')
    matches_ml = matches_ml.merge(avg_rr.rename(columns={'batting_team': 'team2', 'avg_rr': 't2_rr'}), on='team2', how='left')
    matches_ml['rr_diff'] = (matches_ml['t1_rr'] - matches_ml['t2_rr']).fillna(0)

    X_w = matches_ml[['toss_bat', 'toss_is_t1', 'rr_diff']].fillna(0)
    y_w = matches_ml['team1_wins']
    
    log_win = LogisticRegression()
    log_win.fit(X_w, y_w)
    joblib.dump(log_win, "processed_data/model_win.joblib")
    print("[Pipeline] Win probability classifier saved.")

    # Model Training 3: Career vs. Phase-wise Specialist Classifier
    # target: player Death SR is in top 25% of all qualified death hitters
    death_threshold = ml_bat['sr_dth'].quantile(0.75)
    ml_bat['is_death_specialist'] = (ml_bat['sr_dth'] >= death_threshold).astype(int)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        ml_bat, ml_bat['is_death_specialist'], test_size=0.3, random_state=42
    )
    
    # Baseline: Career SR features
    feats_career = ['career_sr', 'runs']
    clf_career = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf_career.fit(X_train[feats_career], y_train)
    
    # Proposed: Phase-wise features
    feats_phase = ['sr_pp', 'sr_mid', 'runs']
    clf_phase = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf_phase.fit(X_train[feats_phase], y_train)
    
    # Save models
    joblib.dump(clf_career, "processed_data/classifier_career.joblib")
    joblib.dump(clf_phase, "processed_data/classifier_phase.joblib")
    
    # Calculate confusion matrix & metrics
    pred_c = clf_career.predict(X_test[feats_career])
    pred_p = clf_phase.predict(X_test[feats_phase])
    
    acc_c = accuracy_score(y_test, pred_c)
    acc_p = accuracy_score(y_test, pred_p)
    
    p_c, r_c, f_c, _ = precision_recall_fscore_support(y_test, pred_c, average='binary')
    p_p, r_p, f_p, _ = precision_recall_fscore_support(y_test, pred_p, average='binary')
    
    cm_c = confusion_matrix(y_test, pred_c).tolist() # [[tn, fp], [fn, tp]]
    cm_p = confusion_matrix(y_test, pred_p).tolist()
    
    metrics = {
        'career': {
            'accuracy': round(acc_c, 3),
            'precision': round(p_c, 3),
            'recall': round(r_c, 3),
            'f1': round(f_c, 3),
            'cm': cm_c
        },
        'phase': {
            'accuracy': round(acc_p, 3),
            'precision': round(p_p, 3),
            'recall': round(r_p, 3),
            'f1': round(f_p, 3),
            'cm': cm_p
        },
        'threshold': round(death_threshold, 1)
    }
    
    with open("processed_data/classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("[Pipeline] Classifier validation metrics and models saved.")
    print("[Pipeline] Pipeline executed successfully!")

if __name__ == '__main__':
    run_pipeline()
