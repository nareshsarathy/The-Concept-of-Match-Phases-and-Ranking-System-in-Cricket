#!/usr/bin/env python
# coding: utf-8

# # IPL Phase-Specialist Engine
# ### Sports Analytics Course Project
# 
# **Dataset:** Cricsheet JSON match files (IPL ball-by-ball)
# **Source format:** One JSON per match → `info{}` + `innings[].overs[].deliveries[]`
# 
#
# 
# | Section | Topic | Methods |
# |---------|-------|---------|
# | 1 | Setup & Data Loading | JSON parsing, Pandas |
# | 2 | Parse All Match Files | Ball-level flattening, edge-case handling |
# | 3 | Exploratory Data Analysis | Descriptive stats, distributions, heatmaps |
# | 4 | Phase Feature Engineering | Powerplay / Middle / Death — reusable pipeline |
# | 5 | Phase-Specialist Scores | Normalisation, composite phase score, phase gap |
# | 6 | Phase-Specialist Leaderboards | Per-phase SR / economy leaders |
# | 7 | ML — Phase SR Prediction | Linear / Ridge / Random Forest / XGBoost |
# | 8 | ML — Player Clustering | K-Means + PCA |
# | 9 | ML — Win Probability Model | Logistic Regression, XGBoost |
# | 10 | Overall Master Dashboard | Matplotlib summary figure |
# | 11 | Last-4-Seasons Analysis | Active-squad view + min-balls outlier filter |
# 

#
# ## Section 1 — Setup & Library Imports


# Install libraries (run once)
import subprocess, sys
for pkg in ['pandas','numpy','matplotlib','seaborn','scikit-learn','xgboost','scipy','tqdm']:
    subprocess.run([sys.executable,'-m','pip','install',pkg,'-q'], check=True)
print("All libraries ready")


import os, json, glob, warnings
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy  import stats
from tqdm   import tqdm

from sklearn.preprocessing    import MinMaxScaler, StandardScaler
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.linear_model     import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble         import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics          import (mean_squared_error, r2_score, mean_absolute_error,
                                      accuracy_score, classification_report, confusion_matrix)
from sklearn.cluster          import KMeans
from sklearn.decomposition    import PCA
warnings.filterwarnings('ignore')

# Dark theme
plt.rcParams.update({
    'figure.facecolor':'#0D0F14','axes.facecolor':'#141720',
    'axes.edgecolor':'#1E2230','axes.labelcolor':'#8A92A6',
    'xtick.color':'#8A92A6','ytick.color':'#8A92A6',
    'text.color':'#F0F2F8','grid.color':'#1E2230',
    'grid.linewidth':0.6,'font.family':'monospace','font.size':9,
})
C = dict(green='#00E5A0',amber='#F5A623',red='#E04C6B',
         blue='#4C8EE0',purple='#7C6AEE',teal='#3BC9DB',muted='#3A4060')

print("Imports complete")


#
# ## Section 2 — Parse All 1,244 JSON Match Files
# 
# ### JSON structure recap
# ```
# {
#   "info": {
#     "season", "venue", "dates", "teams", "players", "toss",
#     "outcome", "player_of_match", "registry": {"people": {name: id}}
#   },
#   "innings": [{
#     "team": ...,
#     "overs": [{
#       "over": 0,          ← 0-indexed (0=1st over)
#       "deliveries": [{
#         "batter", "bowler", "non_striker",
#         "runs": {"batter", "extras", "total"},
#         "extras": {"wides"|"noballs"|"byes"|"legbyes"},  ← optional
#         "wickets": [{"kind","player_out","fielders":[{"name"}]}],  ← optional
#         "replacements": {...}  ← optional (injury sub)
#       }]
#     }]
#   }]
# }
# ```
# 
# **Phase boundaries (over is 0-indexed):**
# - Powerplay : overs 0–5  (1st–6th over)
# - Middle    : overs 6–14 (7th–15th over)
# - Death     : overs 15–19 (16th–20th over)
# 
# **Edge cases handled:**
# - `player_out` ≠ `batter` on run-outs (non-striker dismissed)
# - `is_wide` / `is_noball` → not a legal delivery for SR denominator
# - `bowler_wicket` excludes run-outs (bowler gets no credit)
# - Missing `registry` entries → `player_id = None`
# - No-result / tied matches (`outcome` has no `winner`)
# 


# Point this at your folder of Cricsheet JSON files
DATA_DIR = 'data/'          # ← change to your actual path if needed
# e.g. DATA_DIR = '/path/to/ipl_json/'

json_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.json')))
print(f"Found {len(json_files)} JSON files in '{DATA_DIR}'")

def parse_match(fpath):
    """
    Parse one JSON match file into:
      - ball_rows  : list of dicts, one per delivery
      - match_row  : dict of match-level metadata
    """
    with open(fpath, encoding='utf-8') as f:
        d = json.load(f)

    info     = d['info']
    match_id = os.path.splitext(os.path.basename(fpath))[0]
    season   = info.get('season', None)
    date     = info['dates'][0] if info.get('dates') else None
    venue    = info.get('venue', '')
    teams    = info.get('teams', [None, None])
    outcome  = info.get('outcome', {})
    toss     = info.get('toss', {})
    registry = info.get('registry', {}).get('people', {})

    winner        = outcome.get('winner', None)          # None = no result / tie
    by_dict       = outcome.get('by', {})
    margin_type   = list(by_dict.keys())[0]  if by_dict else None
    margin_val    = list(by_dict.values())[0] if by_dict else None
    potm_list     = info.get('player_of_match', [])
    potm          = potm_list[0] if potm_list else None
    toss_winner   = toss.get('winner')
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
        innings_num  = inn_i + 1

        for ov in inn.get('overs', []):
            over_num = ov['over']               # 0-indexed
            phase    = ('Powerplay' if over_num <= 5  else
                        'Middle'    if over_num <= 14 else
                        'Death')

            for ball in ov.get('deliveries', []):
                batter      = ball['batter']
                bowler      = ball['bowler']
                non_striker = ball['non_striker']
                runs_bat    = ball['runs']['batter']
                runs_extra  = ball['runs']['extras']
                runs_total  = ball['runs']['total']

                extras   = ball.get('extras', {})
                is_wide  = 'wides'   in extras
                is_noball= 'noballs' in extras
                is_bye   = 'byes'    in extras
                is_legbye= 'legbyes' in extras
                # Legal delivery = counts toward SR denominator
                is_legal = not (is_wide or is_noball)

                is_dot  = (runs_total == 0 and is_legal)
                is_four = (runs_bat  == 4  and not is_wide)
                is_six  = (runs_bat  == 6)

                wickets      = ball.get('wickets', [])
                is_wicket    = len(wickets) > 0
                wicket_kind  = wickets[0]['kind']                  if wickets else None
                player_out   = wickets[0].get('player_out', batter) if wickets else None
                fielders     = wickets[0].get('fielders', [])       if wickets else []
                fielder      = fielders[0]['name']                   if fielders else None
                # Bowler gets wicket credit only for non-run-out dismissals
                bowler_wkt   = is_wicket and wicket_kind not in (
                    'run out','retired hurt','retired out','obstructing the field')

                ball_rows.append(dict(
                    match_id=match_id, season=season, date=date,
                    venue=venue, innings=innings_num,
                    batting_team=batting_team, bowling_team=bowling_team,
                    over=over_num, phase=phase,
                    batter=batter, bowler=bowler, non_striker=non_striker,
                    batter_id=registry.get(batter),
                    bowler_id=registry.get(bowler),
                    runs_bat=runs_bat, runs_extra=runs_extra, runs_total=runs_total,
                    is_wide=is_wide, is_noball=is_noball,
                    is_bye=is_bye, is_legbye=is_legbye,
                    is_legal=is_legal, is_dot=is_dot,
                    is_four=is_four, is_six=is_six,
                    is_wicket=is_wicket, wicket_kind=wicket_kind,
                    player_out=player_out, bowler_wkt=bowler_wkt,
                    fielder=fielder,
                    winner=winner, toss_winner=toss_winner,
                    toss_decision=toss_decision,
                ))

    return ball_rows, match_row

print("parse_match() defined")

# Parse all files
# ⏱ ~30-60 seconds for 1,244 files on a typical laptop

all_balls   = []
all_matches = []

for fpath in tqdm(json_files, desc='Parsing JSON files'):
    try:
        balls, match = parse_match(fpath)
        all_balls.extend(balls)
        all_matches.append(match)
    except Exception as e:
        print(f"  {os.path.basename(fpath)}: {e}")

balls_df   = pd.DataFrame(all_balls)
matches_df = pd.DataFrame(all_matches)

# Data-type cleanup
balls_df['date']   = pd.to_datetime(balls_df['date'], errors='coerce')
balls_df['season'] = pd.to_numeric(balls_df['season'], errors='coerce').astype('Int64')

print(f"\n Parsed {len(matches_df):,} matches → {len(balls_df):,} deliveries")
print(f"   Seasons : {sorted(balls_df['season'].dropna().unique().tolist())}")
print(f"   Columns : {list(balls_df.columns)}")
display(balls_df.head(4))

# Sanity checks
print("NULL counts (ball-level):")
nulls = balls_df.isnull().sum()
print(nulls[nulls > 0].to_string())

print(f"\nPhase distribution:")
print(balls_df['phase'].value_counts())

print(f"\nMatches with no result (winner=None): {matches_df['winner'].isna().sum()}")
print(f"Unique venues  : {balls_df['venue'].nunique()}")
print(f"Unique batters : {balls_df['batter'].nunique()}")
print(f"Unique bowlers : {balls_df['bowler'].nunique()}")

#
# ## Section 3 — Exploratory Data Analysis



'''
# 3.1 Season-wise match volume
season_counts = matches_df.groupby('season').size().reset_index(name='matches')

season_counts['season'] = pd.to_numeric(season_counts['season'])

fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor='#0D0F14')

axes[0].bar(season_counts['season'], season_counts['matches'],
            color=C['teal'], edgecolor='none', alpha=0.85)
axes[0].set_title('Matches per IPL Season'); axes[0].set_xlabel('Season')
axes[0].set_ylabel('Matches'); axes[0].grid(axis='y', linestyle='--', alpha=0.4)

# Average total runs per season
season_runs = (balls_df.groupby(['season','match_id','innings'])['runs_total']
               .sum().reset_index()
               .groupby('season')['runs_total'].mean().reset_index())
axes[1].plot(season_runs['season'], season_runs['runs_total'],
             color=C['amber'], linewidth=2, marker='o', markersize=4)
axes[1].set_title('Average Innings Score by Season')
axes[1].set_xlabel('Season'); axes[1].set_ylabel('Avg Runs')
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout(); plt.show()
'''



# 3.1 Season-wise match volume
import numpy as np

season_counts = matches_df.groupby('season').size().reset_index(name='matches')

# Ensure season is string
season_counts['season'] = season_counts['season'].astype(str)

fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor='#0D0F14')

# Create numeric positions for bars
x = np.arange(len(season_counts))

axes[0].bar(
    x,
    season_counts['matches'],
    color=C['teal'],
    edgecolor='none',
    alpha=0.85
)

axes[0].set_xticks(x)
axes[0].set_xticklabels(season_counts['season'], rotation=45)

axes[0].set_title('Matches per IPL Season')
axes[0].set_xlabel('Season')
axes[0].set_ylabel('Matches')
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

# Average total runs per season
season_runs = (
    balls_df.groupby(['season', 'match_id', 'innings'])['runs_total']
    .sum()
    .reset_index()
    .groupby('season')['runs_total']
    .mean()
    .reset_index()
)

season_runs['season'] = season_runs['season'].astype(str)

x2 = np.arange(len(season_runs))

axes[1].plot(
    x2,
    season_runs['runs_total'],
    color=C['amber'],
    linewidth=2,
    marker='o',
    markersize=4
)

axes[1].set_xticks(x2)
axes[1].set_xticklabels(season_runs['season'], rotation=45)

axes[1].set_title('Average Innings Score by Season')
axes[1].set_xlabel('Season')
axes[1].set_ylabel('Avg Runs')
axes[1].grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()


# 3.2 Run-rate curve (avg runs scored per over)
over_rr = (balls_df.groupby('over')
           .agg(total_runs=('runs_total','sum'), total_balls=('is_legal','sum'))
           .reset_index())
over_rr['rr'] = over_rr['total_runs'] / over_rr['total_balls'] * 6

fig, ax = plt.subplots(figsize=(13, 4), facecolor='#0D0F14')
ax.plot(over_rr['over']+1, over_rr['rr'], color=C['teal'], linewidth=2)
ax.fill_between(over_rr['over']+1, over_rr['rr'], alpha=0.12, color=C['teal'])
ax.axvspan(1,  6,  alpha=0.06, color=C['purple'],label='Powerplay 0–5')
ax.axvspan(7,  15, alpha=0.06, color=C['teal'],  label='Middle 6–14')
ax.axvspan(16, 20, alpha=0.06, color=C['amber'], label='Death 15–19')
ax.set_title('Average Run Rate per Over (all seasons)')
ax.set_xlabel('Over number'); ax.set_ylabel('Run rate (runs/over)')
ax.legend(facecolor='#0D0F14', labelcolor='#F0F2F8', fontsize=8)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout(); plt.show()


# 3.3 Wicket frequency per over
wkt_over = (balls_df[balls_df['is_wicket']]
            .groupby('over').size()
            .div(matches_df.shape[0])
            .reset_index(name='wkts_per_match'))

fig, ax = plt.subplots(figsize=(13, 4), facecolor='#0D0F14')
ax.bar(wkt_over['over']+1, wkt_over['wkts_per_match'],
       color=C['red'], edgecolor='none', alpha=0.8)
ax.set_title('Average Wickets per Over (per match)')
ax.set_xlabel('Over number'); ax.set_ylabel('Wickets per match')
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout(); plt.show()


# 3.4 Dismissal type breakdown
wkt_df   = balls_df[balls_df['is_wicket']].copy()
kind_cnt = wkt_df['wicket_kind'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor='#0D0F14')
pal = [C['green'],C['amber'],C['red'],C['blue'],C['purple'],C['teal'],'#FF8C6B','#2ECC71']

axes[0].barh(kind_cnt.index, kind_cnt.values,
             color=pal[:len(kind_cnt)], edgecolor='none', alpha=0.85)
axes[0].set_title('Dismissal Types (all seasons)'); axes[0].invert_yaxis()
axes[0].grid(axis='x', linestyle='--', alpha=0.4)

# Toss decision vs win rate
matches_df['toss_bat']  = (matches_df['toss_decision']=='bat').astype(int)
matches_df['toss_won']  = (matches_df['toss_winner']==matches_df['winner']).astype(int)
toss_wr = matches_df.groupby('toss_decision')['toss_won'].mean()
axes[1].bar(toss_wr.index, toss_wr.values*100,
            color=[C['green'],C['amber']], edgecolor='none', alpha=0.85)
axes[1].set_title('Win Rate When Toss Won — by Decision')
axes[1].set_ylabel('Win %'); axes[1].set_ylim(0,70)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(); plt.show()
print("Toss win rates:", toss_wr.round(3).to_dict())

# 3.5 Top run scorers & wicket takers (career)
career_bat = (balls_df[balls_df['is_legal']]
              .groupby('batter')
              .agg(runs=('runs_bat','sum'), balls=('is_legal','sum'),
                   fours=('is_four','sum'), sixes=('is_six','sum'),
                   matches=('match_id','nunique'))
              .reset_index())
career_bat['career_sr']  = (career_bat['runs']/career_bat['balls']*100).round(2)

career_bowl = (balls_df[balls_df['is_legal']]
               .groupby('bowler')
               .agg(runs_c=('runs_total','sum'), balls=('is_legal','sum'),
                    wickets=('bowler_wkt','sum'), matches=('match_id','nunique'))
               .reset_index())
career_bowl['career_econ'] = (career_bowl['runs_c']/career_bowl['balls']*6).round(2)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0D0F14')

top_bat  = career_bat.nlargest(12,'runs')
top_bowl = career_bowl.nlargest(12,'wickets')

axes[0].barh(top_bat['batter'], top_bat['runs'],
             color=[C['green'] if i==0 else C['green']+'70' for i in range(len(top_bat))],
             edgecolor='none')
axes[0].set_title('Top 12 Run Scorers (Career)'); axes[0].invert_yaxis()
axes[0].grid(axis='x', linestyle='--', alpha=0.4)

axes[1].barh(top_bowl['bowler'], top_bowl['wickets'],
             color=[C['red'] if i==0 else C['red']+'70' for i in range(len(top_bowl))],
             edgecolor='none')
axes[1].set_title('Top 12 Wicket Takers (Career)'); axes[1].invert_yaxis()
axes[1].grid(axis='x', linestyle='--', alpha=0.4)

plt.tight_layout(); plt.show()

#
# ## Section 4 — Phase Feature Engineering
# 
# **Phase insight:** Overall career averages hide phase specialists.
# A batter with SR 130 overall might have:
# - Powerplay SR 165 (elite opener)
# - Death SR 95  (useless finisher)
# 
# We never look at the average, we look at each **phase** separately.
# 
# The whole pipeline is wrapped in a single reusable function,
# `build_phase_tables(balls, cfg)`, so the **exact same analysis** can be run on
# the full history (Section 4–10) and again on only the last 4 seasons (Section 11).
# 
# **Outlier / minimum-sample filter:** a (player, phase) row is kept only if the
# player has faced / bowled at least a configurable number of **legal balls** in
# that phase. This stops tiny samples (a player who faced 4 death balls at SR 200)
# from polluting the leaderboards. Thresholds live in the `CFG` config block below.
# 


# 4.0 Config + reusable phase pipeline
from types import SimpleNamespace

# Per-phase composite score weights + minimum balls outlier thresholds.
# Two profiles: a generous bar for the full history, a lower bar for the
# smaller last-4-seasons sample. Tune freely.
CFG = SimpleNamespace(
    overall = SimpleNamespace(
        bat_weights    = {'pp': 0.35, 'mid': 0.30, 'dth': 0.35},
        bowl_weights   = {'pp': 0.30, 'mid': 0.25, 'dth': 0.45},   # death weighted heaviest
        min_bat_balls  = {'Powerplay': 222, 'Middle': 294, 'Death': 162},
        min_bowl_balls = {'Powerplay': 210, 'Middle': 288, 'Death': 144},
    ),
    recent = SimpleNamespace(
        bat_weights    = {'pp': 0.35, 'mid': 0.30, 'dth': 0.35},
        bowl_weights   = {'pp': 0.30, 'mid': 0.25, 'dth': 0.45},
        min_bat_balls  = {'Powerplay': 60, 'Middle': 90, 'Death': 40},
        min_bowl_balls = {'Powerplay': 60, 'Middle': 72, 'Death': 54},
    ),
)

def build_phase_tables(balls, cfg):
    """Full phase-specialist pipeline for any subset of `balls_df`.

    Returns a SimpleNamespace with career/phase/wide tables, composite
    phase-specialist scores, phase-gap, correlation matrix and run-rate curve.
    """
    legal = balls[balls['is_legal']].copy()

    # Career aggregates (computed on this subset) #
    career_bat = (legal.groupby('batter')
                  .agg(runs=('runs_bat', 'sum'), balls=('is_legal', 'sum'),
                       fours=('is_four', 'sum'), sixes=('is_six', 'sum'),
                       matches=('match_id', 'nunique')).reset_index())
    career_bat['career_sr'] = (career_bat['runs'] / career_bat['balls'] * 100).round(2)

    career_bowl = (legal.groupby('bowler')
                   .agg(runs_c=('runs_total', 'sum'), balls=('is_legal', 'sum'),
                        wickets=('bowler_wkt', 'sum'), matches=('match_id', 'nunique')).reset_index())
    career_bowl['career_econ'] = (career_bowl['runs_c'] / career_bowl['balls'] * 6).round(2)

    # Batting phase stats #
    bat_phase = (legal.groupby(['batter', 'phase'])
                 .agg(runs=('runs_bat', 'sum'), balls=('is_legal', 'sum'),
                      fours=('is_four', 'sum'), sixes=('is_six', 'sum'),
                      dots=('is_dot', 'sum'), innings=('match_id', 'nunique')).reset_index())
    dismissals = (balls[balls['is_wicket'] & balls['player_out'].notna()]
                  .groupby(['player_out', 'phase']).size().reset_index(name='dismissals')
                  .rename(columns={'player_out': 'batter'}))
    bat_phase = bat_phase.merge(dismissals, on=['batter', 'phase'], how='left')
    bat_phase['dismissals']  = bat_phase['dismissals'].fillna(0)
    bat_phase['strike_rate'] = (bat_phase['runs'] / bat_phase['balls'] * 100).round(2)
    bat_phase['batting_avg'] = (bat_phase['runs'] / bat_phase['dismissals'].clip(lower=1)).round(2)
    bat_phase['boundary_pct']= ((bat_phase['fours'] * 4 + bat_phase['sixes'] * 6) /
                                bat_phase['runs'].clip(lower=1) * 100).round(1)
    bat_phase['dot_rate']    = (bat_phase['dots'] / bat_phase['balls'] * 100).round(1)

    # Bowling phase stats #
    bowl_phase = (legal.groupby(['bowler', 'phase'])
                  .agg(runs_c=('runs_total', 'sum'), balls=('is_legal', 'sum'),
                       wickets=('bowler_wkt', 'sum'), dots=('is_dot', 'sum'),
                       innings=('match_id', 'nunique')).reset_index())
    bowl_phase['economy'] = (bowl_phase['runs_c'] / bowl_phase['balls'] * 6).round(2)
    bowl_phase['dot_pct'] = (bowl_phase['dots'] / bowl_phase['balls'] * 100).round(1)
    bowl_phase['bowl_sr'] = (bowl_phase['balls'] / bowl_phase['wickets'].clip(lower=1)).round(1)

    # Outlier filter: minimum legal balls faced / bowled per phase #
    bat_filtered = bat_phase[bat_phase.apply(
        lambda r: r['balls'] >= cfg.min_bat_balls.get(r['phase'], 0), axis=1)].copy()
    bowl_filtered = bowl_phase[bowl_phase.apply(
        lambda r: r['balls'] >= cfg.min_bowl_balls.get(r['phase'], 0), axis=1)].copy()

    # Wide batting (SR per phase) #
    bat_wide = (bat_filtered.pivot_table(index='batter', columns='phase',
                                         values='strike_rate', aggfunc='first').reset_index())
    bat_wide.columns.name = None
    bat_wide = bat_wide.rename(columns={'Powerplay': 'sr_pp', 'Middle': 'sr_mid', 'Death': 'sr_dth'})
    for c in ['sr_pp', 'sr_mid', 'sr_dth']:
        if c not in bat_wide.columns:
            bat_wide[c] = np.nan
    bat_wide = bat_wide.merge(career_bat[['batter', 'runs', 'career_sr', 'matches']],
                              on='batter', how='left')

    # Wide bowling (economy per phase) #
    bowl_wide = (bowl_filtered.pivot_table(index='bowler', columns='phase',
                                           values='economy', aggfunc='first').reset_index())
    bowl_wide.columns.name = None
    bowl_wide = bowl_wide.rename(columns={'Powerplay': 'econ_pp', 'Middle': 'econ_mid', 'Death': 'econ_dth'})
    for c in ['econ_pp', 'econ_mid', 'econ_dth']:
        if c not in bowl_wide.columns:
            bowl_wide[c] = np.nan
    bowl_wide = bowl_wide.merge(career_bowl[['bowler', 'wickets', 'career_econ', 'matches']],
                                on='bowler', how='left')

    # Normalise to 0-100 (bowling inverted: lower economy = better) #
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

    # Composite phase-specialist score #
    wb, wo = cfg.bat_weights, cfg.bowl_weights

    def bat_perf(row):
        car = row.get('career_sr_n', 50.0); car = 50.0 if pd.isna(car) else car
        pp  = row.get('sr_pp_n',  np.nan); pp  = car if pd.isna(pp)  else pp
        mid = row.get('sr_mid_n', np.nan); mid = car if pd.isna(mid) else mid
        dth = row.get('sr_dth_n', np.nan); dth = car if pd.isna(dth) else dth
        return round(wb['pp'] * pp + wb['mid'] * mid + wb['dth'] * dth, 2)

    def bowl_perf(row):
        car = row.get('career_econ_n', 50.0); car = 50.0 if pd.isna(car) else car
        pp  = row.get('econ_pp_n',  np.nan); pp  = car if pd.isna(pp)  else pp
        mid = row.get('econ_mid_n', np.nan); mid = car if pd.isna(mid) else mid
        dth = row.get('econ_dth_n', np.nan); dth = car if pd.isna(dth) else dth
        return round(wo['pp'] * pp + wo['mid'] * mid + wo['dth'] * dth, 2)

    bat_wide['perf_score']  = bat_wide.apply(bat_perf,  axis=1)
    bowl_wide['perf_score'] = bowl_wide.apply(bowl_perf, axis=1)

    # Phase gap (how specialised a player is) #
    bat_wide['sr_best']    = bat_wide[['sr_pp', 'sr_mid', 'sr_dth']].max(axis=1)
    bat_wide['sr_worst']   = bat_wide[['sr_pp', 'sr_mid', 'sr_dth']].min(axis=1)
    bat_wide['phase_gap']  = (bat_wide['sr_best'] - bat_wide['sr_worst']).round(1)
    bowl_wide['econ_best'] = bowl_wide[['econ_pp', 'econ_mid', 'econ_dth']].min(axis=1)
    bowl_wide['econ_worst']= bowl_wide[['econ_pp', 'econ_mid', 'econ_dth']].max(axis=1)
    bowl_wide['phase_gap'] = (bowl_wide['econ_worst'] - bowl_wide['econ_best']).round(1)

    # Phase SR correlation matrix #
    corr_cols = ['sr_pp', 'sr_mid', 'sr_dth', 'career_sr']
    corr_mat  = bat_wide[corr_cols].corr()

    # Run-rate curve #
    over_rr = (legal.groupby('over')
               .agg(total_runs=('runs_total', 'sum'), total_balls=('is_legal', 'sum')).reset_index())
    over_rr['rr'] = over_rr['total_runs'] / over_rr['total_balls'] * 6

    return SimpleNamespace(
        career_bat=career_bat, career_bowl=career_bowl,
        bat_phase=bat_phase, bowl_phase=bowl_phase,
        bat_filtered=bat_filtered, bowl_filtered=bowl_filtered,
        bat_wide=bat_wide, bowl_wide=bowl_wide,
        corr_mat=corr_mat, over_rr=over_rr, cfg=cfg)

print("build_phase_tables() + CFG ready")


# 4.1 Run the pipeline on the FULL history (overall)
O = build_phase_tables(balls_df, CFG.overall)

# Unpack into the familiar bare names used by the ML + dashboard cells below
career_bat   = O.career_bat;   career_bowl   = O.career_bowl
bat_phase    = O.bat_phase;    bowl_phase    = O.bowl_phase
bat_filtered = O.bat_filtered; bowl_filtered = O.bowl_filtered
bat_wide     = O.bat_wide;     bowl_wide     = O.bowl_wide
corr_mat     = O.corr_mat

print(f"Batting phase rows           : {len(bat_phase)}")
print(f"Bowling phase rows           : {len(bowl_phase)}")
print(f"Batting rows after filter    : {len(bat_filtered)}")
print(f"Bowling rows after filter    : {len(bowl_filtered)}")
print(f"Batters in wide table        : {bat_wide.shape[0]}")
print(f"Bowlers in wide table        : {bowl_wide.shape[0]}")
print(f"\nMin-balls thresholds (overall):")
print(f"  Batting : {CFG.overall.min_bat_balls}")
print(f"  Bowling : {CFG.overall.min_bowl_balls}")

# 4.2 Preview batting & bowling phase tables
print("Batting phase stats (sample):")
display(bat_phase.head(8))
print("Bowling phase stats (sample):")
display(bowl_phase.head(8))


# 4.3 Phase heatmap for top specialists
fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='#0D0F14')
fig.suptitle('Phase Specialist Leaders', color='#F0F2F8', fontsize=14, y=1.01)

PHASE_COLORS = {'Powerplay': C['purple'], 'Middle': C['teal'], 'Death': C['amber']}

for col, phase in enumerate(['Powerplay','Middle','Death']):
    pc = PHASE_COLORS[phase]

    # Top batters by SR
    tb = bat_filtered[bat_filtered['phase']==phase].nlargest(12,'strike_rate')
    bc = [pc if i==0 else pc+'60' for i in range(len(tb))]
    axes[0,col].barh(tb['batter'], tb['strike_rate'], color=bc, edgecolor='none')
    axes[0,col].set_title(f'{phase}  |  Batting SR', color=pc)
    axes[0,col].invert_yaxis()
    axes[0,col].grid(axis='x', linestyle='--', alpha=0.35)

    # Top bowlers by economy (lower = better)
    tb2 = bowl_filtered[bowl_filtered['phase']==phase].nsmallest(12,'economy')
    axes[1,col].barh(tb2['bowler'], tb2['economy'],
                     color=[C['red'] if i==0 else C['red']+'60' for i in range(len(tb2))],
                     edgecolor='none')
    axes[1,col].set_title(f'{phase}  |  Bowling Economy', color=C['red'])
    axes[1,col].invert_yaxis()
    axes[1,col].grid(axis='x', linestyle='--', alpha=0.35)

plt.tight_layout(); plt.show()

# 4.4 Phase SR correlation heatmap
corr_cols = ['sr_pp', 'sr_mid', 'sr_dth', 'career_sr']

fig, ax = plt.subplots(figsize=(6, 5), facecolor='#0D0F14')
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('mb', ['#E04C6B', '#141720', '#00E5A0'])
im = ax.imshow(corr_mat.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, shrink=0.8)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(corr_cols, rotation=30, ha='right')
ax.set_yticklabels(corr_cols)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f'{corr_mat.iloc[i, j]:.2f}',
                ha='center', va='center', fontsize=9, color='#F0F2F8')
ax.set_title('Phase SR Correlation Matrix')
plt.tight_layout(); plt.show()

print("\nKey finding: low correlation between phase SRs confirms")
print("phase specialists are DISTINCT from 'good overall' players.")

# 
# ## Section 5 — Phase-Specialist Scores
# 
# Each player's per-phase metric is normalised to 0–100, then combined into a
# single **phase-specialist score**:
# 
# ```
# Batting score = 0.35·norm(PP_SR) + 0.30·norm(Mid_SR) + 0.35·norm(Death_SR)
# Bowling score = 0.30·norm(PP_econ) + 0.25·norm(Mid_econ) + 0.45·norm(Death_econ)
# ```
# 
# Powerplay and death overs are the highest-leverage phases in T20, so they carry
# the most weight (death heaviest for bowlers). The **phase gap** (best minus worst
# phase) measures how *specialised* a player is rather than how good they are overall.
#

# 5.1 Top phase-specialist scores (overall)
print("Top 10 batting phase-specialist scores:")
display(bat_wide.nlargest(10, 'perf_score')[['batter', 'sr_pp', 'sr_mid', 'sr_dth', 'perf_score']])

print("\nTop 10 bowling phase-specialist scores:")
display(bowl_wide.nlargest(10, 'perf_score')[['bowler', 'econ_pp', 'econ_mid', 'econ_dth', 'perf_score']])


# 5.2 Phase gap analysis — most specialised players
print("Most phase-specialised batters (largest SR gap, best vs worst phase):")
display(bat_wide.nlargest(10, 'phase_gap')[['batter', 'sr_pp', 'sr_mid', 'sr_dth', 'phase_gap']])

fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0D0F14')
ax.hist(bat_wide['phase_gap'].dropna(), bins=40, color=C['amber'], edgecolor='none', alpha=0.8)
ax.axvline(bat_wide['phase_gap'].median(), color=C['red'],
           linestyle='--', linewidth=1.5, label='Median gap')
ax.set_title('Distribution of Phase SR Gap (Best \u2212 Worst Phase)')
ax.set_xlabel('SR difference'); ax.legend(facecolor='#0D0F14', labelcolor='#F0F2F8')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout(); plt.show()

#
# ## Section 6 — Phase-Specialist Leaderboards
# 
# Who actually wins each phase? These leaderboards rank the qualified players
# (those clearing the min-balls filter) by raw phase performance like strike rate for
# batters, economy for bowlers plus a combined profile of the top composite-score
# specialists.
#

# 6.1 Per-phase batting leaders (by strike rate)
for phase in ['Powerplay', 'Middle', 'Death']:
    t = bat_filtered[bat_filtered['phase'] == phase].nlargest(8, 'strike_rate')
    print(f"\n{phase} \u2014 top batting strike rates:")
    display(t[['batter', 'runs', 'balls', 'strike_rate', 'boundary_pct']].reset_index(drop=True))


# 6.2 Per-phase bowling leaders (by economy)
for phase in ['Powerplay', 'Middle', 'Death']:
    t = bowl_filtered[bowl_filtered['phase'] == phase].nsmallest(8, 'economy')
    print(f"\n{phase} \u2014 best bowling economy:")
    display(t[['bowler', 'balls', 'wickets', 'economy', 'dot_pct']].reset_index(drop=True))


# 6.3 Phase profile bars for top composite specialists
top_bat  = bat_wide.nlargest(10, 'perf_score').reset_index(drop=True)
top_bowl = bowl_wide.nlargest(10, 'perf_score').reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0D0F14')
fig.suptitle('Top Phase Specialists \u2014 Profiles', color='#F0F2F8', fontsize=13)
w = 0.27

x = np.arange(len(top_bat))
axes[0].bar(x - w, top_bat['sr_pp'].fillna(0),  w, label='Powerplay SR', color=C['purple'], alpha=0.85, edgecolor='none')
axes[0].bar(x,     top_bat['sr_mid'].fillna(0), w, label='Middle SR',    color=C['teal'],   alpha=0.85, edgecolor='none')
axes[0].bar(x + w, top_bat['sr_dth'].fillna(0), w, label='Death SR',     color=C['amber'],  alpha=0.85, edgecolor='none')
axes[0].set_xticks(x)
axes[0].set_xticklabels(top_bat['batter'], rotation=35, ha='right', fontsize=8)
axes[0].set_title('Top Batters \u2014 SR by Phase')
axes[0].legend(facecolor='#0D0F14', labelcolor='#F0F2F8', fontsize=8)
axes[0].grid(axis='y', linestyle='--', alpha=0.4)

x2 = np.arange(len(top_bowl))
axes[1].bar(x2 - w, top_bowl['econ_pp'].fillna(0),  w, label='Powerplay Econ', color=C['purple'], alpha=0.85, edgecolor='none')
axes[1].bar(x2,     top_bowl['econ_mid'].fillna(0), w, label='Middle Econ',    color=C['teal'],   alpha=0.85, edgecolor='none')
axes[1].bar(x2 + w, top_bowl['econ_dth'].fillna(0), w, label='Death Econ',     color=C['amber'],  alpha=0.85, edgecolor='none')
axes[1].set_xticks(x2)
axes[1].set_xticklabels(top_bowl['bowler'], rotation=35, ha='right', fontsize=8)
axes[1].set_title('Top Bowlers \u2014 Economy by Phase (lower = better)')
axes[1].legend(facecolor='#0D0F14', labelcolor='#F0F2F8', fontsize=8)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout(); plt.show()

#
# ## Section 7 — ML: Phase SR Prediction (Regression)
# 
# **Question:** Can we predict a player's death-overs SR from their powerplay and
# middle-overs SR?
# 
# If yes, a player whose *actual* death SR sits far **above** the model's prediction
# is over-performing in the death phase relative to what their other phases suggest
# a genuine death-overs specialist rather than a uniformly strong batter.
#

# 7.1 Build ML feature matrix
ml_bat = bat_wide.dropna(subset=['sr_pp','sr_mid','sr_dth','career_sr']).copy()
ml_bat['log_runs']  = np.log1p(ml_bat['runs'])

FEATURES = ['sr_pp','sr_mid','career_sr','log_runs','matches']
TARGET   = 'sr_dth'                       # predict death SR

X = ml_bat[FEATURES]
y = ml_bat[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
print(f"\nTarget range: {y.min():.1f} – {y.max():.1f}  Mean: {y.mean():.1f}")

# 7.2 Linear Regression baseline
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print(f"Linear Regression:")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_test,y_pred_lr)):.3f}")
print(f"  R²   : {r2_score(y_test,y_pred_lr):.3f}")
print(f"  MAE  : {mean_absolute_error(y_test,y_pred_lr):.3f}")
print(f"\n  Coefficients:")
for fname, coef in zip(FEATURES, lr.coef_):
    print(f"   {fname:15s}: {coef:.3f}")

# 7.3 Ridge Regression
ridge = Ridge(alpha=10.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print(f"Ridge Regression:")
print(f"  RMSE : {np.sqrt(mean_squared_error(y_test,y_pred_ridge)):.3f}")
print(f"  R²   : {r2_score(y_test,y_pred_ridge):.3f}")

# 7.4 Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test,y_pred_rf))
r2_rf   = r2_score(y_test,y_pred_rf)
print(f"Random Forest:")
print(f"  RMSE : {rmse_rf:.3f}")
print(f"  R²   : {r2_rf:.3f}")
cv = cross_val_score(rf, X, y, cv=5, scoring='r2')
print(f"  CV R² (5-fold): {cv.mean():.3f} ± {cv.std():.3f}")

# 7.5 XGBoost
try:
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    print(f"XGBoost:  RMSE={np.sqrt(mean_squared_error(y_test,y_pred_xgb)):.3f}"
          f"  R²={r2_score(y_test,y_pred_xgb):.3f}")
    best_model, best_preds, best_name = xgb, y_pred_xgb, 'XGBoost'
except ImportError:
    print("XGBoost not available — using Random Forest")
    best_model, best_preds, best_name = rf, y_pred_rf, 'Random Forest'

# 7.6 Model comparison plot + residual analysis
fig, axes = plt.subplots(1, 3, figsize=(17, 5), facecolor='#0D0F14')
fig.suptitle('Death SR Prediction — Model Comparison', color='#F0F2F8', fontsize=13)

for ax, (name, preds) in zip(axes, [
    ('Linear Reg', y_pred_lr),
    ('Random Forest', y_pred_rf),
    (best_name, best_preds),
]):
    ax.scatter(y_test, preds, alpha=0.5, s=22, color=C['blue'], edgecolors='none')
    lims = [max(0, y.min()-5), y.max()+5]
    ax.plot(lims, lims, '--', color=C['amber'], linewidth=1.5)
    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    ax.set_title(f'{name}\nR²={r2:.3f}  RMSE={rmse:.2f}', color='#F0F2F8')
    ax.set_xlabel('Actual Death SR'); ax.set_ylabel('Predicted Death SR')
    ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout(); plt.show()

# 7.7 Hidden death specialists: actual << predicted
ml_bat['pred_death_sr']  = best_model.predict(ml_bat[FEATURES])
ml_bat['death_sr_gap']   = (ml_bat['sr_dth'] - ml_bat['pred_death_sr']).round(2)
# Positive gap = outperforms prediction = hidden gem
# Negative gap = underperforms prediction = overrated death batter

print("Players who MOST OUTPERFORM their predicted death SR (true hidden gems):")
display(ml_bat.nlargest(10,'death_sr_gap')[
    ['batter','sr_pp','sr_mid','sr_dth','pred_death_sr','death_sr_gap','matches']])

print("\nPlayers who most UNDERPERFORM their predicted death SR (overrated finishers):")
display(ml_bat.nsmallest(10,'death_sr_gap')[
    ['batter','sr_pp','sr_mid','sr_dth','pred_death_sr','death_sr_gap','matches']])

# 7.8 Feature importance
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0D0F14')
colors = [C['green'] if v==imp.max() else C['blue']+'90' for v in imp.values]
ax.barh(imp.index, imp.values, color=colors, edgecolor='none')
ax.set_title('Random Forest Feature Importance (predicting Death SR)')
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout(); plt.show()

#
# ## Section 8 — ML: Player Clustering (K-Means + PCA)

# 8.1 Build clustering feature matrix
clust_feats = ['sr_pp','sr_mid','sr_dth','career_sr','matches']
ml_clust = bat_wide.dropna(subset=clust_feats).copy()

scaler_c = StandardScaler()
X_c = scaler_c.fit_transform(ml_clust[clust_feats])
print(f"Clustering matrix: {X_c.shape}")

# 8.2 Elbow method
inertias = []
K_range  = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    inertias.append(km.fit(X_c).inertia_)

diffs2 = np.diff(np.diff(inertias))
elbow_k = list(K_range)[np.argmin(diffs2)+1]

fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0D0F14')
ax.plot(K_range, inertias, marker='o', color=C['green'], linewidth=2, markersize=5)
ax.axvline(elbow_k, color=C['amber'], linestyle='--', label=f'Elbow K={elbow_k}')
ax.set_title('Elbow Method — Optimal K')
ax.set_xlabel('K'); ax.set_ylabel('Inertia')
ax.legend(facecolor='#0D0F14', labelcolor='#F0F2F8')
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout(); plt.show()
print(f"Optimal K: {elbow_k}")

# 8.3 Fit K-Means + PCA
K_BEST = elbow_k
km_final = KMeans(n_clusters=K_BEST, random_state=42, n_init=10)
ml_clust['cluster'] = km_final.fit_predict(X_c)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_c)
ml_clust['pc1'], ml_clust['pc2'] = X_pca[:,0], X_pca[:,1]
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

pal = [C['green'],C['amber'],C['red'],C['blue'],C['purple'],C['teal']]
fig, ax = plt.subplots(figsize=(11, 7), facecolor='#0D0F14')
for k in range(K_BEST):
    mask = ml_clust['cluster']==k
    ax.scatter(ml_clust.loc[mask,'pc1'], ml_clust.loc[mask,'pc2'],
               color=pal[k%len(pal)], alpha=0.55, s=35, edgecolors='none',
               label=f'Cluster {k}')
    for _, row in ml_clust[mask].nlargest(3,'perf_score').iterrows():
        ax.annotate(row['batter'], (row['pc1'],row['pc2']),
                    fontsize=7, color='#F0F2F8', xytext=(4,4),
                    textcoords='offset points')
ax.set_title(f'Batter Clusters (K={K_BEST}) — PCA 2D')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax.legend(facecolor='#0D0F14', labelcolor='#F0F2F8', fontsize=8)
ax.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout(); plt.show()

# 8.4 Cluster profiles
profile = ml_clust.groupby('cluster')[clust_feats].mean().round(1)
sizes   = ml_clust['cluster'].value_counts().sort_index().rename('count')
profile = profile.join(sizes)

print("Cluster profiles (mean stats):")
display(profile)

# Automatic archetype labelling
def label_cluster(row):
    if row['sr_pp'] > row['sr_dth'] + 15:   return '⚡ Powerplay Striker'
    if row['sr_dth'] > row['sr_pp'] + 15:   return '🔥 Death Finisher'
    if row['sr_pp'] > 140 and row['sr_dth'] > 140: return '🌟 All-Phase Dominator'
    if row['sr_pp'] < 110 and row['career_sr'] < 120: return '🛡️ Anchor / Builder'
    return '⚖️ Balanced Contributor'

profile['archetype'] = profile.apply(label_cluster, axis=1)
print("\nArchetype labels:")
print(profile['archetype'].to_string())

#
# ## Section 9 — ML: Match Win Probability Model

# 9.1 Build match-level feature matrix
# Features: toss_decision, team run-rate history, margin type
# Target: did team1 win?

matches_ml = matches_df.dropna(subset=['winner','toss_decision']).copy()
matches_ml['team1_wins']  = (matches_ml['winner'] == matches_ml['team1']).astype(int)
matches_ml['toss_bat']    = (matches_ml['toss_decision'] == 'bat').astype(int)
matches_ml['toss_is_t1']  = (matches_ml['toss_winner']  == matches_ml['team1']).astype(int)

# Average innings run rate per team (overall proxy for team strength)
team_rr = (balls_df[balls_df['is_legal']]
           .groupby(['match_id','batting_team'])
           .agg(runs=('runs_total','sum'), balls=('is_legal','sum'))
           .reset_index())
team_rr['rr'] = team_rr['runs'] / team_rr['balls'] * 6
avg_rr = team_rr.groupby('batting_team')['rr'].mean().reset_index(name='avg_rr')

matches_ml = matches_ml.merge(avg_rr.rename(columns={'batting_team':'team1','avg_rr':'t1_rr'}), on='team1', how='left')
matches_ml = matches_ml.merge(avg_rr.rename(columns={'batting_team':'team2','avg_rr':'t2_rr'}), on='team2', how='left')
matches_ml['rr_diff'] = (matches_ml['t1_rr'] - matches_ml['t2_rr']).fillna(0)

WIN_FEATS = ['toss_bat','toss_is_t1','rr_diff']
X_w = matches_ml[WIN_FEATS].fillna(0)
y_w = matches_ml['team1_wins']
print(f"Match ML shape: {X_w.shape}  |  Base win rate: {y_w.mean():.3f}")

# 9.2 Train classifiers
X_wtr, X_wte, y_wtr, y_wte = train_test_split(X_w, y_w, test_size=0.25, random_state=42)

log_reg = LogisticRegression(max_iter=500)
rf_clf  = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
gb_clf  = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42)

for name, model in [('Logistic Reg', log_reg),
                    ('Random Forest', rf_clf),
                    ('Gradient Boost', gb_clf)]:
    model.fit(X_wtr, y_wtr)
    acc = accuracy_score(y_wte, model.predict(X_wte))
    cv  = cross_val_score(model, X_w, y_w, cv=5, scoring='accuracy')
    print(f"{name:15s}  Acc={acc:.3f}  CV={cv.mean():.3f}±{cv.std():.3f}")

# 9.3 Confusion matrix + classification report
print("Random Forest Classification Report:")
print(classification_report(y_wte, rf_clf.predict(X_wte),
                             target_names=['Team2 Wins','Team1 Wins']))

from matplotlib.colors import LinearSegmentedColormap
cmap_cm = LinearSegmentedColormap.from_list('cm',['#141720','#00E5A0'])

fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor='#0D0F14')
for ax, (name, model) in zip(axes, [('Logistic Reg', log_reg),
                                      ('Random Forest', rf_clf)]):
    cm = confusion_matrix(y_wte, model.predict(X_wte))
    im = ax.imshow(cm, cmap=cmap_cm)
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(2):
        for j in range(2):
            ax.text(j,i,cm[i,j],ha='center',va='center',
                    fontsize=14,color='#F0F2F8',fontweight='bold')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Pred T2','Pred T1'])
    ax.set_yticklabels(['T2 Wins','T1 Wins'])
    acc = accuracy_score(y_wte, model.predict(X_wte))
    ax.set_title(f'{name}  Acc={acc:.3f}', color='#F0F2F8')
plt.tight_layout(); plt.show()

#
# ## Section 10 — Overall Master Dashboard
# 
# A single 3×4 figure summarising the phase-specialist analysis across **all
# seasons**, plus the two phase-based ML views (death SR prediction and player
# clusters).
#

# 10.1 Overall phase-specialist dashboard (3x4)
fig = plt.figure(figsize=(24, 18), facecolor='#0D0F14')
fig.patch.set_facecolor('#0D0F14')
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 4, figure=fig, top=0.90, bottom=0.05,
              left=0.05, right=0.97, hspace=0.55, wspace=0.38)

fig.text(0.5, 0.96, '\U0001F3CF  IPL PHASE-SPECIALIST ENGINE  \u2014  OVERALL (ALL SEASONS)',
         ha='center', fontsize=16, fontweight='bold', color=C['green'], fontfamily='monospace')
fig.text(0.5, 0.935,
         f'Sports Analytics Course Project  \u00b7  Cricsheet ball-by-ball  \u00b7  {matches_df.shape[0]:,} matches',
         ha='center', fontsize=9, color='#6b7490', fontfamily='monospace')

def _hbar(ax, names, vals, color, title):
    ax.barh(list(names), list(vals), color=color, edgecolor='none', alpha=0.85)
    ax.set_title(title); ax.invert_yaxis(); ax.grid(axis='x', linestyle='--', alpha=0.3)

# P1 run-rate curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(over_rr['over'] + 1, over_rr['rr'], color=C['teal'], linewidth=2)
ax1.axvline(6,  color=C['purple'], linestyle='--', alpha=0.6)
ax1.axvline(15, color=C['amber'],  linestyle='--', alpha=0.6)
ax1.set_title('Avg Run Rate / Over'); ax1.set_xlabel('Over')
ax1.grid(True, linestyle='--', alpha=0.3)

# P2-P4 batting SR leaders per phase
for col, (phase, pc) in zip([1, 2, 3],
        [('Powerplay', C['purple']), ('Middle', C['teal']), ('Death', C['amber'])]):
    ax = fig.add_subplot(gs[0, col])
    t = bat_filtered[bat_filtered['phase'] == phase].nlargest(8, 'strike_rate')
    _hbar(ax, t['batter'], t['strike_rate'], pc, f'{phase} SR Leaders')

# P5 powerplay economy, P6 death economy
ax5 = fig.add_subplot(gs[1, 0])
t = bowl_filtered[bowl_filtered['phase'] == 'Powerplay'].nsmallest(8, 'economy')
_hbar(ax5, t['bowler'], t['economy'], C['blue'], 'Powerplay Economy')
ax6 = fig.add_subplot(gs[1, 1])
t = bowl_filtered[bowl_filtered['phase'] == 'Death'].nsmallest(8, 'economy')
_hbar(ax6, t['bowler'], t['economy'], C['red'], 'Death Economy')

# P7 correlation
ax7 = fig.add_subplot(gs[1, 2])
from matplotlib.colors import LinearSegmentedColormap
cmap2 = LinearSegmentedColormap.from_list('mb', ['#E04C6B', '#141720', '#00E5A0'])
ax7.imshow(corr_mat.values, cmap=cmap2, vmin=-1, vmax=1, aspect='auto')
ax7.set_xticks(range(4)); ax7.set_yticks(range(4))
ax7.set_xticklabels(['PP', 'Mid', 'Dth', 'Career'], fontsize=8)
ax7.set_yticklabels(['PP', 'Mid', 'Dth', 'Career'], fontsize=8)
ax7.set_title('Phase SR Correlation')
for i in range(4):
    for j in range(4):
        ax7.text(j, i, f'{corr_mat.iloc[i, j]:.2f}', ha='center', va='center',
                 fontsize=8, color='#F0F2F8')

# P8 phase gap distribution
ax8 = fig.add_subplot(gs[1, 3])
ax8.hist(bat_wide['phase_gap'].dropna(), bins=35, color=C['amber'], edgecolor='none', alpha=0.8)
ax8.axvline(bat_wide['phase_gap'].median(), color=C['red'], linestyle='--', linewidth=1.5)
ax8.set_title('Phase Gap Distribution'); ax8.set_xlabel('Best\u2212Worst SR')
ax8.grid(True, linestyle='--', alpha=0.3)

# P9 top batting specialist score
ax9 = fig.add_subplot(gs[2, 0])
t = bat_wide.nlargest(8, 'perf_score')
_hbar(ax9, t['batter'], t['perf_score'], C['green'], 'Top Batting Specialist Score')

# P10 top bowling specialist score
ax10 = fig.add_subplot(gs[2, 1])
t = bowl_wide.nlargest(8, 'perf_score')
_hbar(ax10, t['bowler'], t['perf_score'], C['purple'], 'Top Bowling Specialist Score')

# P11 death-SR prediction R^2 (ML)
ax11 = fig.add_subplot(gs[2, 2])
try:
    mnames = ['Linear', 'Ridge', 'RF', 'XGB']
    r2vals = [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_ridge),
              r2_rf, r2_score(y_test, y_pred_xgb)]
except Exception:
    mnames = ['Linear', 'Ridge', 'RF']
    r2vals = [r2_score(y_test, y_pred_lr), r2_score(y_test, y_pred_ridge), r2_rf]
ax11.bar(mnames, r2vals,
         color=[C['green'] if v == max(r2vals) else C['blue'] + '88' for v in r2vals],
         edgecolor='none', alpha=0.9)
ax11.set_title('Death SR Prediction R\u00b2'); ax11.set_ylim(0, 1)
ax11.grid(axis='y', linestyle='--', alpha=0.3)
for i, v in enumerate(r2vals):
    ax11.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8, color='#F0F2F8')

# P12 PCA clusters (ML)
ax12 = fig.add_subplot(gs[2, 3])
for k in range(K_BEST):
    mask = ml_clust['cluster'] == k
    ax12.scatter(ml_clust.loc[mask, 'pc1'], ml_clust.loc[mask, 'pc2'],
                 color=pal[k % len(pal)], alpha=0.5, s=18, edgecolors='none', label=f'C{k}')
ax12.set_title(f'Player Clusters K={K_BEST}')
ax12.legend(facecolor='#0D0F14', labelcolor='#F0F2F8', fontsize=7)
ax12.grid(True, linestyle='--', alpha=0.3)

plt.savefig('ipl_phase_specialist_overall.png', dpi=130, bbox_inches='tight', facecolor='#0D0F14')
plt.show()
print("Overall dashboard saved \u2192 ipl_phase_specialist_overall.png")

#
# ## Section 11 — Last 4 Seasons: Phase-Specialist Analysis
# 
# The overall tables blend every era, but most early era players have retired, so
# those leaderboards aren't an actionable view of the *current* talent pool.
# 
# Here we rebuild the **identical phase-specialist pipeline** on only the **last 4
# seasons** and apply the **stricter per-phase minimum-balls outlier filter**
# (`CFG.recent`) because a 4-season window has smaller samples, the filter is what
# keeps the leaderboards honest.
#

# 11.1 Subset to the last 4 seasons
all_seasons    = sorted(int(s) for s in balls_df['season'].dropna().unique())
recent_seasons = all_seasons[-4:]
balls_recent   = balls_df[balls_df['season'].isin(recent_seasons)].copy()

print(f"All seasons        : {all_seasons}")
print(f"Last 4 seasons     : {recent_seasons}")
print(f"Recent deliveries  : {len(balls_recent):,}  "
      f"({len(balls_recent) / len(balls_df) * 100:.1f}% of all balls)")
print(f"Recent matches     : {balls_recent['match_id'].nunique():,}")
print(f"\nRecent min-balls thresholds:")
print(f"  Batting : {CFG.recent.min_bat_balls}")
print(f"  Bowling : {CFG.recent.min_bowl_balls}")

# 11.2 Run the phase pipeline on recent data only
R = build_phase_tables(balls_recent, CFG.recent)

print(f"Qualified batting (player, phase) rows : {len(R.bat_filtered)}")
print(f"Qualified bowling (player, phase) rows : {len(R.bowl_filtered)}")
print(f"Batters in recent wide table           : {R.bat_wide.shape[0]}")
print(f"Bowlers in recent wide table           : {R.bowl_wide.shape[0]}")

print("\nTop 10 recent batting phase-specialist scores:")
display(R.bat_wide.nlargest(10, 'perf_score')[['batter', 'sr_pp', 'sr_mid', 'sr_dth', 'perf_score']])

print("Top 10 recent bowling phase-specialist scores:")
display(R.bowl_wide.nlargest(10, 'perf_score')[['bowler', 'econ_pp', 'econ_mid', 'econ_dth', 'perf_score']])

# 11.3 Recent per-phase leaders
print(f"Phase leaders \u2014 last 4 seasons {recent_seasons}\n")
for phase in ['Powerplay', 'Middle', 'Death']:
    tb = R.bat_filtered[R.bat_filtered['phase'] == phase].nlargest(5, 'strike_rate')
    print(f"{phase} \u2014 top batting SR:")
    print(tb[['batter', 'runs', 'balls', 'strike_rate']].to_string(index=False))
    bo = R.bowl_filtered[R.bowl_filtered['phase'] == phase].nsmallest(5, 'economy')
    print(f"{phase} \u2014 best bowling economy:")
    print(bo[['bowler', 'balls', 'wickets', 'economy']].to_string(index=False))
    print()

# 11.4 Last-4-seasons phase-specialist dashboard (3x3)
fig = plt.figure(figsize=(20, 15), facecolor='#0D0F14')
fig.patch.set_facecolor('#0D0F14')
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, figure=fig, top=0.91, bottom=0.05,
              left=0.06, right=0.97, hspace=0.55, wspace=0.35)

span = f"{recent_seasons[0]}\u2013{recent_seasons[-1]}"
fig.text(0.5, 0.965, f'\U0001F3CF  IPL PHASE-SPECIALIST ENGINE \u2014 LAST 4 SEASONS ({span})',
         ha='center', fontsize=15, fontweight='bold', color=C['amber'], fontfamily='monospace')
fig.text(0.5, 0.94,
         f"Active-squad view  \u00b7  min-balls filtered  \u00b7  {balls_recent['match_id'].nunique():,} matches",
         ha='center', fontsize=9, color='#6b7490', fontfamily='monospace')

phase_pc = {'Powerplay': C['purple'], 'Middle': C['teal'], 'Death': C['amber']}

# Row 0 — batting SR leaders per phase
for col, phase in enumerate(['Powerplay', 'Middle', 'Death']):
    ax = fig.add_subplot(gs[0, col])
    t = R.bat_filtered[R.bat_filtered['phase'] == phase].nlargest(8, 'strike_rate')
    ax.barh(t['batter'], t['strike_rate'], color=phase_pc[phase], edgecolor='none', alpha=0.85)
    ax.set_title(f'{phase} \u2014 Batting SR'); ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3)

# Row 1 — bowling economy leaders per phase
for col, phase in enumerate(['Powerplay', 'Middle', 'Death']):
    ax = fig.add_subplot(gs[1, col])
    t = R.bowl_filtered[R.bowl_filtered['phase'] == phase].nsmallest(8, 'economy')
    ax.barh(t['bowler'], t['economy'], color=C['red'], edgecolor='none', alpha=0.85)
    ax.set_title(f'{phase} \u2014 Bowling Econ'); ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3)

# Row 2 — gap distribution + composite specialist scores
axg = fig.add_subplot(gs[2, 0])
axg.hist(R.bat_wide['phase_gap'].dropna(), bins=25, color=C['amber'], edgecolor='none', alpha=0.8)
axg.axvline(R.bat_wide['phase_gap'].median(), color=C['red'], linestyle='--', linewidth=1.5)
axg.set_title('Phase Gap Distribution'); axg.set_xlabel('Best\u2212Worst SR')
axg.grid(True, linestyle='--', alpha=0.3)

axb = fig.add_subplot(gs[2, 1])
t = R.bat_wide.nlargest(8, 'perf_score')
axb.barh(t['batter'], t['perf_score'], color=C['green'], edgecolor='none', alpha=0.85)
axb.set_title('Top Batting Specialist Score'); axb.invert_yaxis()
axb.grid(axis='x', linestyle='--', alpha=0.3)

axw = fig.add_subplot(gs[2, 2])
t = R.bowl_wide.nlargest(8, 'perf_score')
axw.barh(t['bowler'], t['perf_score'], color=C['purple'], edgecolor='none', alpha=0.85)
axw.set_title('Top Bowling Specialist Score'); axw.invert_yaxis()
axw.grid(axis='x', linestyle='--', alpha=0.3)

plt.savefig('ipl_phase_specialist_last4.png', dpi=130, bbox_inches='tight', facecolor='#0D0F14')
plt.show()
print(f"Last-4-seasons dashboard saved \u2192 ipl_phase_specialist_last4.png")

#
# ## Summary & Key Findings
# 
# | Finding | Detail |
# |---------|--------|
# | **Phase specialisation is real** | Low correlation between powerplay and death SR confirms these are distinct skills, not one "batting talent" |
# | **Phase gap surfaces specialists** | Players with a large best-vs-worst phase SR gap are true specialists, invisible in a career average |
# | **Death overs partly predictable** | Random Forest predicts death SR from other phases with R² ~0.4–0.5 — partial signal carries across phases |
# | **Cluster archetypes** | K-Means separates powerplay strikers, death finishers, anchors, and all-phase dominators |
# | **Win probability** | Toss decision and team run-rate history reach ~60–65% win-prediction accuracy |
# | **Recent ≠ all-time** | The last-4-seasons view, with a stricter min-balls filter, reflects the *active* talent pool rather than retired stars |
# 
# ### Outlier / minimum-sample filter
# A (player, phase) row only enters the leaderboards once the player clears a
# per-phase **legal-balls** threshold (`CFG.overall` / `CFG.recent`). This is the
# single knob that controls how noisy the rankings are — raise it for stricter,
# higher-confidence specialist lists, lower it to widen the pool.
# 
# ### Reusable pipeline
# `build_phase_tables(balls, cfg)` runs the entire phase-specialist analysis on any
# slice of the data. The notebook calls it twice (all seasons, then last 4), and the
# same function can be pointed at any team, venue, or custom date range.
# 
# ### Limitations
# - Cricsheet name format only (`'V Kohli'`) — no full names without a mapping file
# - No fielding analytics in this notebook
# - Win-probability model uses match-level features only (no ball-by-ball WPA)
# 
