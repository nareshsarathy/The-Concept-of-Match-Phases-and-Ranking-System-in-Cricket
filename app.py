# app.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Set page config
st.set_page_config(
    page_title="IPL Phase-Specialist Engine",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium dark styling
st.markdown("""
    <style>
    .main {
        background-color: #0D0F14;
    }
    h1, h2, h3 {
        color: #00E5A0 !important;
        font-family: 'Courier New', Courier, monospace;
    }
    .reportview-container {
        background: #0D0F14;
    }
    div[data-testid="stMetricValue"] {
        color: #00E5A0 !important;
        font-family: monospace;
    }
    div[data-testid="stMetricDelta"] {
        color: #F5A623 !important;
    }
    .stMetric {
        background-color: #141720 !important;
        padding: 15px !important;
        border-radius: 8px !important;
        border: 1px solid #1E2230 !important;
    }
    .stAlert {
        background-color: #141720 !important;
        color: #F0F2F8 !important;
        border: 1px solid #1E2230 !important;
    }
    /* Tabs customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #141720;
        border: 1px solid #1E2230;
        border-radius: 4px 4px 0px 0px;
        padding: 8px 16px;
        color: #8A92A6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00E5A0 !important;
        color: #0D0F14 !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Color Dictionary matches the Dark Theme configuration
C = {
    'green': '#00E5A0',
    'amber': '#F5A623',
    'red': '#E04C6B',
    'blue': '#4C8EE0',
    'purple': '#7C6AEE',
    'teal': '#3BC9DB',
    'muted': '#3A4060'
}

# DATA LOADING WITH CACHING
@st.cache_data
def load_data():
    balls_df = pd.read_parquet("processed_data/balls_df.parquet")
    matches_df = pd.read_csv("processed_data/matches_df.csv")
    bat_wide_raw = pd.read_csv("processed_data/bat_wide.csv")
    bowl_wide_raw = pd.read_csv("processed_data/bowl_wide.csv")
    venue_stats = pd.read_csv("processed_data/venue_stats.csv")
    matchups = pd.read_csv("processed_data/match_ups.csv")
    
    with open("processed_data/classifier_metrics.json", "r") as f:
        clf_metrics = json.load(f)
        
    return balls_df, matches_df, bat_wide_raw, bowl_wide_raw, venue_stats, matchups, clf_metrics

@st.cache_resource
def load_models():
    model_rf = joblib.load("processed_data/model_rf.joblib")
    model_win = joblib.load("processed_data/model_win.joblib")
    clf_career = joblib.load("processed_data/classifier_career.joblib")
    clf_phase = joblib.load("processed_data/classifier_phase.joblib")
    return model_rf, model_win, clf_career, clf_phase

try:
    balls_df, matches_df, bat_wide_raw, bowl_wide_raw, venue_stats, matchups, clf_metrics = load_data()
    model_rf, model_win, clf_career, clf_phase = load_models()
except Exception as e:
    st.error(f"Error loading precomputed files. Please ensure you ran the pipeline first: {e}")
    st.stop()

# SIDEBAR CONFIGURATION & GLOBAL FILTERING
st.sidebar.markdown(f"<h2 style='color:{C['green']};'>🏏 Config Panel</h2>", unsafe_allow_html=True)

# Season Filter
all_seasons = sorted(balls_df['season'].dropna().unique())
season_mode = st.sidebar.radio("Seasons Analysis Scope", ["All Seasons (2008 - 2026)", "Last 4 Seasons (Active Talent)"])

if "Last 4 Seasons" in season_mode:
    recent_seasons = all_seasons[-4:]
    active_balls = balls_df[balls_df['season'].isin(recent_seasons)].copy()
    active_matches = matches_df[matches_df['season'].isin(recent_seasons)].copy()
    st.sidebar.info(f"Analyzing Seasons: {recent_seasons}")
    # Adjust default min-balls dynamically for a smaller sample
    min_pp_b = st.sidebar.slider("Min Powerplay Balls Faced/Bowled", 10, 150, 60)
    min_mid_b = st.sidebar.slider("Min Middle Balls Faced/Bowled", 10, 200, 90)
    min_dth_b = st.sidebar.slider("Min Death Balls Faced/Bowled", 10, 150, 50)
else:
    active_balls = balls_df.copy()
    active_matches = matches_df.copy()
    st.sidebar.info(f"Analyzing all seasons: {all_seasons[0]} - {all_seasons[-1]}")
    # High filters for complete historical dataset to avoid noise
    min_pp_b = st.sidebar.slider("Min Powerplay Balls Faced/Bowled", 50, 400, 222)
    min_mid_b = st.sidebar.slider("Min Middle Balls Faced/Bowled", 50, 500, 294)
    min_dth_b = st.sidebar.slider("Min Death Balls Faced/Bowled", 50, 300, 162)

# Recompute wide phase stats dynamically based on sliders
@st.cache_data
def get_filtered_phase_tables(balls_subset, min_pp, min_mid, min_dth):
    legal = balls_subset[balls_subset['is_legal']].copy()
    
    # Career summaries on this subset
    career_bat = (legal.groupby('batter')
                  .agg(runs=('runs_bat', 'sum'), balls=('is_legal', 'sum'),
                       fours=('is_four', 'sum'), sixes=('is_six', 'sum'),
                       matches=('match_id', 'nunique')).reset_index())
    career_bat['career_sr'] = (career_bat['runs'] / career_bat['balls'] * 100).round(2)

    career_bowl = (legal.groupby('bowler')
                   .agg(runs_c=('runs_total', 'sum'), balls=('is_legal', 'sum'),
                        wickets=('bowler_wkt', 'sum'), matches=('match_id', 'nunique')).reset_index())
    career_bowl['career_econ'] = (career_bowl['runs_c'] / career_bowl['balls'] * 6).round(2)

    # Batting
    bat_phase = (legal.groupby(['batter', 'phase'])
                 .agg(runs=('runs_bat', 'sum'), balls=('is_legal', 'sum'),
                      fours=('is_four', 'sum'), sixes=('is_six', 'sum'),
                      dots=('is_dot', 'sum'), innings=('match_id', 'nunique')).reset_index())
    dismissals = (balls_subset[balls_subset['is_wicket'] & balls_subset['player_out'].notna()]
                  .groupby(['player_out', 'phase']).size().reset_index(name='dismissals')
                  .rename(columns={'player_out': 'batter'}))
    bat_phase = bat_phase.merge(dismissals, on=['batter', 'phase'], how='left').fillna(0)
    bat_phase['strike_rate'] = (bat_phase['runs'] / bat_phase['balls'] * 100).round(2)
    bat_phase['boundary_pct'] = ((bat_phase['fours'] * 4 + bat_phase['sixes'] * 6) / bat_phase['runs'].clip(lower=1) * 100).round(1)

    # Bowling
    bowl_phase = (legal.groupby(['bowler', 'phase'])
                  .agg(runs_c=('runs_total', 'sum'), balls=('is_legal', 'sum'),
                       wickets=('bowler_wkt', 'sum'), dots=('is_dot', 'sum'),
                       innings=('match_id', 'nunique')).reset_index())
    bowl_phase['economy'] = (bowl_phase['runs_c'] / bowl_phase['balls'] * 6).round(2)
    bowl_phase['dot_pct'] = (bowl_phase['dots'] / bowl_phase['balls'] * 100).round(1)

    # Filters
    bat_limits = {'Powerplay': min_pp, 'Middle': min_mid, 'Death': min_dth}
    bowl_limits = {'Powerplay': min_pp, 'Middle': min_mid, 'Death': min_dth}
    
    bat_filt = bat_phase[bat_phase.apply(lambda r: r['balls'] >= bat_limits.get(r['phase'], 0), axis=1)].copy()
    bowl_filt = bowl_phase[bowl_phase.apply(lambda r: r['balls'] >= bowl_limits.get(r['phase'], 0), axis=1)].copy()

    # Pivots
    bat_w = (bat_filt.pivot_table(index='batter', columns='phase', values='strike_rate', aggfunc='first').reset_index())
    bat_w.columns.name = None
    bat_w = bat_w.rename(columns={'Powerplay': 'sr_pp', 'Middle': 'sr_mid', 'Death': 'sr_dth'})
    for c in ['sr_pp', 'sr_mid', 'sr_dth']:
        if c not in bat_w.columns: bat_w[c] = np.nan
    bat_w = bat_w.merge(career_bat[['batter', 'runs', 'career_sr', 'matches']], on='batter', how='inner')

    bowl_w = (bowl_filt.pivot_table(index='bowler', columns='phase', values='economy', aggfunc='first').reset_index())
    bowl_w.columns.name = None
    bowl_w = bowl_w.rename(columns={'Powerplay': 'econ_pp', 'Middle': 'econ_mid', 'Death': 'econ_dth'})
    for c in ['econ_pp', 'econ_mid', 'econ_dth']:
        if c not in bowl_w.columns: bowl_w[c] = np.nan
    bowl_w = bowl_w.merge(career_bowl[['bowler', 'wickets', 'career_econ', 'matches']], on='bowler', how='inner')

    # Scaling & Scores
    scaler = MinMaxScaler(feature_range=(0, 100))
    for col in ['sr_pp', 'sr_mid', 'sr_dth', 'career_sr']:
        valid = bat_w[col].notna()
        if valid.sum() >= 2:
            bat_w.loc[valid, f'{col}_n'] = scaler.fit_transform(bat_w.loc[valid, [col]])
    for col in ['econ_pp', 'econ_mid', 'econ_dth', 'career_econ']:
        valid = bowl_w[col].notna()
        if valid.sum() >= 2:
            inv = -bowl_w.loc[valid, col]
            bowl_w.loc[valid, f'{col}_n'] = scaler.fit_transform(inv.values.reshape(-1, 1))

    # Weight scores
    bat_w['perf_score'] = bat_w.apply(lambda r: round(
        0.35 * r.get('sr_pp_n', 50.0) + 0.30 * r.get('sr_mid_n', 50.0) + 0.35 * r.get('sr_dth_n', 50.0), 2), axis=1)
    bowl_w['perf_score'] = bowl_w.apply(lambda r: round(
        0.30 * r.get('econ_pp_n', 50.0) + 0.25 * r.get('econ_mid_n', 50.0) + 0.45 * r.get('econ_dth_n', 50.0), 2), axis=1)

    # Phase gaps
    bat_w['sr_best'] = bat_w[['sr_pp', 'sr_mid', 'sr_dth']].max(axis=1)
    bat_w['sr_worst'] = bat_w[['sr_pp', 'sr_mid', 'sr_dth']].min(axis=1)
    bat_w['phase_gap'] = (bat_w['sr_best'] - bat_w['sr_worst']).round(1)

    bowl_w['econ_best'] = bowl_w[['econ_pp', 'econ_mid', 'econ_dth']].min(axis=1)
    bowl_w['econ_worst'] = bowl_w[['econ_pp', 'econ_mid', 'econ_dth']].max(axis=1)
    bowl_w['phase_gap'] = (bowl_w['econ_worst'] - bowl_w['econ_best']).round(1)

    return bat_w, bowl_w, bat_filt, bowl_filt

bat_wide, bowl_wide, bat_filtered, bowl_filtered = get_filtered_phase_tables(
    active_balls, min_pp_b, min_mid_b, min_dth_b
)

# Title Header
st.markdown(f"<h1 style='text-align: center; color:{C['green']};'>🏏 IPL PHASE-SPECIALIST ENGINE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color:#8A92A6; font-size:14px; font-family:monospace;'>"
            "Universität des Saarlandes | Seminar: Sports Analytics</p>", unsafe_allow_html=True)
st.markdown("---")

# Main Tabs Setup
tabs = st.tabs([
    "🏛️ Background",
    "🔍 Player Explorer",
    "🏆 Leaderboards",
    "🏟️ Venue Analytics",
    "⚔️ H2H Matchups",
    "📈 Era Evolution",
    "🧪 Classifier Validation",
    "🤖 ML Playground"
])

# TAB 1: PRESENTATION BACKGROUND & CONTEXT
with tabs[0]:
    st.markdown(f"<h3 style='color:{C['green']};'>Why Look Beyond Career Averages?</h3>", unsafe_allow_html=True)
    
    col_bg1, col_bg2 = st.columns([1, 1])
    
    with col_bg1:
        st.markdown(f"<div style='background-color:#141720; padding:15px; border-radius:8px; border:1px solid #1E2230; margin-bottom:12px;'>"
                    f"<h4 style='color:{C['amber']}; margin-top:0;'>📖 The Blended Average Problem</h4>"
                    f"A career strike rate is a single blended number that erases specialization. "
                    f"A batsman with a career SR of 130 could actually be an elite <b>165 SR opener</b> in the Powerplay, "
                    f"but only a <b>95 SR hitter</b> in the Death overs (or vice versa)."
                    f"</div>", unsafe_allow_html=True)
                    
        st.markdown(f"<div style='background-color:#141720; padding:15px; border-radius:8px; border:1px solid #1E2230; margin-bottom:12px;'>"
                    f"<h4 style='color:{C['purple']}; margin-top:0;'>🛡️ Playing Conditions & ICC Rules</h4>"
                    f"The ICC Men's T20I conditions restrict fielders outside the 30-yard circle to just <b>two</b> during Overs 1-6 (Powerplay), "
                    f"rising to <b>five</b> from Over 7 onward. The Powerplay/non-Powerplay boundary is a hard fielding-restriction rule, not a modeling choice."
                    f"</div>", unsafe_allow_html=True)
        
        # Formulas
        st.latex(r"SR = \frac{Runs\ Scored}{Balls\ Faced} \times 100")
        st.latex(r"Econ = \frac{Runs\ Conceded}{Balls\ Bowled} \times 6")
        
    with col_bg2:
        # Visual Proof: Career SR vs. Death Over SR Scatter
        clean_bat = bat_wide.dropna(subset=['career_sr', 'sr_dth', 'phase_gap'])
        fig_scat = px.scatter(
            clean_bat,
            x='career_sr',
            y='sr_dth',
            color='phase_gap',
            hover_name='batter',
            hover_data=['runs', 'matches', 'sr_pp', 'sr_mid'],
            title='Visual Proof: Career SR vs. Death Over SR',
            labels={'career_sr': 'Overall Career Strike Rate', 'sr_dth': 'Death Over Strike Rate (Overs 16-20)', 'phase_gap': 'Phase Gap (Max - Min SR)'},
            color_continuous_scale=px.colors.sequential.Viridis
        )
        # Add diagonal y=x line
        fig_scat.add_shape(
            type='line',
            x0=100, y0=100, x1=210, y1=210,
            line=dict(color=C['amber'], dash='dash', width=2),
            name='Career = Death SR'
        )
        fig_scat.update_layout(
            paper_bgcolor='#0D0F14',
            plot_bgcolor='#0D0F14',
            font=dict(color='#F0F2F8'),
            height=380,
            coloraxis_colorbar=dict(title='Specialization Gap')
        )
        st.plotly_chart(fig_scat, use_container_width=True)

# TAB 2: INTERACTIVE PLAYER EXPLORER
with tabs[1]:
    st.markdown("### Compare Player Metrics Against Tournament Averages")
    
    role = st.radio("Choose Role to Explore", ["Batsman", "Bowler"], horizontal=True)
    
    if role == "Batsman":
        player_list = sorted(bat_wide['batter'].unique())
        selected_player = st.selectbox("Select Batsman", player_list)
        
        player_row = bat_wide[bat_wide['batter'] == selected_player].iloc[0]
        
        # Calculate tournament averages
        avg_pp = bat_wide['sr_pp'].mean()
        avg_mid = bat_wide['sr_mid'].mean()
        avg_dth = bat_wide['sr_dth'].mean()
        
        val_pp = player_row['sr_pp'] if not pd.isna(player_row['sr_pp']) else 100.0
        val_mid = player_row['sr_mid'] if not pd.isna(player_row['sr_mid']) else 100.0
        val_dth = player_row['sr_dth'] if not pd.isna(player_row['sr_dth']) else 100.0
        
        # Plotly Radar Chart
        categories = ['Powerplay SR', 'Middle SR', 'Death SR']
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[val_pp, val_mid, val_dth],
            theta=categories,
            fill='toself',
            name=selected_player,
            fillcolor='rgba(0, 229, 160, 0.3)',
            line=dict(color=C['green'], width=2)
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[avg_pp, avg_mid, avg_dth],
            theta=categories,
            fill='toself',
            name='Tournament Avg',
            fillcolor='rgba(245, 166, 35, 0.1)',
            line=dict(color=C['amber'], width=1.5, dash='dash')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[80, 220], gridcolor='#1E2230'),
                angularaxis=dict(gridcolor='#1E2230')
            ),
            showlegend=True,
            paper_bgcolor='#0D0F14',
            plot_bgcolor='#0D0F14',
            font=dict(color='#F0F2F8')
        )
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Composite Specialist Score", f"{player_row['perf_score']:.1f}")
        m_col2.metric("Strike Rate (PP)", f"{player_row['sr_pp']:.1f}" if not pd.isna(player_row['sr_pp']) else "N/A")
        m_col3.metric("Strike Rate (Middle)", f"{player_row['sr_mid']:.1f}" if not pd.isna(player_row['sr_mid']) else "N/A")
        m_col4.metric("Strike Rate (Death)", f"{player_row['sr_dth']:.1f}" if not pd.isna(player_row['sr_dth']) else "N/A")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"#### 📊 Profile Analysis: **{selected_player}**")
            st.write(f"- **Career Runs**: {int(player_row['runs'])}")
            st.write(f"- **Career Strike Rate**: {player_row['career_sr']:.2f}")
            st.write(f"- **Matches Played**: {int(player_row['matches'])}")
            st.write(f"- **Phase Gap (Max - Min SR)**: {player_row['phase_gap']:.1f} runs/100b")
            
            if player_row['phase_gap'] > 50:
                st.warning(f"**Highly Specialized**: This player has a large phase gap ({player_row['phase_gap']:.1f} SR gap). They perform best in one segment and struggle in others.")
            else:
                st.success(f"**Balanced Anchor**: This player maintains consistent strike rates across phases (gap of only {player_row['phase_gap']:.1f}).")
                
    else:
        player_list = sorted(bowl_wide['bowler'].unique())
        selected_player = st.selectbox("Select Bowler", player_list)
        
        player_row = bowl_wide[bowl_wide['bowler'] == selected_player].iloc[0]
        
        avg_pp = bowl_wide['econ_pp'].mean()
        avg_mid = bowl_wide['econ_mid'].mean()
        avg_dth = bowl_wide['econ_dth'].mean()
        
        val_pp = player_row['econ_pp'] if not pd.isna(player_row['econ_pp']) else 9.0
        val_mid = player_row['econ_mid'] if not pd.isna(player_row['econ_mid']) else 8.0
        val_dth = player_row['econ_dth'] if not pd.isna(player_row['econ_dth']) else 10.0
        
        # Plotly Radar Chart (Inverted logic: lower economy is closer to outer edge)
        # To show lower values as better on radar, we can subtract from a maximum constant (e.g. 15)
        categories = ['Powerplay Econ', 'Middle Econ', 'Death Econ']
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=[15 - val_pp, 15 - val_mid, 15 - val_dth],
            theta=categories,
            fill='toself',
            name=selected_player,
            fillcolor='rgba(124, 106, 238, 0.3)',
            line=dict(color=C['purple'], width=2)
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=[15 - avg_pp, 15 - avg_mid, 15 - avg_dth],
            theta=categories,
            fill='toself',
            name='Tournament Avg',
            fillcolor='rgba(245, 166, 35, 0.1)',
            line=dict(color=C['amber'], width=1.5, dash='dash')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[2, 11], tickvals=[3,5,7,9], 
                                ticktext=['12.0 Econ', '10.0 Econ', '8.0 Econ', '6.0 Econ'], gridcolor='#1E2230'),
                angularaxis=dict(gridcolor='#1E2230')
            ),
            showlegend=True,
            paper_bgcolor='#0D0F14',
            plot_bgcolor='#0D0F14',
            font=dict(color='#F0F2F8')
        )
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Composite Specialist Score", f"{player_row['perf_score']:.1f}")
        m_col2.metric("Economy (PP)", f"{player_row['econ_pp']:.2f}" if not pd.isna(player_row['econ_pp']) else "N/A")
        m_col3.metric("Economy (Middle)", f"{player_row['econ_mid']:.2f}" if not pd.isna(player_row['econ_mid']) else "N/A")
        m_col4.metric("Economy (Death)", f"{player_row['econ_dth']:.2f}" if not pd.isna(player_row['econ_dth']) else "N/A")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown(f"#### 📊 Profile Analysis: **{selected_player}**")
            st.write(f"- **Career Wickets**: {int(player_row['wickets'])}")
            st.write(f"- **Career Economy**: {player_row['career_econ']:.2f}")
            st.write(f"- **Matches Played**: {int(player_row['matches'])}")
            st.write(f"- **Econ Gap (Worst - Best)**: {player_row['phase_gap']:.2f} runs/over")
            
            if player_row['phase_gap'] > 2.5:
                st.warning(f"**Highly Specialized**: This bowler has a large phase gap ({player_row['phase_gap']:.2f} runs/over difference). They are optimized for a specific segment.")
            else:
                st.success(f"**All-Phase Bowler**: They maintain a consistent econ ceiling (only {player_row['phase_gap']:.2f} variance).")

# TAB 3: INTERACTIVE LEADERBOARDS
with tabs[2]:
    st.markdown("### Top Performers Filtered Dynamically by Min-Balls & Seasons")
    
    lead_col1, lead_col2 = st.columns(2)
    
    with lead_col1:
        st.markdown(f"<h4 style='color:{C['green']};'>🔥 Batting Strike Rate Leaders</h4>", unsafe_allow_html=True)
        p_select = st.selectbox("Select Phase (Batting)", ["Powerplay", "Middle", "Death"])
        tb = bat_filtered[bat_filtered['phase'] == p_select].nlargest(10, 'strike_rate')
        st.dataframe(
            tb[['batter', 'runs', 'balls', 'strike_rate', 'boundary_pct']].reset_index(drop=True),
            use_container_width=True
        )
        
    with lead_col2:
        st.markdown(f"<h4 style='color:{C['red']};'>🎯 Bowling Economy Leaders (Tidy spells)</h4>", unsafe_allow_html=True)
        b_select = st.selectbox("Select Phase (Bowling)", ["Powerplay", "Middle", "Death"])
        bo = bowl_filtered[bowl_filtered['phase'] == b_select].nsmallest(10, 'economy')
        st.dataframe(
            bo[['bowler', 'balls', 'wickets', 'economy', 'dot_pct']].reset_index(drop=True),
            use_container_width=True
        )

# TAB 4: VENUE-SPECIFIC PHASE ANALYZER
with tabs[3]:
    st.markdown("### How Stadium Conditions Shift Phase Dynamics")
    
    selected_venue = st.selectbox("Select IPL Venue/Stadium", sorted(venue_stats['venue'].unique()))
    venue_subset = venue_stats[venue_stats['venue'] == selected_venue]
    
    if len(venue_subset) > 0:
        pitch_type = venue_subset['pitch_type'].iloc[0]
        st.markdown(f"**Pitch Characterization**: `{pitch_type}`")
        
        # Display comparison
        fig_v = px.bar(
            venue_subset,
            x='phase',
            y='run_rate',
            color='phase',
            title=f"Run Rate per Phase at {selected_venue}",
            color_discrete_map={'Powerplay': C['purple'], 'Middle': C['teal'], 'Death': C['amber']}
        )
        fig_v.update_layout(paper_bgcolor='#0D0F14', plot_bgcolor='#0D0F14', font=dict(color='#F0F2F8'))
        
        col_v1, col_v2 = st.columns([2, 1])
        with col_v1:
            st.plotly_chart(fig_v, use_container_width=True)
        with col_v2:
            st.markdown("#### Venue Metrics Breakdown")
            for _, row in venue_subset.iterrows():
                st.markdown(f"**{row['phase']}**:")
                st.markdown(f"- Avg Run Rate: `{row['run_rate']:.2f}` (Runs/Over)")
                st.markdown(f"- Boundary Rate: `{row['boundary_rate']:.2f}%` (percentage of balls hit for 4/6)")
                st.markdown(f"- Wickets Fallen: `{int(row['wickets'])}` wickets")
                st.markdown("---")

# TAB 5: BOWLER-VS-BATSMAN MATCHUP SIMULATOR
with tabs[4]:
    st.markdown("### Simulated Head-to-Head Headaches")
    
    m_bat_list = sorted(matchups['batter'].unique())
    m_bowl_list = sorted(matchups['bowler'].unique())
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    
    with sim_col1:
        sel_batter = st.selectbox("Select Batter", m_bat_list, index=m_bat_list.index("V Kohli") if "V Kohli" in m_bat_list else 0)
    with sim_col2:
        sel_bowler = st.selectbox("Select Bowler", m_bowl_list, index=m_bowl_list.index("JJ Bumrah") if "JJ Bumrah" in m_bowl_list else 0)
    with sim_col3:
        sel_phase = st.selectbox("Select Overs Phase", ["Powerplay", "Middle", "Death"])

    # Historical direct match records
    hist = matchups[(matchups['batter'] == sel_batter) & (matchups['bowler'] == sel_bowler)]
    
    # Extract general phase strengths
    bat_p = bat_wide[bat_wide['batter'] == sel_batter]
    bowl_p = bowl_wide[bowl_wide['bowler'] == sel_bowler]
    
    b_sr = bat_p[f'sr_{"pp" if sel_phase=="Powerplay" else "mid" if sel_phase=="Middle" else "dth"}'].values
    w_econ = bowl_p[f'econ_{"pp" if sel_phase=="Powerplay" else "mid" if sel_phase=="Middle" else "dth"}'].values
    
    # Default values if no history exists
    batter_sr = b_sr[0] if len(b_sr) > 0 and not pd.isna(b_sr[0]) else 125.0
    bowler_econ = w_econ[0] if len(w_econ) > 0 and not pd.isna(w_econ[0]) else 8.2
    
    # direct head to head calculation
    direct_balls = hist['balls'].values[0] if len(hist) > 0 else 0
    direct_runs = hist['runs'].values[0] if len(hist) > 0 else 0
    direct_outs = hist['dismissals'].values[0] if len(hist) > 0 else 0
    direct_sr = (direct_runs / direct_balls * 100) if direct_balls > 0 else batter_sr
    
    # Bayesian matchup probability logic
    # combine head-to-head weight based on sample size
    w1 = min(direct_balls / 20.0, 0.40) # cap H2H weight at 40%
    w2 = 0.40 # general batsman phase strength
    w3 = 0.20 # general bowler economy translated to SR (econ * 12)
    
    sim_sr = w1 * direct_sr + w2 * batter_sr + w3 * (bowler_econ * 15.0)
    
    # Outcome probabilities based on estimated SR
    boundary_prob = min(max(sim_sr / 1200.0, 0.05), 0.35)
    wicket_prob = min(max(0.02 + (direct_outs * 0.05) + (bowler_econ * 0.002), 0.01), 0.08)
    dot_prob = min(max(0.50 - (sim_sr / 600.0), 0.20), 0.60)
    single_prob = 1.0 - (boundary_prob + wicket_prob + dot_prob)
    
    st.markdown(f"#### 📊 Simulated Match-up Outlook in the `{sel_phase}` Overs:")
    
    fig_sim = go.Figure(go.Bar(
        x=[dot_prob*100, single_prob*100, boundary_prob*100, wicket_prob*100],
        y=['Dot Ball %', 'Single/Double %', 'Boundary (4/6) %', 'Wicket Dismissal %'],
        orientation='h',
        marker_color=[C['muted'], C['blue'], C['green'], C['red']]
    ))
    fig_sim.update_layout(
        xaxis=dict(title='Probability (%)', range=[0, 100]),
        paper_bgcolor='#0D0F14',
        plot_bgcolor='#0D0F14',
        font=dict(color='#F0F2F8'),
        height=300
    )
    
    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        st.plotly_chart(fig_sim, use_container_width=True)
    with col_s2:
        st.markdown("<div style='background-color:#141720; padding:15px; border-radius:8px; border:1px solid #1E2230;'>", unsafe_allow_html=True)
        st.markdown(f"**Direct head-to-head records:**")
        st.markdown(f"- Direct Deliveries: `{int(direct_balls)}` balls")
        st.markdown(f"- Runs Scored: `{int(direct_runs)}` runs")
        st.markdown(f"- Times Out: `{int(direct_outs)}` dismissals")
        st.markdown(f"- Head-to-head SR: `{direct_sr:.2f}`")
        st.markdown("</div>", unsafe_allow_html=True)

# TAB 6: ERA EVOLUTION TIMELINE
with tabs[5]:
    st.markdown("### How the Game Evolved (2008 to 2026)")
    
    # Recompile seasonal splits
    legal_b = active_balls[active_balls['is_legal']].copy()
    season_phase_rr = (legal_b.groupby(['season', 'phase'])
                       .agg(runs_c=('runs_total', 'sum'), balls=('is_legal', 'sum'))
                       .reset_index())
    season_phase_rr['run_rate'] = (season_phase_rr['runs_c'] / season_phase_rr['balls'] * 6).round(2)
    
    # Pivot for plotting
    season_pivot = season_phase_rr.pivot(index='season', columns='phase', values='run_rate').reset_index()
    
    fig_era = go.Figure()
    fig_era.add_trace(go.Scatter(x=season_pivot['season'], y=season_pivot['Powerplay'], mode='lines+markers', name='Powerplay (Overs 1-6)', line=dict(color=C['purple'], width=2)))
    fig_era.add_trace(go.Scatter(x=season_pivot['season'], y=season_pivot['Middle'], mode='lines+markers', name='Middle (Overs 7-15)', line=dict(color=C['teal'], width=2)))
    fig_era.add_trace(go.Scatter(x=season_pivot['season'], y=season_pivot['Death'], mode='lines+markers', name='Death (Overs 16-20)', line=dict(color=C['amber'], width=2)))
    
    fig_era.update_layout(
        title="Average Run Rates per Phase Over Years",
        xaxis=dict(tickmode='linear', gridcolor='#1E2230'),
        yaxis=dict(title='Run Rate (runs/over)', gridcolor='#1E2230'),
        paper_bgcolor='#0D0F14',
        plot_bgcolor='#0D0F14',
        font=dict(color='#F0F2F8'),
        height=450
    )
    
    st.plotly_chart(fig_era, use_container_width=True)
    st.markdown(
        f"**Analytics Insights**:\n"
        f"- The **Death Run Rate** has dramatically climbed over the years. Batting depth and hitting range are significantly stronger compared to 2008.\n"
        f"- **Powerplay Run Rate spikes**: The introduction of tactics like using pinch-hitters and rules such as the **Impact Player rule** has allowed teams to utilize riskier strategies in early overs, pushing Powerplay scoring rate limits higher."
    )

# TAB 7: CONFUSION MATRIX CLASSIFIER VALIDATION
with tabs[6]:
    st.markdown(f"<h3 style='color:{C['green']};'>Career-only vs. Phase-wise Specialist Classifier</h3>", unsafe_allow_html=True)
    st.write(
        "To mathematically validate why career averages drag down specialization, we trained two classifiers "
        "to identify **True Death-overs Specialists** (Death Over SR in top 25%, i.e., "
        f"$\ge {clf_metrics['threshold']}$)."
    )
    
    # Visual comparison of metrics
    metrics_df = pd.DataFrame([
        {'Model': 'Career Baseline (Model A)', 'Metric': 'Accuracy', 'Value': clf_metrics['career']['accuracy']},
        {'Model': 'Career Baseline (Model A)', 'Metric': 'Precision', 'Value': clf_metrics['career']['precision']},
        {'Model': 'Career Baseline (Model A)', 'Metric': 'Recall', 'Value': clf_metrics['career']['recall']},
        {'Model': 'Career Baseline (Model A)', 'Metric': 'F1-Score', 'Value': clf_metrics['career']['f1']},
        {'Model': 'Phase-wise Proposed (Model B)', 'Metric': 'Accuracy', 'Value': clf_metrics['phase']['accuracy']},
        {'Model': 'Phase-wise Proposed (Model B)', 'Metric': 'Precision', 'Value': clf_metrics['phase']['precision']},
        {'Model': 'Phase-wise Proposed (Model B)', 'Metric': 'Recall', 'Value': clf_metrics['phase']['recall']},
        {'Model': 'Phase-wise Proposed (Model B)', 'Metric': 'F1-Score', 'Value': clf_metrics['phase']['f1']}
    ])
    
    fig_metrics = px.bar(
        metrics_df,
        x='Metric',
        y='Value',
        color='Model',
        barmode='group',
        title='Model Performance: Career Average vs. Phase-Segregated Features',
        color_discrete_map={'Career Baseline (Model A)': C['amber'], 'Phase-wise Proposed (Model B)': C['green']}
    )
    fig_metrics.update_layout(
        paper_bgcolor='#0D0F14',
        plot_bgcolor='#0D0F14',
        font=dict(color='#F0F2F8'),
        yaxis=dict(range=[0, 1.05], title='Score'),
        height=320
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Side-by-side Confusion Matrices
    col_c, col_p = st.columns(2)
    
    with col_c:
        st.markdown(f"<h4 style='color:{C['amber']}; text-align:center;'>Model A: Career Baseline (Confusion Matrix)</h4>", unsafe_allow_html=True)
        cm_career = np.array(clf_metrics['career']['cm'])
        fig_cm_c = px.imshow(
            cm_career,
            labels=dict(x="Predicted", y="Actual"),
            x=['Normal Batter', 'Death Specialist'],
            y=['Normal Batter', 'Death Specialist'],
            color_continuous_scale=[[0.0, '#141720'], [1.0, C['amber']]],
            text_auto=True
        )
        fig_cm_c.update_layout(width=280, height=280, coloraxis_showscale=False, paper_bgcolor='#0D0F14', plot_bgcolor='#0D0F14', font=dict(color='#F0F2F8'))
        st.plotly_chart(fig_cm_c, use_container_width=True)
        
    with col_p:
        st.markdown(f"<h4 style='color:{C['green']}; text-align:center;'>Model B: Phase-wise Proposed (Confusion Matrix)</h4>", unsafe_allow_html=True)
        cm_phase = np.array(clf_metrics['phase']['cm'])
        fig_cm_p = px.imshow(
            cm_phase,
            labels=dict(x="Predicted", y="Actual"),
            x=['Normal Batter', 'Death Specialist'],
            y=['Normal Batter', 'Death Specialist'],
            color_continuous_scale=[[0.0, '#141720'], [1.0, C['green']]],
            text_auto=True
        )
        fig_cm_p.update_layout(width=280, height=280, coloraxis_showscale=False, paper_bgcolor='#0D0F14', plot_bgcolor='#0D0F14', font=dict(color='#F0F2F8'))
        st.plotly_chart(fig_cm_p, use_container_width=True)

    # Detailed sports analytics write up
    st.markdown(
        f"<div style='background-color:#141720; padding:15px; border-radius:8px; border:1px solid #1E2230; margin-top:15px;'>"
        f"<h4 style='color:{C['green']}; margin-top:0;'>🔬 Key Sports Analytics Takeaway</h4>"
        f"Observe the **Recall (Sensitivity)**. Model A (Career-only) has very low recall (suffering from many <b>False Negatives</b>). "
        f"It misses true death finishers because their slower batting in the middle overs drags down their overall career averages, making them blend in. "
        f"In contrast, Model B correctly classifies them by evaluating phase-specific performance independently."
        f"</div>", unsafe_allow_html=True
    )

# TAB 8: ML PLAYGROUND (REGRESSOR & CLUSTERING)
with tabs[7]:
    st.markdown("### Interactive Machine Learning Models")
    
    ml_sub_tab1, ml_sub_tab2, ml_sub_tab3 = st.tabs([
        "🔮 Death SR Predictor",
        "🎯 K-Means Clusters",
        "⚖️ Live Win Probability"
    ])
    
    # Sub-tab 1: Regressor
    with ml_sub_tab1:
        st.markdown("#### Estimate Death Over Strike Rate based on other overs")
        
        # User input sliders
        input_pp_sr = st.slider("Hypothetical Powerplay SR", 80, 220, 130)
        input_mid_sr = st.slider("Hypothetical Middle Over SR", 80, 220, 120)
        input_career_sr = st.slider("Hypothetical Career SR", 80, 200, 125)
        input_runs = st.number_input("Career Runs", min_value=10, max_value=8000, value=1500)
        input_matches = st.number_input("Matches Played", min_value=1, max_value=300, value=50)
        
        input_log_runs = np.log1p(input_runs)
        
        # Make prediction
        pred_input = pd.DataFrame([{
            'sr_pp': input_pp_sr,
            'sr_mid': input_mid_sr,
            'career_sr': input_career_sr,
            'log_runs': input_log_runs,
            'matches': input_matches
        }])
        
        pred_dth_sr = model_rf.predict(pred_input)[0]
        st.metric("Predicted Death Strike Rate", f"{pred_dth_sr:.2f}")
        
    # Sub-tab 2: Clustering
    with ml_sub_tab2:
        st.markdown("#### K-Means Clustering on Phase Strike Rates")
        
        clust_feats = ['sr_pp', 'sr_mid', 'sr_dth', 'career_sr', 'matches']
        ml_clust = bat_wide.dropna(subset=clust_feats).copy()
        
        # Standardize features
        scaler_c = StandardScaler()
        X_scaled = scaler_c.fit_transform(ml_clust[clust_feats])
        
        # Standard K-Means fits
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        
        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        ml_clust['cluster'] = km.fit_predict(X_scaled)
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        ml_clust['pc1'] = X_pca[:, 0]
        ml_clust['pc2'] = X_pca[:, 1]
        
        # Archetype labeling based on centroids
        centroids = ml_clust.groupby('cluster')[clust_feats].mean()
        
        archetypes = {}
        for c in range(4):
            row = centroids.loc[c]
            if row['sr_pp'] > row['sr_dth'] + 15:
                archetypes[c] = '⚡ Powerplay Striker'
            elif row['sr_dth'] > row['sr_pp'] + 15:
                archetypes[c] = '🔥 Death Finisher'
            elif row['sr_pp'] > 135 and row['sr_dth'] > 135:
                archetypes[c] = '🌟 All-Phase Dominator'
            else:
                archetypes[c] = '⚖️ Anchor / Balanced Contributor'
                
        ml_clust['archetype'] = ml_clust['cluster'].map(archetypes)
        
        # Plot K-Means Interactive PCA 2D Map
        fig_c = px.scatter(
            ml_clust,
            x='pc1',
            y='pc2',
            color='archetype',
            hover_name='batter',
            hover_data=['sr_pp', 'sr_mid', 'sr_dth', 'career_sr'],
            title="K-Means Clusters projected on 2D PCA Space",
            color_discrete_sequence=[C['green'], C['amber'], C['purple'], C['blue']]
        )
        fig_c.update_layout(paper_bgcolor='#0D0F14', plot_bgcolor='#0D0F14', font=dict(color='#F0F2F8'))
        st.plotly_chart(fig_c, use_container_width=True)

    # Sub-tab 3: Live Win probability
    with ml_sub_tab3:
        st.markdown("#### Estimate Win Probability for Team 1")
        
        # Extract unique teams
        all_teams = sorted(balls_df['batting_team'].dropna().unique())
        
        w_col1, w_col2, w_col3 = st.columns(3)
        with w_col1:
            team1 = st.selectbox("Select Team 1", all_teams, index=0)
        with w_col2:
            team2 = st.selectbox("Select Team 2", all_teams, index=1 if len(all_teams) > 1 else 0)
        with w_col3:
            toss_dec = st.selectbox("Toss Decision", ["Bat first", "Field first"])
            
        toss_winner_select = st.selectbox("Toss Winner", [team1, team2])
        
        # Calculate mock proxy strength values
        legal_b = active_balls[active_balls['is_legal']].copy()
        team_rr = (legal_b.groupby(['match_id', 'batting_team'])
                   .agg(runs=('runs_total', 'sum'), balls=('is_legal', 'sum'))
                   .reset_index())
        team_rr['rr'] = team_rr['runs'] / team_rr['balls'] * 6
        avg_rr = team_rr.groupby('batting_team')['rr'].mean().to_dict()
        
        t1_rr = avg_rr.get(team1, 7.5)
        t2_rr = avg_rr.get(team2, 7.5)
        rr_diff = t1_rr - t2_rr
        
        toss_bat = 1 if toss_dec == "Bat first" else 0
        toss_is_t1 = 1 if toss_winner_select == team1 else 0
        
        # Load and run predictor
        features = pd.DataFrame([{
            'toss_bat': toss_bat,
            'toss_is_t1': toss_is_t1,
            'rr_diff': rr_diff
        }])
        
        win_prob = model_win.predict_proba(features)[0][1] # Probability of Team 1 winning
        
        st.metric(f"{team1} Win Probability", f"{win_prob*100:.1f}%")
        st.progress(win_prob)
