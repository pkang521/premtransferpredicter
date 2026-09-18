# STEP 3: A clean and simple dashboard to look at the results.
#
# Run:  streamlit run step3_app.py

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="EPL Value Predictor", layout="wide")
st.title("⚽ EPL Player Value Predictor")
st.write("A linear regression that predicts a Premier League player's Transfermarkt "
         "market value from their stats.")

# From step 2
preds = pd.read_csv("predictions.csv")
preds["club"] = preds["club"].fillna("Unknown club")
saved = joblib.load("model.joblib")
model, features = saved["model"], saved["features"]

# ---------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------
st.header("How accurate is the model?")
r2 = 1 - ((np.log(preds["market_value"]) - np.log(preds["predicted_value"])) ** 2).sum() / \
         ((np.log(preds["market_value"]) - np.log(preds["market_value"]).mean()) ** 2).sum()

col1, col2, col3 = st.columns(3)
col1.metric("R² (log value)", f"{r2:.2f}")
col2.metric("Median error", f"{preds['error_pct'].median():.0%}")
col3.metric("Players tested", len(preds))

fig = px.scatter(
    preds, x="market_value", y="predicted_value", hover_name="name",
    hover_data={"club": True, "age": ":.0f"}, log_x=True, log_y=True,
    labels={"market_value": "Actual value (€)", "predicted_value": "Predicted value (€)"},
)
lo, hi = preds["market_value"].min(), preds["market_value"].max()
fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash="dot", color="gray"))
st.plotly_chart(fig, use_container_width=True)
st.caption("Each dot is a player. On the dotted line = perfect prediction. "
           "Above the line = model thinks he's worth more than the market says.")

# ---------------------------------------------------------------
# Looking up players
# ---------------------------------------------------------------
st.header("Look up a player")
preds["label"] = preds["name"] + " (" + preds["club"] + ")"
choice = st.selectbox("Player", sorted(preds["label"]))
p = preds[preds["label"] == choice].iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric("Actual market value", f"€{p['market_value'] / 1e6:.1f}m")
col2.metric("Model's prediction", f"€{p['predicted_value'] / 1e6:.1f}m")
col3.metric("Difference", f"{p['predicted_value'] / p['market_value'] - 1:+.0%}")
st.write(f"**{p['club']}** · age {p['age']:.0f} · {p['minutes']:.0f} minutes · "
         f"{p['goals']:.0f} goals · {p['assists']:.0f} assists")

# ---------------------------------------------------------------
# Bargains and Ripoffs
# ---------------------------------------------------------------
st.header("Where the model and the market disagree")
preds["model_vs_market"] = preds["predicted_value"] / preds["market_value"] - 1

# Make a tidy copy just for display
show = preds.copy()
show["age"] = show["age"].round(0).astype(int)
show["market_value"] = (show["market_value"] / 1e6).round(1)       # in €m
show["predicted_value"] = (show["predicted_value"] / 1e6).round(1)
show["model_vs_market"] = (show["model_vs_market"] * 100).round(0)  # in %
show = show.rename(columns={"market_value": "market_value_€m",
                            "predicted_value": "predicted_€m",
                            "model_vs_market": "difference_%"})
cols = ["name", "club", "age", "goals", "market_value_€m", "predicted_€m", "difference_%"]

left, right = st.columns(2)
left.subheader("Model says: worth more")
left.dataframe(show.nlargest(10, "difference_%")[cols], hide_index=True)
right.subheader("Model says: worth less")
right.dataframe(show.nsmallest(10, "difference_%")[cols], hide_index=True)

# ---------------------------------------------------------------
# Making a new player
# ---------------------------------------------------------------
st.header("Predict a made-up player")
col1, col2 = st.columns(2)
age = col1.slider("Age", 17, 36, 24)
minutes = col1.slider("Minutes played in the season", 450, 3420, 2500)
position = col1.selectbox("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"])
goals_90 = col2.slider("Goals per 90 minutes", 0.0, 1.0, 0.2)
assists_90 = col2.slider("Assists per 90 minutes", 0.0, 0.6, 0.1)
club_ppg = col2.slider("Club points per game (0.8 = relegation, 2.3 = title)", 0.5, 2.6, 1.4)

player = {"age": age, "age_squared": age ** 2, "minutes": minutes,
          "goals_per_90": goals_90, "assists_per_90": assists_90,
          "club_points_per_game": club_ppg}
for f in features:                                  
    if f.startswith("position_"):
        player[f] = 1 if f == f"position_{position}" else 0

prediction = np.exp(model.predict(pd.DataFrame([player])[features])[0])
st.metric("Predicted market value", f"€{prediction / 1e6:.1f}m")
