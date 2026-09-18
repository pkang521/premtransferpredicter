# EPL Player Value Predictor

Predicts a Premier League player's Transfermarkt market value from his season stats using linear regression.

## What it does

1. **Prepare:** one row per player per Premier League season (at least 450 minutes played), with minutes, goals and assists per 90, age, position, club strength (points per game) and end-of-season market value.
2. **Train:** linear regression on log(market value). Trained on older seasons and tested on the most recent season.
3. **Dashboard:** accuracy, predicted vs actual chart, player lookup, the players the model rates above or below the market, and a what-if predictor.

## How to run

**1. Set up Python and install the libraries:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Get the data.** Download the Kaggle dataset [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores) and put `players.csv`, `player_valuations.csv`, `appearances.csv`, `games.csv` and `clubs.csv` in a `data/` folder.

**3. Run it:**
```bash
python step1_prepare_data.py    # raw CSVs -> player_seasons.csv
python step2_train_model.py     # trains the model, prints accuracy
streamlit run step3_app.py      # opens the dashboard
```
Or in one command: `./run.sh retrain` the first time, then `./run.sh` to open the dashboard.

## Limitations

- Market value is an estimate, not an actual transfer fee.
- Factors such as contract terminations, reputation of a player and injuries are not included in the model.
- Not really relevant for goalkeepers and defenders, as goals and assist don't say much for them

## Data

[transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) by David Cariboo (CC0 license), sourced from Transfermarkt.com. The data runs to July 2026.
