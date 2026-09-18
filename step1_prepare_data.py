# STEP 1: Turn the raw Transfermarkt CSVs into one clean table.
#
# Each row of the final table = one player in one Premier League season, with:
#   - his stats that season (minutes, goals, assists)
#   - his age and position
#   - how good his club was that season
#   - his market value at the end of that season  <- what we want to predict
#
# Run:  python step1_prepare_data.py
# Makes: player_seasons.csv

import pandas as pd

# ---------------------------------------------------------------
# 1. Load the raw data
# ---------------------------------------------------------------
players = pd.read_csv("data/players.csv")
valuations = pd.read_csv("data/player_valuations.csv")
appearance = pd.read_csv("data/appearances.csv")
games = pd.read_csv("data/games.csv")
clubs = pd.read_csv("data/clubs.csv")

print("appearances loaded:", len(appearance), "rows")

# "GB1" is Transfermarkt's code for the Premier League.
# Keep only Premier League appearance and games.
appearance = appearance[appearance["competition_id"] == "GB1"]
games = games[games["competition_id"] == "GB1"]
games = games.dropna(subset=["home_club_goals", "away_club_goals"])   # skip unplayed games

# Appearances don't say which season they're from, so borrow it from the games table.
appearance = appearance.merge(games[["game_id", "season"]], on="game_id")

# ---------------------------------------------------------------
# 2. Add up each player's stats for each season
# ---------------------------------------------------------------
stats = appearance.groupby(["player_id", "season"]).agg(
    games_played=("game_id", "count"),
    minutes=("minutes_played", "sum"),
    goals=("goals", "sum"),
    assists=("assists", "sum"),
    club_id=("player_club_id", lambda s: s.mode().iloc[0]),   # club he played for most,
).reset_index()

# Players with very few minutes are not statistically meaningful
# 900 minutes = 10 full games.
stats = stats[stats["minutes"] >= 900]

# Per-90 stats makes comparing players fair
stats["goals_per_90"] = stats["goals"] / stats["minutes"] * 90
stats["assists_per_90"] = stats["assists"] / stats["minutes"] * 90

# ---------------------------------------------------------------
# 3. How good was each club? (points per game that season)
# ---------------------------------------------------------------
# Each game has both a home and away team
home = pd.DataFrame({
    "club_id": games["home_club_id"],
    "season": games["season"],
    "scored": games["home_club_goals"],
    "conceded": games["away_club_goals"],
})
away = pd.DataFrame({
    "club_id": games["away_club_id"],
    "season": games["season"],
    "scored": games["away_club_goals"],
    "conceded": games["home_club_goals"],
})
results = pd.concat([home, away])

# 3 points for a win, 1 for a draw, 0 for a loss
results["points"] = 0
results.loc[results["scored"] > results["conceded"], "points"] = 3
results.loc[results["scored"] == results["conceded"], "points"] = 1

club_strength = results.groupby(["club_id", "season"])["points"].mean().reset_index()
club_strength = club_strength.rename(columns={"points": "club_points_per_game"})

stats = stats.merge(club_strength, on=["club_id", "season"])

# ---------------------------------------------------------------
# 4. The market value at the end of each season 
# ---------------------------------------------------------------
# Season 2023 means the 2023/24 season, which ends in summer 2024.
# Take each player's LAST valuation between January and July 2024.
valuations["date"] = pd.to_datetime(valuations["date"])
valuations["year"] = valuations["date"].dt.year
valuations["month"] = valuations["date"].dt.month
valuations = valuations[valuations["month"] <= 7].copy()   # Jan–Jul only
valuations["season"] = valuations["year"] - 1              # July 2024 -> season 2023

valuations = valuations.sort_values("date")
end_values = valuations.groupby(["player_id", "season"]).last().reset_index()
end_values = end_values[["player_id", "season", "market_value_in_eur"]]
end_values = end_values.rename(columns={"market_value_in_eur": "market_value"})

stats = stats.merge(end_values, on=["player_id", "season"])

# ---------------------------------------------------------------
# 5. Add name, position, age and club name
# ---------------------------------------------------------------
stats = stats.merge(players[["player_id", "name", "position", "date_of_birth"]], on="player_id")
stats = stats.merge(clubs[["club_id", "name"]].rename(columns={"name": "club"}), on="club_id", how="left")

birth = pd.to_datetime(stats["date_of_birth"])
season_end = pd.to_datetime((stats["season"] + 1).astype(str) + "-06-30")
stats["age"] = (season_end - birth).dt.days / 365.25

# Drop rows with missing info
stats = stats.dropna(subset=["age", "position", "market_value"])
stats = stats[stats["position"] != "Missing"]

# ---------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------
stats.to_csv("player_seasons.csv", index=False)
print("Saved player_seasons.csv with", len(stats), "player-seasons")
print("Seasons:", sorted(stats["season"].unique().tolist()))
print(stats[["name", "season", "minutes", "goals", "age", "market_value"]].head())
