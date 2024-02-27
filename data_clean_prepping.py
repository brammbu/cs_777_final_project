"""### Grab Data"""

"""### Now to explore data"""

import numpy as np
import pandas as pd

bets = pd.read_csv(r"nba_data\nba_betting_money_line.csv")

# correct column names
bets = bets.rename(columns={"team_id": "a_team_id",
                            "a_team_id": "team_id",
                            "price1": "a_price",
                            "price2": "h_price",})

bets.head()

games = pd.read_csv(r"nba_data\nba_games_all.csv")
games.head()


# Select a single book and drop everything else
# Remove useless columns
bets = bets.loc[bets["book_name"] == "Bovada"]
bets = bets.drop(["book_name", "book_id"], axis=1)

# Merge game stats
bets_games = bets.merge(games[["game_id", "team_id", "a_team_id", "is_home", "wl", "season_year"]], on=["game_id", "team_id", "a_team_id"])


# Convert American Odds to Decimal odds
def conv_odds(row):
    if row >= 0:
        odds = (row / 100) + 1
    else:
        odds = (100 / -row) + 1
    return odds

# Convert wl to 0 or 1
def conv_wl(row):
    if row == "W":
        return 1
    elif row == "L":
        return 0

bets_games["h_price"] = bets_games["h_price"].apply(conv_odds)
bets_games["a_price"] = bets_games["a_price"].apply(conv_odds)

bets_games["wl"] = bets_games["wl"].apply(conv_wl)

"""### Looking at player info"""

player_stats = pd.read_csv(r"nba_data\nba_players_game_stats.csv")

filtered_player_stats = player_stats.loc[(player_stats["season_year"] > 2006) & (player_stats["season_year"] <= 2016) & (player_stats["season_type"] == "Regular Season"),]


team_game_group_df = filtered_player_stats[["player_id", "player_name", "game_id", "team_id", "min"]].sort_values(by="min", ascending=False).groupby(by=["game_id", "team_id"])['player_name'].apply(lambda df: df.reset_index(drop=True)).unstack().reset_index()

team_game_lineup = team_game_group_df[['game_id', 'team_id', 0, 1, 2, 3, 4, 5]]
team_game_lineup = team_game_lineup.rename(columns={0: "player_1", 1: "player_2", 2: "player_3", 3: "player_4", 4: "player_5", 5: "player_6"})

"""### Combine player and bet/game data"""

merge_df = team_game_lineup[['game_id', 'team_id', 'player_1', 'player_2', 'player_3',
       'player_4', 'player_5', 'player_6']]

game_df = bets_games.merge(merge_df, on=["game_id", "team_id"])

game_df = game_df.merge(merge_df, left_on=["game_id", "a_team_id"], right_on=["game_id", "team_id"], suffixes=('', '_a'))

# Season number instead of year
game_df["season_num"] = (game_df["season_year"] - game_df["season_year"].min()) / (game_df["season_year"].max() - game_df["season_year"].min())

"""### Split into Train / Test plit"""

from sklearn.model_selection import train_test_split

# split the data into train and test set
# only qualifier is statifying by year to get an even distribution
train_df, test_df = train_test_split(game_df, test_size=0.2, random_state=42, shuffle=True, stratify=game_df[['season_year']])


train_df.to_csv('nba_train_data.csv')

test_df.to_csv('nba_test_data.csv')