import pandas as pd
from src.utils import normalizar_equipo

def build_features(team_a, team_b, df):
    a = df[df["equipo"] == team_a].iloc[0]
    b = df[df["equipo"] == team_b].iloc[0]

    return pd.DataFrame([{
        "diff_elo": a["elo"] - b["elo"],
        "diff_rank": b["rank"] - a["rank"],
        "diff_attack": a["attack"] - b["attack"],
        "diff_defense": a["defense"] - b["defense"],
    }])