import pandas as pd
import os
import kagglehub
from src.utils import normalizar_equipo

def load_datasets():
    try:
        equipos = pd.read_csv("data/teams_2026.csv")
        fixture = pd.read_csv("data/schedule_2026.csv")
    except:
        path = kagglehub.dataset_download("areezvisram12/fifa-world-cup-2026-match-data-unofficial")
        equipos = pd.read_csv(os.path.join(path, "teams.csv"))
        fixture = pd.read_csv(os.path.join(path, "matches.csv"))

    equipos["equipo"] = equipos["team"].apply(normalizar_equipo)
    return equipos, fixture