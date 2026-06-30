"""Utilities for predicting World Cup knockout matches.

This module keeps the prediction logic reusable outside the Streamlit app.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "rf_model.pkl")
DEFAULT_FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")


ALIAS_EQUIPOS = {
	"Canada": "Canada",
	"Canadá": "Canada",
	"South Africa": "South Africa",
	"Sudafrica": "South Africa",
	"Sudáfrica": "South Africa",
	"Brazil": "Brazil",
	"Brasil": "Brazil",
	"Japan": "Japan",
	"Japon": "Japan",
	"Japón": "Japan",
	"Curaçao": "Curaçao",
	"Curacao": "Curaçao",
	"DR Congo": "DR Congo",
	"Bosnia and Herzegovina": "Bosnia and Herzegovina",
}


CONFIRMED_RESULTS = {
	frozenset(("Canada", "South Africa")): {
		"equipo_A": "Canada",
		"equipo_B": "South Africa",
		"goles_A": 1,
		"goles_B": 0,
		"ganador": "Canada",
	},
	frozenset(("Brazil", "Japan")): {
		"equipo_A": "Brazil",
		"equipo_B": "Japan",
		"goles_A": 2,
		"goles_B": 1,
		"ganador": "Brazil",
	},
}


def normalize_team_name(name: Any) -> str:
	value = ALIAS_EQUIPOS.get(str(name).strip(), str(name).strip())
	value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
	value = value.lower().replace("&", " and ")
	value = re.sub(r"[^a-z0-9]+", " ", value)
	return re.sub(r"\s+", " ", value).strip()


def load_model(model_path: Optional[str] = None, features_path: Optional[str] = None) -> Tuple[Any, Any]:
	model_path = model_path or DEFAULT_MODEL_PATH
	features_path = features_path or DEFAULT_FEATURES_PATH
	model = joblib.load(model_path)
	features = joblib.load(features_path)
	return model, features


def load_team_stats(team_df: pd.DataFrame) -> pd.DataFrame:
	data = team_df.copy()
	data.columns = [c.lower().strip() for c in data.columns]
	if "equipo" not in data.columns:
		team_col = next((c for c in data.columns if "team" in c or "country" in c or "name" in c), data.columns[0])
		data = data.rename(columns={team_col: "equipo"})
	data["equipo_normalizado"] = data["equipo"].apply(normalize_team_name)
	return data


def get_team_stats(team: str, team_stats: pd.DataFrame) -> pd.Series:
	normalized = normalize_team_name(team)
	if "equipo_normalizado" not in team_stats.columns:
		team_stats = load_team_stats(team_stats)

	match = team_stats[team_stats["equipo_normalizado"] == normalized]
	if not match.empty:
		return match.iloc[0]

	numeric = team_stats.select_dtypes(include=[np.number])
	fallback = numeric.median(numeric_only=True) if not numeric.empty else pd.Series(dtype=float)
	fallback["equipo"] = team
	fallback["equipo_normalizado"] = normalized
	if "elo_actual" not in fallback or pd.isna(fallback.get("elo_actual")):
		fallback["elo_actual"] = 1500
	if "rank_fifa" not in fallback or pd.isna(fallback.get("rank_fifa")):
		fallback["rank_fifa"] = 50
	if "goles_favor_prom" not in fallback or pd.isna(fallback.get("goles_favor_prom")):
		fallback["goles_favor_prom"] = 1.0
	if "goles_contra_prom" not in fallback or pd.isna(fallback.get("goles_contra_prom")):
		fallback["goles_contra_prom"] = 1.0
	if "pct_victorias" not in fallback or pd.isna(fallback.get("pct_victorias")):
		fallback["pct_victorias"] = 0.33
	if "participaciones" not in fallback or pd.isna(fallback.get("participaciones")):
		fallback["participaciones"] = 0
	if "grupos_superados" not in fallback or pd.isna(fallback.get("grupos_superados")):
		fallback["grupos_superados"] = 0
	return fallback


def build_match_features(team_a: str, team_b: str, team_stats: pd.DataFrame, feature_cols: pd.Index) -> pd.DataFrame:
	stats_a = get_team_stats(team_a, team_stats)
	stats_b = get_team_stats(team_b, team_stats)

	row = {
		"diff_elo": stats_a["elo_actual"] - stats_b["elo_actual"],
		"diff_ranking_fifa": stats_b["rank_fifa"] - stats_a["rank_fifa"],
		"diff_goles_favor_prom": stats_a["goles_favor_prom"] - stats_b["goles_favor_prom"],
		"diff_goles_contra_prom": stats_a["goles_contra_prom"] - stats_b["goles_contra_prom"],
		"diff_pct_victorias": stats_a["pct_victorias"] - stats_b["pct_victorias"],
		"diff_participaciones_mundial": stats_a["participaciones"] - stats_b["participaciones"],
		"diff_mejor_instancia": stats_a["grupos_superados"] - stats_b["grupos_superados"],
	}
	return pd.DataFrame([row])[feature_cols]


def predict_knockout_match(
	team_a: str,
	team_b: str,
	team_stats: pd.DataFrame,
	model: Any,
	feature_cols: pd.Index,
) -> Dict[str, Any]:
	confirmed = CONFIRMED_RESULTS.get(frozenset((team_a, team_b)))
	if confirmed:
		marcador = f"{confirmed['goles_A']}-{confirmed['goles_B']}"
		if confirmed["equipo_A"] != team_a:
			marcador = f"{confirmed['goles_B']}-{confirmed['goles_A']}"
		return {
			"partido": f"{team_a} vs {team_b}",
			"estado": "Confirmado",
			"marcador": marcador,
			"ganador": confirmed["ganador"],
			"prob_A": None,
			"prob_empate": None,
			"prob_B": None,
		}

	features = build_match_features(team_a, team_b, team_stats, feature_cols)
	probabilities = model.predict_proba(features)[0]
	classes = list(model.classes_)
	idx_a = classes.index("Gana_A")
	idx_e = classes.index("Empate")
	idx_b = classes.index("Gana_B")

	score_a = probabilities[idx_a] + (probabilities[idx_e] / 2)
	score_b = probabilities[idx_b] + (probabilities[idx_e] / 2)
	winner = team_a if score_a >= score_b else team_b

	return {
		"partido": f"{team_a} vs {team_b}",
		"estado": "Pronosticado",
		"marcador": "-",
		"ganador": winner,
		"prob_A": round(probabilities[idx_a] * 100, 1),
		"prob_empate": round(probabilities[idx_e] * 100, 1),
		"prob_B": round(probabilities[idx_b] * 100, 1),
	}
