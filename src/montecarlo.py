"""Monte Carlo helpers for knockout-stage simulations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from .predictor import predict_knockout_match


def simulate_knockout_round(
	fixtures: Iterable[Dict[str, str]],
	team_stats: pd.DataFrame,
	model: Any,
	feature_cols: pd.Index,
	n_iter: int = 1000,
) -> pd.DataFrame:
	"""Return a classification-frequency table for a knockout round."""

	fixtures_list: List[Dict[str, str]] = list(fixtures)
	win_counts = {}

	for fixture in fixtures_list:
		win_counts.setdefault(fixture["equipo_A"], 0)
		win_counts.setdefault(fixture["equipo_B"], 0)

	deterministic_results = [
		predict_knockout_match(fixture["equipo_A"], fixture["equipo_B"], team_stats, model, feature_cols)
		for fixture in fixtures_list
	]

	for _ in range(n_iter):
		for fixture, result in zip(fixtures_list, deterministic_results):
			if result["estado"] == "Confirmado":
				win_counts[result["ganador"]] += 1
				continue

			probabilities = np.array([
				result["prob_A"] / 100,
				result["prob_empate"] / 100,
				result["prob_B"] / 100,
			])
			outcome = np.random.choice(["Gana_A", "Empate", "Gana_B"], p=probabilities)

			if outcome == "Gana_A":
				winner = fixture["equipo_A"]
			elif outcome == "Gana_B":
				winner = fixture["equipo_B"]
			else:
				winner = fixture["equipo_A"] if result["prob_A"] >= result["prob_B"] else fixture["equipo_B"]

			win_counts[winner] += 1

	rows = [
		{
			"Equipo": team,
			"Veces clasifica": count,
			"Probabilidad de clasificar %": round((count / n_iter) * 100, 1),
		}
		for team, count in win_counts.items()
		if count > 0
	]

	return pd.DataFrame(rows).sort_values("Probabilidad de clasificar %", ascending=False).reset_index(drop=True)
