"""Bracket helpers for the World Cup knockout stage.

This module holds the reusable simulation logic so the Streamlit app can stay
focused on data loading and presentation.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd


ETAPAS_BRACKET = ["16avos", "Octavos", "Cuartos", "Semifinal", "Final"]
ETAPA_SIGUIENTE = {
	"16avos": "Octavos",
	"Octavos": "Cuartos",
	"Cuartos": "Semifinal",
	"Semifinal": "Final",
	"Final": "Campeón",
}


def construir_partidos_16avos(
	knockout_lookup: Mapping[str, Mapping[str, Any]],
	round_of_32_bracket_order: Sequence[str],
) -> list[tuple[str, str]]:
	"""Return the fixed first-round bracket in official order."""
	partidos: list[tuple[str, str]] = []
	for codigo in round_of_32_bracket_order:
		info = knockout_lookup.get(codigo)
		if info:
			partidos.append((str(info["equipo_a"]), str(info["equipo_b"])))
	return partidos


def obtener_probabilidades_partido(
	equipo_A: str,
	equipo_B: str,
	cache_probs: MutableMapping[frozenset[str], Tuple[float, float, float, str]],
	model: Any,
	calcular_features_partido: Callable[[str, str], pd.DataFrame],
) -> tuple[float, float, float]:
	"""Return the model probabilities for a matchup, with symmetric caching."""
	llave = frozenset((equipo_A, equipo_B))
	if llave in cache_probs:
		prob_A, prob_E, prob_B, equipo_de_referencia = cache_probs[llave]
		if equipo_de_referencia == equipo_A:
			return prob_A, prob_E, prob_B
		return prob_B, prob_E, prob_A

	X_pred = calcular_features_partido(equipo_A, equipo_B)
	probs = model.predict_proba(X_pred)[0]
	clases = list(model.classes_)
	idx_A, idx_E, idx_B = clases.index("Gana_A"), clases.index("Empate"), clases.index("Gana_B")
	prob_A, prob_E, prob_B = probs[idx_A], probs[idx_E], probs[idx_B]
	cache_probs[llave] = (prob_A, prob_E, prob_B, equipo_A)
	return prob_A, prob_E, prob_B


def resolver_partido_bracket(
	equipo_A: str,
	equipo_B: str,
	cache_probs: MutableMapping[frozenset[str], Tuple[float, float, float, str]],
	model: Any,
	calcular_features_partido: Callable[[str, str], pd.DataFrame],
	confirmed_results: Mapping[frozenset[str], Mapping[str, Any]],
	aleatorio: bool,
) -> tuple[str, str]:
	"""Resolve a knockout match respecting confirmed results when available."""
	llave = frozenset((equipo_A, equipo_B))
	if llave in confirmed_results:
		confirmado = confirmed_results[llave]
		return str(confirmado["ganador"]), "Confirmado"

	prob_A, prob_E, prob_B = obtener_probabilidades_partido(
		equipo_A,
		equipo_B,
		cache_probs,
		model,
		calcular_features_partido,
	)

	if aleatorio:
		outcome = np.random.choice(["Gana_A", "Empate", "Gana_B"], p=[prob_A, prob_E, prob_B])
		if outcome == "Gana_A":
			return equipo_A, "Simulado"
		if outcome == "Gana_B":
			return equipo_B, "Simulado"
		ganador_penales = equipo_A if np.random.random() < 0.5 else equipo_B
		return ganador_penales, "Simulado (penales)"

	score_A = prob_A + prob_E / 2
	score_B = prob_B + prob_E / 2
	ganador = equipo_A if score_A >= score_B else equipo_B
	return ganador, "Pronosticado"


def jugar_ronda_bracket(
	partidos: Iterable[tuple[str, str]],
	cache_probs: MutableMapping[frozenset[str], Tuple[float, float, float, str]],
	model: Any,
	calcular_features_partido: Callable[[str, str], pd.DataFrame],
	confirmed_results: Mapping[frozenset[str], Mapping[str, Any]],
	aleatorio: bool,
) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
	"""Play a full round and pair consecutive winners for the next round."""
	detalle: list[dict[str, str]] = []
	ganadores: list[str] = []
	for equipo_A, equipo_B in partidos:
		ganador, estado = resolver_partido_bracket(
			equipo_A,
			equipo_B,
			cache_probs,
			model,
			calcular_features_partido,
			confirmed_results,
			aleatorio,
		)
		detalle.append({
			"equipo_A": equipo_A,
			"equipo_B": equipo_B,
			"ganador": ganador,
			"estado": estado,
		})
		ganadores.append(ganador)

	siguiente_ronda = [(ganadores[i], ganadores[i + 1]) for i in range(0, len(ganadores) - 1, 2)]
	return detalle, siguiente_ronda


def simular_bracket_unico(
	construir_partidos_fn: Callable[[], list[tuple[str, str]]],
	model: Any,
	calcular_features_partido: Callable[[str, str], pd.DataFrame],
	confirmed_results: Mapping[frozenset[str], Mapping[str, Any]],
	aleatorio: bool = False,
) -> tuple[dict[str, list[dict[str, str]]], str]:
	"""Simulate the full knockout path once and return the champion."""
	cache_probs: dict[frozenset[str], tuple[float, float, float, str]] = {}
	partidos = construir_partidos_fn()
	detalle_por_etapa: dict[str, list[dict[str, str]]] = {}

	for etapa in ETAPAS_BRACKET:
		detalle, siguiente = jugar_ronda_bracket(
			partidos,
			cache_probs,
			model,
			calcular_features_partido,
			confirmed_results,
			aleatorio,
		)
		detalle_por_etapa[etapa] = detalle
		partidos = siguiente

	campeon = detalle_por_etapa["Final"][0]["ganador"]
	return detalle_por_etapa, campeon


def simular_bracket_montecarlo(
	n_iter: int,
	construir_partidos_fn: Callable[[], list[tuple[str, str]]],
	model: Any,
	calcular_features_partido: Callable[[str, str], pd.DataFrame],
	confirmed_results: Mapping[frozenset[str], Mapping[str, Any]],
) -> pd.DataFrame:
	"""Estimate advancement percentages for the whole knockout bracket."""
	equipos_iniciales = set()
	for equipo_A, equipo_B in construir_partidos_fn():
		equipos_iniciales.add(equipo_A)
		equipos_iniciales.add(equipo_B)

	conteo = {
		equipo: {"Octavos": 0, "Cuartos": 0, "Semifinal": 0, "Final": 0, "Campeón": 0}
		for equipo in equipos_iniciales
	}

	cache_probs: dict[frozenset[str], tuple[float, float, float, str]] = {}
	for _ in range(n_iter):
		partidos = construir_partidos_fn()
		for etapa in ETAPAS_BRACKET:
			detalle, siguiente = jugar_ronda_bracket(
				partidos,
				cache_probs,
				model,
				calcular_features_partido,
				confirmed_results,
				aleatorio=True,
			)
			etapa_alcanzada = ETAPA_SIGUIENTE[etapa]
			for fila in detalle:
				conteo[fila["ganador"]][etapa_alcanzada] += 1
			partidos = siguiente

	filas = []
	for equipo, stats in conteo.items():
		filas.append({
			"Equipo": equipo,
			"Octavos %": round(stats["Octavos"] / n_iter * 100, 1),
			"Cuartos %": round(stats["Cuartos"] / n_iter * 100, 1),
			"Semifinal %": round(stats["Semifinal"] / n_iter * 100, 1),
			"Final %": round(stats["Final"] / n_iter * 100, 1),
			"Campeón %": round(stats["Campeón"] / n_iter * 100, 1),
		})

	return pd.DataFrame(filas).sort_values(
		by=["Campeón %", "Final %", "Semifinal %"], ascending=False
	).reset_index(drop=True)


def simular_mundial_2026_montecarlo(
	n_iter: int,
	construir_partidos_fn: Callable[[], list[tuple[str, str]]],
	model: Any,
	calcular_features_partido: Callable[[str, str], pd.DataFrame],
	confirmed_results: Mapping[frozenset[str], Mapping[str, Any]],
) -> pd.DataFrame:
	"""Alias explícito para el tramo eliminatorio completo."""
	return simular_bracket_montecarlo(
		n_iter,
		construir_partidos_fn,
		model,
		calcular_features_partido,
		confirmed_results,
	)
