"""
ETL - Scout Mundial 2026 (Versión Alta Precisión)
=========================
Integra fuentes dinámicas con normalización de entidades para evitar
pérdida de datos por string matching. Imputación lógica de variables.
"""

import pandas as pd
import numpy as np
import os
import kagglehub

# ---------------------------------------------------------------------------
# 0. NORMALIZACIÓN DE ENTIDADES (DATA CLEANSING)
# ---------------------------------------------------------------------------
def unificar_nombres(df, columna):
    """Estandariza los nombres de los países entre distintos datasets de Kaggle."""
    mapeo = {
        "USA": "United States",
        "United States of America": "United States",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Bosnia": "Bosnia and Herzegovina",
        "South Korea": "Korea Republic",
        "North Korea": "Korea DPR",
        "Czech Republic": "Czechia",
        "Turkey": "Türkiye",
        "Cape Verde": "Cabo Verde",
        "DR Congo": "Congo DR",
        "Democratic Republic of the Congo": "Congo DR",
        "Iran": "IR Iran",
        "Ivory Coast": "Côte d'Ivoire"
    }
    df[columna] = df[columna].replace(mapeo)
    return df

# ---------------------------------------------------------------------------
# 1. CARGA DE FUENTES (Automatizada)
# ---------------------------------------------------------------------------

def cargar_resultados_historicos():
    print("Descargando: international-football-results...")
    base_path = kagglehub.dataset_download("martj42/international-football-results-from-1872-to-2017")
    df = pd.read_csv(os.path.join(base_path, "results.csv"), parse_dates=["date"])
    df = df.rename(columns={
        "home_team": "equipo_local", "away_team": "equipo_visitante",
        "home_score": "goles_local", "away_score": "goles_visitante"
    })
    df = unificar_nombres(df, "equipo_local")
    df = unificar_nombres(df, "equipo_visitante")
    return df

def cargar_elo():
    print("Descargando: historical-elo-ratings...")
    base_path = kagglehub.dataset_download("afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings")
    df = pd.read_csv(os.path.join(base_path, "elo_ratings_wc2026.csv"))
    df = df.rename(columns={"country": "equipo", "rating": "elo", "year": "anio"})
    df = unificar_nombres(df, "equipo")
    return df

def cargar_team_dataset():
    print("Descargando: fifa-world-cup-team-dataset...")
    base_path = kagglehub.dataset_download("harrachimustapha/fifa-world-cup-team-dataset")
    train = pd.read_csv(os.path.join(base_path, "train.csv"))
    train = unificar_nombres(train, "team")
    return train

def cargar_grupos_2026():
    print("Descargando: fifa-world-cup-2026-match-data...")
    base_path = kagglehub.dataset_download("areezvisram12/fifa-world-cup-2026-match-data-unofficial")
    teams = pd.read_csv(os.path.join(base_path, "teams.csv"))
    schedule = pd.read_csv(os.path.join(base_path, "matches.csv"))
    return teams, schedule


def filtrar_partidos_con_elo(resultados, elo):
    """Conserva partidos donde ambos equipos tienen un Elo real para ese anio."""
    claves_elo = set(zip(elo["equipo"], elo["anio"]))
    anios = resultados["date"].dt.year

    tiene_elo_local = [
        (equipo, anio) in claves_elo
        for equipo, anio in zip(resultados["equipo_local"], anios)
    ]
    tiene_elo_visitante = [
        (equipo, anio) in claves_elo
        for equipo, anio in zip(resultados["equipo_visitante"], anios)
    ]

    mascara = np.asarray(tiene_elo_local) & np.asarray(tiene_elo_visitante)
    filtrados = resultados.loc[mascara].copy()

    print(f"Partidos con Elo real para ambos equipos: {len(filtrados)}")
    print(f"Partidos descartados por falta de Elo: {len(resultados) - len(filtrados)}")
    return filtrados

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING - VARIABLES COMPARATIVAS POR PARTIDO
# ---------------------------------------------------------------------------

def calcular_forma_reciente(resultados, equipo, fecha_referencia, ventana=10):
    mask_local = (resultados["equipo_local"] == equipo) & (resultados["date"] < fecha_referencia)
    mask_visit = (resultados["equipo_visitante"] == equipo) & (resultados["date"] < fecha_referencia)

    partidos = pd.concat([
        resultados[mask_local].assign(
            gf=resultados.loc[mask_local, "goles_local"],
            gc=resultados.loc[mask_local, "goles_visitante"],
            resultado=np.where(resultados.loc[mask_local, "goles_local"] > resultados.loc[mask_local, "goles_visitante"], "G",
                      np.where(resultados.loc[mask_local, "goles_local"] == resultados.loc[mask_local, "goles_visitante"], "E", "P"))
        ),
        resultados[mask_visit].assign(
            gf=resultados.loc[mask_visit, "goles_visitante"],
            gc=resultados.loc[mask_visit, "goles_local"],
            resultado=np.where(resultados.loc[mask_visit, "goles_visitante"] > resultados.loc[mask_visit, "goles_local"], "G",
                      np.where(resultados.loc[mask_visit, "goles_visitante"] == resultados.loc[mask_visit, "goles_local"], "E", "P"))
        ),
    ]).sort_values("date").tail(ventana)

    # Imputación Lógica: Si no hay historial, se asume un rendimiento neutral en lugar de 0 absoluto
    if partidos.empty:
        return {"pct_victorias": 0.33, "goles_favor_prom": 1.0, "goles_contra_prom": 1.0}

    return {
        "pct_victorias": (partidos["resultado"] == "G").mean(),
        "goles_favor_prom": partidos["gf"].mean(),
        "goles_contra_prom": partidos["gc"].mean(),
    }

def construir_dataset_partidos(
    resultados, elo, team_hist, historial_resultados=None
):
    filas = []
    historial_resultados = (
        resultados if historial_resultados is None else historial_resultados
    )
    
    # Pre-indexar ELO para búsquedas ultra rápidas y precisas
    elo_dict = elo.groupby(['equipo', 'anio'])['elo'].last().to_dict()
    
    for _, row in resultados.iterrows():
        equipo_a, equipo_b = row["equipo_local"], row["equipo_visitante"]
        fecha = row["date"]
        anio_partido = fecha.year

        # Busqueda estricta: ambos valores existen gracias al filtro previo.
        elo_a = elo_dict[(equipo_a, anio_partido)]
        elo_b = elo_dict[(equipo_b, anio_partido)]

        forma_a = calcular_forma_reciente(historial_resultados, equipo_a, fecha)
        forma_b = calcular_forma_reciente(historial_resultados, equipo_b, fecha)

        hist_a = team_hist[team_hist["team"] == equipo_a]
        hist_b = team_hist[team_hist["team"] == equipo_b]
        
        part_a = hist_a["world_cup_participations_before"].iloc[-1] if not hist_a.empty else 0
        part_b = hist_b["world_cup_participations_before"].iloc[-1] if not hist_b.empty else 0
        best_a = hist_a["groups_passed_before"].iloc[-1] if not hist_a.empty else 0
        best_b = hist_b["groups_passed_before"].iloc[-1] if not hist_b.empty else 0

        if row["goles_local"] > row["goles_visitante"]: etiqueta = "Gana_A"
        elif row["goles_local"] == row["goles_visitante"]: etiqueta = "Empate"
        else: etiqueta = "Gana_B"

        filas.append({
            "fecha": fecha,
            "equipo_A": equipo_a,
            "equipo_B": equipo_b,
            "diff_elo": elo_a - elo_b,
            "diff_goles_favor_prom": forma_a["goles_favor_prom"] - forma_b["goles_favor_prom"],
            "diff_goles_contra_prom": forma_a["goles_contra_prom"] - forma_b["goles_contra_prom"],
            "diff_pct_victorias": forma_a["pct_victorias"] - forma_b["pct_victorias"],
            "diff_participaciones_mundial": part_a - part_b,
            "diff_mejor_instancia": best_a - best_b,
            "etiqueta": etiqueta,
        })

    return pd.DataFrame(filas)

# ---------------------------------------------------------------------------
# 3. PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------
def main():
    print("Iniciando tubería de datos ETL...")
    resultados = cargar_resultados_historicos()
    elo = cargar_elo()
    team_train = cargar_team_dataset()
    teams_2026, schedule_2026 = cargar_grupos_2026()

    resultados_recientes = resultados[resultados["date"] >= "2010-01-01"]
    print(f"Partidos desde la era moderna (2010) a procesar: {len(resultados_recientes)}")

    resultados_con_elo = filtrar_partidos_con_elo(resultados_recientes, elo)

    print("\nConstruyendo dataset integrado con Features (Esto puede tardar unos minutos)...")
    dataset = construir_dataset_partidos(
        resultados_con_elo,
        elo,
        team_train,
        historial_resultados=resultados_recientes,
    )

    dataset.to_csv("dataset_integrado.csv", index=False)
    teams_2026.to_csv("teams_2026.csv", index=False)
    schedule_2026.to_csv("schedule_2026.csv", index=False)
    print("✅ ETL Finalizado con precisión mejorada. Archivos listos.")

if __name__ == "__main__":
    main()
