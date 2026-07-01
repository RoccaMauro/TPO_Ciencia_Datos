import streamlit as st
import pandas as pd
import numpy as np
import joblib
import itertools
import kagglehub
import os
import re
import unicodedata
from simulator.eliminatorias import (
    ETAPA_SIGUIENTE,
    ETAPAS_BRACKET,
    construir_partidos_16avos as construir_partidos_16avos_base,
    jugar_ronda_bracket,
    simular_bracket_montecarlo as simular_bracket_montecarlo_base,
    simular_bracket_unico as simular_bracket_unico_base,
    simular_mundial_2026_montecarlo as simular_mundial_2026_montecarlo_base,
)

st.set_page_config(page_title="Scout Mundial 2026", page_icon="🏆", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
GROUPS_CSV_PATH = os.path.join(DATA_DIR, "grupos_2026.csv")
KNOCKOUT_CSV_PATH = os.path.join(DATA_DIR, "llaves_16avos.csv")
st.sidebar.divider()

# --- Carga de Modelos y Datos ---
@st.cache_resource
def load_model():
    rf = joblib.load("rf_model.pkl")
    features = joblib.load("features.pkl")
    return rf, features


@st.cache_data
def load_data():
    try:
        equipos = pd.read_csv("teams_2026.csv")
        fixture = pd.read_csv("schedule_2026.csv")
    except FileNotFoundError:
        base_path = kagglehub.dataset_download("areezvisram12/fifa-world-cup-2026-match-data-unofficial")
        equipos = pd.read_csv(os.path.join(base_path, "teams.csv"))
        fixture = pd.read_csv(os.path.join(base_path, "matches.csv"))

    def limpiar_placeholder(nombre):
        nombre_str = str(nombre).strip()
        nombre_lower = nombre_str.lower()
        nombre_norm = nombre_lower.replace('-', ' ').replace('_', ' ')

        mapeo_repechajes = {
            'uefa playoff a': 'Bosnia and Herzegovina',
            'uefa playoff b': 'Sweden',
            'uefa playoff c': 'Turkey',
            'uefa playoff d': 'Czech Republic',
            'intercontinental playoff 1': 'DR Congo',
            'intercontinental playoff 2': 'Iraq'
        }

        for placeholder, ganador_real in mapeo_repechajes.items():
            p_norm = placeholder.replace('-', ' ').replace('_', ' ')
            if p_norm in nombre_norm:
                return ganador_real

        if 'tbc' in nombre_norm or ('play off' in nombre_norm and not any(k in nombre_norm for k in mapeo_repechajes.keys())):
            return 'Panama'

        return nombre_str

    equipos.columns = [c.lower().strip() for c in equipos.columns]
    col_equipo = next((c for c in equipos.columns if 'team' in c or 'country' in c or 'name' in c), equipos.columns[0])
    equipos = equipos.rename(columns={col_equipo: 'equipo'})
    equipos['equipo'] = equipos['equipo'].apply(limpiar_placeholder)

    col_grupo = next((c for c in equipos.columns if 'group' in c), None)
    if col_grupo:
        equipos = equipos.rename(columns={col_grupo: 'grupo'})
        equipos['grupo'] = equipos['grupo'].astype(str).str.replace('Group ', '', regex=False).str.strip()
    else:
        equipos['grupo'] = 'A'

    fixture.columns = [c.lower().strip() for c in fixture.columns]
    col_home = next((c for c in fixture.columns if 'home' in c or 'team1' in c or 'team_1' in c), 'equipo_local')
    col_away = next((c for c in fixture.columns if 'away' in c or 'team2' in c or 'team_2' in c), 'equipo_visitante')
    col_fixture_group = next((c for c in fixture.columns if 'group' in c), 'grupo')

    fixture = fixture.rename(columns={col_home: 'equipo_local', col_away: 'equipo_visitante', col_fixture_group: 'grupo'})
    if 'grupo' in fixture.columns:
        fixture['grupo'] = fixture['grupo'].astype(str).str.replace('Group ', '', regex=False).str.strip()

    fixture['equipo_local'] = fixture['equipo_local'].apply(limpiar_placeholder)
    fixture['equipo_visitante'] = fixture['equipo_visitante'].apply(limpiar_placeholder)

    try:
        team_dataset_path = kagglehub.dataset_download("harrachimustapha/fifa-world-cup-team-dataset")
        df_historico = pd.read_csv(os.path.join(team_dataset_path, "train.csv"))
        df_historico = df_historico.sort_values('version').drop_duplicates('team', keep='last')

        equipos['elo_actual'] = 1500
        equipos['rank_fifa'] = 50
        equipos['goles_favor_prom'] = 1.0
        equipos['goles_contra_prom'] = 1.0
        equipos['pct_victorias'] = 0.33
        equipos['participaciones'] = 0
        equipos['grupos_superados'] = 0

        for idx, row in equipos.iterrows():
            pais = row['equipo']
            datos_pais = df_historico[df_historico['team'] == pais]

            if not datos_pais.empty:
                stats = datos_pais.iloc[0]
                equipos.at[idx, 'rank_fifa'] = stats.get('fifa_rank_pre_tournament', 50)
                equipos.at[idx, 'goles_favor_prom'] = stats.get('goals_scored_last_4y', 4) / 4.0
                equipos.at[idx, 'goles_contra_prom'] = stats.get('goals_received_last_4y', 4) / 4.0

                wins = stats.get('wins_last_4y', 0)
                losses = stats.get('losses_last_4y', 0)
                draws = stats.get('draws_last_4y', 0)
                total_games = wins + losses + draws
                if total_games > 0:
                    equipos.at[idx, 'pct_victorias'] = round(wins / total_games, 2)

                equipos.at[idx, 'participaciones'] = stats.get('world_cup_participations_before', 0)
                equipos.at[idx, 'grupos_superados'] = stats.get('groups_passed_before', 0)

                rank = stats.get('fifa_rank_pre_tournament', 50)
                equipos.at[idx, 'elo_actual'] = max(1000, 2100 - (rank * 8))

    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas reales, usando fallback. Error: {e}")

    return equipos, fixture


rf_model, feature_cols = load_model()
equipos_df, fixture_df = load_data()


def construir_partidos_16avos():
    return construir_partidos_16avos_base(KNOCKOUT_LOOKUP, ROUND_OF_32_BRACKET_ORDER)


def calcular_features_partido(equipo_A, equipo_B):
    stats_A = obtener_stats_equipo(equipo_A)
    stats_B = obtener_stats_equipo(equipo_B)

    features = {
        'diff_elo': stats_A['elo_actual'] - stats_B['elo_actual'],
        'diff_ranking_fifa': stats_B['rank_fifa'] - stats_A['rank_fifa'],
        'diff_goles_favor_prom': stats_A['goles_favor_prom'] - stats_B['goles_favor_prom'],
        'diff_goles_contra_prom': stats_A['goles_contra_prom'] - stats_B['goles_contra_prom'],
        'diff_pct_victorias': stats_A['pct_victorias'] - stats_B['pct_victorias'],
        'diff_participaciones_mundial': stats_A['participaciones'] - stats_B['participaciones'],
        'diff_mejor_instancia': stats_A['grupos_superados'] - stats_B['grupos_superados'],
    }
    return pd.DataFrame([features])[feature_cols]


def simular_bracket_unico(aleatorio=False):
    return simular_bracket_unico_base(
        construir_partidos_16avos,
        rf_model,
        calcular_features_partido,
        RESULTADOS_CONFIRMADOS_16AVOS,
        aleatorio=aleatorio,
    )


def simular_bracket_montecarlo(n_iter=2000):
    return simular_bracket_montecarlo_base(
        n_iter,
        construir_partidos_16avos,
        rf_model,
        calcular_features_partido,
        RESULTADOS_CONFIRMADOS_16AVOS,
    )


def simular_mundial_2026_montecarlo(n_iter=2000):
    return simular_mundial_2026_montecarlo_base(
        n_iter,
        construir_partidos_16avos,
        rf_model,
        calcular_features_partido,
        RESULTADOS_CONFIRMADOS_16AVOS,
    )


def simulacion_monte_carlo(probabilidades_partidos, n_iter=10000):
    equipos_unicos = set()
    for eq_A, eq_B in probabilidades_partidos.keys():
        equipos_unicos.add(eq_A)
        equipos_unicos.add(eq_B)
        
    resultados = {equipo: {'Sale 1.0': 0, 'Sale 2.0': 0, 'Eliminada': 0} for equipo in equipos_unicos}
    
    for _ in range(n_iter):
        puntos = {equipo: 0 for equipo in resultados.keys()}
        
        for partido, probs in probabilidades_partidos.items():
            eq_A, eq_B = partido
            resultado = np.random.choice(rf_model.classes_, p=probs)
            
            if resultado == 'Gana_A':
                puntos[eq_A] += 3
            elif resultado == 'Gana_B':
                puntos[eq_B] += 3
            else:
                puntos[eq_A] += 1
                puntos[eq_B] += 1
                
        posiciones = sorted(puntos.items(), key=lambda x: x[1], reverse=True)
        
        resultados[posiciones[0][0]]['Sale 1.0'] += 1
        resultados[posiciones[1][0]]['Sale 2.0'] += 1
        for i in range(2, len(posiciones)):
            resultados[posiciones[i][0]]['Eliminada'] += 1
            
    for eq in resultados:
        for pos in resultados[eq]:
            resultados[eq][pos] = (resultados[eq][pos] / n_iter) * 100
            
    return pd.DataFrame.from_dict(resultados, orient='index')


def _formatear_nombre_mostrar(nombre):
    normalizaciones = {
        'Curacao': 'Curaçao',
    }
    return normalizaciones.get(str(nombre), str(nombre))

ALIAS_EQUIPOS = {
    'Canada': 'Canada',
    'Canadá': 'Canada',
    'South Africa': 'South Africa',
    'Sudafrica': 'South Africa',
    'Sudáfrica': 'South Africa',
    'Brazil': 'Brazil',
    'Brasil': 'Brazil',
    'Japan': 'Japan',
    'Japon': 'Japan',
    'Japón': 'Japan',
}

TEAM_CODE_TO_NAME = {
    'ALE': 'Germany',
    'PAR': 'Paraguay',
    'FRA': 'France',
    'SUE': 'Sweden',
    'CAN': 'Canada',
    'RSA': 'South Africa',
    'HOL': 'Netherlands',
    'MAR': 'Morocco',
    'POR': 'Portugal',
    'CRO': 'Croatia',
    'ESP': 'Spain',
    'AUT': 'Austria',
    'USA': 'United States',
    'BOS': 'Bosnia and Herzegovina',
    'BEL': 'Belgium',
    'SEN': 'Senegal',
    'BRA': 'Brazil',
    'JPN': 'Japan',
    'CIV': "Côte d'Ivoire",
    'NOR': 'Norway',
    'MEX': 'Mexico',
    'ECU': 'Ecuador',
    'ENG': 'England',
    'COD': 'DR Congo',
    'ARG': 'Argentina',
    'CPV': 'Cape Verde',
    'AUS': 'Australia',
    'EGY': 'Egypt',
    'SUI': 'Switzerland',
    'ALG': 'Algeria',
    'COL': 'Colombia',
    'GHA': 'Ghana',
}

GROUP_STANDINGS = {
    'A': [
        ('Mexico', 3, 3, 0, 0, 6, 0, 6, 9),
        ('South Africa', 3, 1, 1, 1, 2, 3, -1, 4),
        ('South Korea', 3, 1, 0, 2, 2, 3, -1, 3),
        ('Czech Republic', 3, 0, 1, 2, 2, 6, -4, 1),
    ],
    'B': [
        ('Switzerland', 3, 2, 1, 0, 7, 3, 4, 7),
        ('Canada', 3, 1, 1, 1, 8, 3, 5, 4),
        ('Bosnia and Herzegovina', 3, 1, 1, 1, 5, 6, -1, 4),
        ('Qatar', 3, 0, 1, 2, 2, 10, -8, 1),
    ],
    'C': [
        ('Brazil', 3, 2, 1, 0, 7, 1, 6, 7),
        ('Morocco', 3, 2, 1, 0, 6, 3, 3, 7),
        ('Scotland', 3, 1, 0, 2, 1, 4, -3, 3),
        ('Haiti', 3, 0, 0, 3, 2, 8, -6, 0),
    ],
    'D': [
        ('United States', 3, 2, 0, 1, 8, 4, 4, 6),
        ('Australia', 3, 1, 1, 1, 2, 2, 0, 4),
        ('Paraguay', 3, 1, 1, 1, 2, 4, -2, 4),
        ('Turkey', 3, 1, 0, 2, 3, 5, -2, 3),
    ],
    'E': [
        ('Germany', 3, 2, 0, 1, 10, 4, 6, 6),
        ('Ivory Coast', 3, 2, 0, 1, 4, 2, 2, 6),
        ('Ecuador', 3, 1, 1, 1, 2, 2, 0, 4),
        ('Curaçao', 3, 0, 1, 2, 1, 9, -8, 1),
    ],
    'F': [
        ('Netherlands', 3, 2, 1, 0, 10, 4, 6, 7),
        ('Japan', 3, 1, 2, 0, 7, 3, 4, 5),
        ('Sweden', 3, 1, 1, 1, 7, 7, 0, 4),
        ('Tunisia', 3, 0, 0, 3, 2, 12, -10, 0),
    ],
    'G': [
        ('Belgium', 3, 1, 2, 0, 6, 2, 4, 5),
        ('Egypt', 3, 1, 2, 0, 5, 3, 2, 5),
        ('Iran', 3, 0, 3, 0, 3, 3, 0, 3),
        ('New Zealand', 3, 0, 1, 2, 4, 10, -6, 1),
    ],
    'H': [
        ('Spain', 3, 2, 1, 0, 5, 0, 5, 7),
        ('Cape Verde', 3, 0, 3, 0, 2, 2, 0, 3),
        ('Uruguay', 3, 0, 2, 1, 3, 4, -1, 2),
        ('Saudi Arabia', 3, 0, 2, 1, 1, 5, -4, 2),
    ],
    'I': [
        ('France', 3, 3, 0, 0, 10, 2, 8, 9),
        ('Norway', 3, 2, 0, 1, 8, 7, 1, 6),
        ('Senegal', 3, 1, 0, 2, 8, 6, 2, 3),
        ('Iraq', 3, 0, 0, 3, 1, 12, -11, 0),
    ],
    'J': [
        ('Argentina', 3, 3, 0, 0, 8, 1, 7, 9),
        ('Austria', 3, 1, 1, 1, 6, 6, 0, 4),
        ('Algeria', 3, 1, 1, 1, 5, 7, -2, 4),
        ('Jordan', 3, 0, 0, 3, 3, 8, -5, 0),
    ],
    'K': [
        ('Colombia', 3, 2, 1, 0, 4, 1, 3, 7),
        ('Portugal', 3, 1, 2, 0, 6, 1, 5, 5),
        ('DR Congo', 3, 1, 1, 1, 4, 3, 1, 4),
        ('Uzbekistan', 3, 0, 0, 3, 2, 11, -9, 0),
    ],
    'L': [
        ('England', 3, 2, 1, 0, 6, 2, 4, 7),
        ('Croatia', 3, 2, 0, 1, 5, 5, 0, 6),
        ('Ghana', 3, 1, 1, 1, 2, 2, 0, 4),
        ('Panama', 3, 0, 0, 3, 0, 4, -4, 0),
    ],
}

ROUND_OF_32_FIXTURES = [
    ('ALE-PAR', 'Germany', 'Paraguay'),
    ('FRA-SUE', 'France', 'Sweden'),
    ('CAN-RSA', 'Canada', 'South Africa'),
    ('HOL-MAR', 'Netherlands', 'Morocco'),
    ('POR-CRO', 'Portugal', 'Croatia'),
    ('ESP-AUT', 'Spain', 'Austria'),
    ('USA-BOS', 'United States', 'Bosnia and Herzegovina'),
    ('BEL-SEN', 'Belgium', 'Senegal'),
    ('BRA-JPN', 'Brazil', 'Japan'),
    ('CIV-NOR', "Côte d'Ivoire", 'Norway'),
    ('MEX-ECU', 'Mexico', 'Ecuador'),
    ('ENG-COD', 'England', 'DR Congo'),
    ('ARG-CPV', 'Argentina', 'Cape Verde'),
    ('AUS-EGY', 'Australia', 'Egypt'),
    ('SUI-ALG', 'Switzerland', 'Algeria'),
    ('COL-GHA', 'Colombia', 'Ghana'),
]

ROUND_OF_32_LEFT_BRANCHES = [
    ('ALE-PAR', 'FRA-SUE'),
    ('CAN-RSA', 'HOL-MAR'),
    ('POR-CRO', 'ESP-AUT'),
    ('USA-BOS', 'BEL-SEN'),
]

ROUND_OF_32_RIGHT_BRANCHES = [
    ('BRA-JPN', 'CIV-NOR'),
    ('MEX-ECU', 'ENG-COD'),
    ('ARG-CPV', 'AUS-EGY'),
    ('SUI-ALG', 'COL-GHA'),
]

# Orden canónico del cuadro de 16avos (izquierda + derecha) usado para
# encadenar las rondas siguientes: octavos, cuartos, semifinal y final.
ROUND_OF_32_BRACKET_ORDER = [codigo for par in ROUND_OF_32_LEFT_BRANCHES for codigo in par] + \
    [codigo for par in ROUND_OF_32_RIGHT_BRANCHES for codigo in par]

CRUCES_CONFIRMADOS_16AVOS = [
    ('Canada', 'South Africa', '1-0', 'Canada'),
    ('Brazil', 'Japan', '2-1', 'Brazil'),
]


def cargar_grupos_desde_csv():
    try:
        df = pd.read_csv(GROUPS_CSV_PATH)
        columnas = {c.lower().strip(): c for c in df.columns}
        requeridas = ['grupo', 'pos', 'equipo', 'pj', 'g', 'e', 'p', 'gf', 'gc', 'dg', 'pts']
        if df.empty or not all(col in columnas for col in requeridas):
            raise ValueError('CSV de grupos incompleto')

        renombres = {columnas[key]: key.capitalize() if key not in {'pj', 'gf', 'gc', 'dg', 'pts'} else key.upper() for key in columnas if key in requeridas}
        df = df.rename(columns=renombres)
        if 'estado' in columnas and columnas['estado'] != 'Estado':
            df = df.rename(columns={columnas['estado']: 'Estado'})
        if 'Estado' not in df.columns:
            df['Estado'] = df['Pos'].apply(lambda pos: 'Avanza' if int(pos) <= 2 else 'Eliminado')

        df['Grupo'] = df['Grupo'].astype(str).str.strip().str.upper()
        df['Equipo'] = df['Equipo'].astype(str).map(_formatear_nombre_mostrar)
        order = ['Grupo', 'Pos', 'Equipo', 'PJ', 'G', 'E', 'P', 'GF', 'GC', 'DG', 'Pts', 'Estado']
        return df[order].sort_values(['Grupo', 'Pos']).reset_index(drop=True)
    except Exception:
        filas = []
        for grupo, equipos in GROUP_STANDINGS.items():
            for posicion, (equipo, pld, w, d, l, gf, ga, gd, pts) in enumerate(equipos, start=1):
                filas.append({
                    'Grupo': grupo,
                    'Pos': posicion,
                    'Equipo': _formatear_nombre_mostrar(equipo),
                    'PJ': pld,
                    'G': w,
                    'E': d,
                    'P': l,
                    'GF': gf,
                    'GC': ga,
                    'DG': gd,
                    'Pts': pts,
                    'Estado': 'Avanza' if posicion <= 2 else 'Eliminado',
                })
        return pd.DataFrame(filas)


def cargar_llaves_desde_csv():
    try:
        df = pd.read_csv(KNOCKOUT_CSV_PATH)
        columnas = {c.lower().strip(): c for c in df.columns}
        requeridas = ['codigo', 'equipo_a', 'equipo_b']
        if df.empty or not all(col in columnas for col in requeridas):
            raise ValueError('CSV de llaves incompleto')

        df = df.rename(columns={
            columnas['codigo']: 'Codigo',
            columnas['equipo_a']: 'Equipo_A',
            columnas['equipo_b']: 'Equipo_B',
        })
        if 'estado' in columnas:
            df = df.rename(columns={columnas['estado']: 'Estado'})
        if 'marcador' in columnas:
            df = df.rename(columns={columnas['marcador']: 'Marcador'})
        if 'penales' in columnas:
            df = df.rename(columns={columnas['penales']: 'Penales'})
        if 'ganador' in columnas:
            df = df.rename(columns={columnas['ganador']: 'Ganador'})

        if 'Estado' not in df.columns:
            df['Estado'] = 'Por jugar'
        if 'Marcador' not in df.columns:
            df['Marcador'] = ''
        if 'Penales' not in df.columns:
            df['Penales'] = ''
        if 'Ganador' not in df.columns:
            df['Ganador'] = ''

        df['Equipo_A'] = df['Equipo_A'].astype(str).map(_formatear_nombre_mostrar)
        df['Equipo_B'] = df['Equipo_B'].astype(str).map(_formatear_nombre_mostrar)
        return df[['Codigo', 'Equipo_A', 'Equipo_B', 'Estado', 'Marcador', 'Penales', 'Ganador']].reset_index(drop=True)
    except Exception:
        filas = []
        for codigo, equipo_a, equipo_b in ROUND_OF_32_FIXTURES:
            filas.append({
                'Codigo': codigo,
                'Equipo_A': equipo_a,
                'Equipo_B': equipo_b,
                'Estado': 'Por jugar',
                'Marcador': '',
                'Penales': '',
                'Ganador': '',
            })
        return pd.DataFrame(filas)


def obtener_mejores_terceros(df_grupos):
    terceros = df_grupos[df_grupos['Pos'] == 3].copy()
    terceros = terceros.sort_values(by=['Pts', 'DG', 'GF', 'Equipo'], ascending=[False, False, False, True])
    return set(terceros.head(8)['Equipo'].tolist())


def _marcador_con_penales(marcador, penales):
    marcador_str = str(marcador).strip() if pd.notna(marcador) else ''
    penales_str = str(penales).strip() if pd.notna(penales) else ''
    if not marcador_str:
        return '-'
    if penales_str:
        return f"{marcador_str} (pen. {penales_str})"
    return marcador_str


GROUPS_FINAL_DF = cargar_grupos_desde_csv()
BEST_THIRD_TEAMS = obtener_mejores_terceros(GROUPS_FINAL_DF)
KNOCKOUT_FIXTURES_DF = cargar_llaves_desde_csv()
KNOCKOUT_LOOKUP = {
    row['Codigo']: {
        'equipo_a': row['Equipo_A'],
        'equipo_b': row['Equipo_B'],
        'estado': row['Estado'],
        'marcador': row['Marcador'],
        'penales': row['Penales'],
        'ganador': row['Ganador'],
    }
    for _, row in KNOCKOUT_FIXTURES_DF.iterrows()
}

# Única fuente de verdad: se arma a partir del CSV, no hay datos hardcodeados
# que se puedan desincronizar entre sí.
RESULTADOS_CONFIRMADOS_16AVOS = {
    frozenset((row['Equipo_A'], row['Equipo_B'])): {
        'equipo_A': row['Equipo_A'],
        'equipo_B': row['Equipo_B'],
        'marcador': row['Marcador'],
        'penales': row['Penales'],
        'ganador': row['Ganador'],
    }
    for _, row in KNOCKOUT_FIXTURES_DF.iterrows()
    if str(row['Estado']).strip().lower() == 'confirmado'
}

CRUCES_CONFIRMADOS_16AVOS = [
    (row['Equipo_A'], row['Equipo_B'], _marcador_con_penales(row['Marcador'], row['Penales']), row['Ganador'])
    for _, row in KNOCKOUT_FIXTURES_DF.iterrows()
    if str(row['Estado']).strip().lower() == 'confirmado'
]


def obtener_mejores_terceros(df_grupos):
    terceros = df_grupos[df_grupos['Pos'] == 3].copy()
    terceros = terceros.sort_values(by=['Pts', 'DG', 'GF', 'Equipo'], ascending=[False, False, False, True])
    return set(terceros.head(8)['Equipo'].tolist())


def normalizar_equipo(nombre):
    nombre_limpio = str(nombre).strip()
    return ALIAS_EQUIPOS.get(nombre_limpio, nombre_limpio)


def construir_grupo_dataframe(grupo):
    df_grupo = GROUPS_FINAL_DF[GROUPS_FINAL_DF['Grupo'] == grupo].copy()
    if df_grupo.empty:
        filas = []
        for posicion, (equipo, pld, w, d, l, gf, ga, gd, pts) in enumerate(GROUP_STANDINGS[grupo], start=1):
            filas.append({
                'Grupo': grupo,
                'Pos': posicion,
                'Equipo': _formatear_nombre_mostrar(equipo),
                'PJ': pld,
                'G': w,
                'E': d,
                'P': l,
                'GF': gf,
                'GC': ga,
                'DG': gd,
                'Pts': pts,
                'Estado': 'Avanza' if posicion <= 2 or (posicion == 3 and equipo in BEST_THIRD_TEAMS) else 'Eliminado',
            })
        df_grupo = pd.DataFrame(filas)
    df_grupo['Estado'] = df_grupo.apply(
        lambda row: 'Avanza' if row['Pos'] <= 2 or (row['Pos'] == 3 and row['Equipo'] in BEST_THIRD_TEAMS) else 'Eliminado',
        axis=1,
    )
    return df_grupo[['Pos', 'Equipo', 'PJ', 'G', 'E', 'P', 'GF', 'GC', 'DG', 'Pts', 'Estado']].sort_values('Pos').reset_index(drop=True)


def etiqueta_fixture(codigo):
    if codigo in TEAM_CODE_TO_NAME:
        return codigo
    return codigo


def render_branch_side(title, branches, lookup):
    st.markdown(f"#### {title}")
    for idx, (fixture_a, fixture_b) in enumerate(branches, start=1):
        with st.container(border=True):
            st.markdown(f"**Cruce {idx}**")
            left_fixture = lookup.get(fixture_a, {})
            right_fixture = lookup.get(fixture_b, {})
            left_label = f"{left_fixture.get('equipo_a', fixture_a)} vs {left_fixture.get('equipo_b', '')}".strip()
            right_label = f"{right_fixture.get('equipo_a', fixture_b)} vs {right_fixture.get('equipo_b', '')}".strip()
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;gap:12px;align-items:center;'>"
                f"<div style='font-weight:700;font-size:0.95rem;'>{left_label}</div>"
                f"<div style='opacity:0.65;font-size:0.85rem;'>vs</div>"
                f"<div style='font-weight:700;font-size:0.95rem;text-align:right;'>{right_label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )


def obtener_fixtures_predictivos():
    fixtures = []
    for _, row in KNOCKOUT_FIXTURES_DF.iterrows():
        fixtures.append({
            'codigo': row['Codigo'],
            'equipo_A': row['Equipo_A'],
            'equipo_B': row['Equipo_B'],
        })
    return fixtures


def normalizar_equipo_busqueda(nombre):
    nombre_limpio = ALIAS_EQUIPOS.get(str(nombre).strip(), str(nombre).strip())
    nombre_limpio = unicodedata.normalize('NFKD', nombre_limpio).encode('ascii', 'ignore').decode('ascii')
    nombre_limpio = nombre_limpio.lower().replace('&', ' and ')
    nombre_limpio = re.sub(r'[^a-z0-9]+', ' ', nombre_limpio)
    return re.sub(r'\s+', ' ', nombre_limpio).strip()


def obtener_stats_equipo(equipo):
    equipo_norm = normalizar_equipo_busqueda(equipo)
    if 'equipo_normalizado' not in equipos_df.columns:
        equipos_df['equipo_normalizado'] = equipos_df['equipo'].apply(normalizar_equipo_busqueda)

    coincidencias = equipos_df[equipos_df['equipo_normalizado'] == equipo_norm]
    if not coincidencias.empty:
        return coincidencias.iloc[0]

    fallback = equipos_df.select_dtypes(include=[np.number]).median(numeric_only=True)
    fallback['equipo'] = equipo
    fallback['equipo_normalizado'] = equipo_norm
    for columna, valor in {
        'elo_actual': 1500,
        'rank_fifa': 50,
        'goles_favor_prom': 1.0,
        'goles_contra_prom': 1.0,
        'pct_victorias': 0.33,
        'participaciones': 0,
        'grupos_superados': 0,
    }.items():
        if columna not in fallback.index or pd.isna(fallback[columna]):
            fallback[columna] = valor
    return fallback


def obtener_partidos_16avos():
    if not fixture_df.empty:
        columnas = {c.lower(): c for c in fixture_df.columns}
        col_local = next((columnas[c] for c in columnas if 'home' in c or 'team1' in c or 'team_1' in c), None)
        col_visitante = next((columnas[c] for c in columnas if 'away' in c or 'team2' in c or 'team_2' in c), None)
        col_fase = next((columnas[c] for c in columnas if 'round' in c or 'stage' in c or 'phase' in c or 'fase' in c), None)

        if col_local and col_visitante:
            df_16avos = fixture_df.copy()
            if col_fase:
                mascara = df_16avos[col_fase].astype(str).str.contains('16|round of 16|octavos', case=False, na=False)
                if mascara.any():
                    df_16avos = df_16avos[mascara]

            partidos = []
            for _, row in df_16avos.iterrows():
                eq_A = normalizar_equipo(row[col_local])
                eq_B = normalizar_equipo(row[col_visitante])
                if eq_A and eq_B and eq_A != eq_B:
                    partidos.append((eq_A, eq_B))

            if partidos:
                return partidos

    return [('Canada', 'South Africa'), ('Brazil', 'Japan')]


def predecir_partido_eliminatoria(equipo_A, equipo_B):
    llave = frozenset((equipo_A, equipo_B))
    if llave in RESULTADOS_CONFIRMADOS_16AVOS:
        confirmado = RESULTADOS_CONFIRMADOS_16AVOS[llave]
        marcador = confirmado['marcador']
        penales = confirmado['penales']

        # Si nos piden el partido en el orden inverso al que está cargado
        # en el CSV, invertimos marcador y penales para que sigan siendo
        # consistentes con equipo_A/equipo_B recibidos.
        if confirmado['equipo_A'] != equipo_A:
            if '-' in str(marcador):
                g_a, g_b = str(marcador).split('-', 1)
                marcador = f"{g_b}-{g_a}"
            if penales and '-' in str(penales):
                p_a, p_b = str(penales).split('-', 1)
                penales = f"{p_b}-{p_a}"

        return {
            'partido': f'{equipo_A} vs {equipo_B}',
            'estado': 'Confirmado',
            'marcador': _marcador_con_penales(marcador, penales),
            'ganador': confirmado['ganador'],
            'prob_A': None,
            'prob_empate': None,
            'prob_B': None,
        }

    X_pred = calcular_features_partido(equipo_A, equipo_B)
    probs = rf_model.predict_proba(X_pred)[0]
    clases = list(rf_model.classes_)
    idx_A = clases.index('Gana_A')
    idx_E = clases.index('Empate')
    idx_B = clases.index('Gana_B')

    score_A = probs[idx_A] + (probs[idx_E] / 2)
    score_B = probs[idx_B] + (probs[idx_E] / 2)
    ganador = equipo_A if score_A >= score_B else equipo_B

    return {
        'partido': f'{equipo_A} vs {equipo_B}',
        'estado': 'Pronosticado',
        'marcador': '-',
        'ganador': ganador,
        'prob_A': probs[idx_A] * 100,
        'prob_empate': probs[idx_E] * 100,
        'prob_B': probs[idx_B] * 100,
    }


@st.cache_data
def simular_llaves_montecarlo(fixtures, n_iter=5000):
    conteo_ganadores = {fixture['equipo_A']: 0 for fixture in fixtures}
    conteo_ganadores.update({fixture['equipo_B']: 0 for fixture in fixtures})

    escenarios = []
    for fixture in fixtures:
        resultado = predecir_partido_eliminatoria(fixture['equipo_A'], fixture['equipo_B'])
        escenarios.append((fixture, resultado))

    for _ in range(n_iter):
        for fixture, resultado in escenarios:

            if resultado['estado'] == 'Confirmado':
                conteo_ganadores[resultado['ganador']] += 1
                continue

            probabilidades = np.array([
                resultado['prob_A'] / 100,
                resultado['prob_empate'] / 100,
                resultado['prob_B'] / 100,
            ])
            outcome = np.random.choice(['Gana_A', 'Empate', 'Gana_B'], p=probabilidades)

            if outcome == 'Gana_A':
                ganador = fixture['equipo_A']
            elif outcome == 'Gana_B':
                ganador = fixture['equipo_B']
            else:
                ganador = fixture['equipo_A'] if resultado['prob_A'] >= resultado['prob_B'] else fixture['equipo_B']

            conteo_ganadores[ganador] += 1

    filas = []
    for equipo, victorias in conteo_ganadores.items():
        if victorias > 0:
            filas.append({
                'Equipo': equipo,
                'Veces clasifica': victorias,
                'Probabilidad de clasificar %': round((victorias / n_iter) * 100, 1),
            })

    return pd.DataFrame(filas).sort_values('Probabilidad de clasificar %', ascending=False).reset_index(drop=True)


# --- Interfaz de Usuario ---
st.sidebar.header("Parámetros de Simulación")
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2rem;
        max-width: 1350px;
    }
    .hero-shell {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.24);
        border-radius: 28px;
        padding: 1.4rem 1.5rem 1.25rem;
        margin: 0.15rem 0 1rem;
        background:
            radial-gradient(circle at top right, rgba(255, 214, 102, 0.28), transparent 24%),
            radial-gradient(circle at bottom left, rgba(52, 211, 153, 0.16), transparent 26%),
            linear-gradient(135deg, #09111f 0%, #111c33 52%, #17324f 100%);
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.24);
    }
    .hero-shell::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, rgba(255,255,255,0.08), transparent 38%, rgba(255,255,255,0.04));
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.12);
        color: #e5eefb;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        backdrop-filter: blur(8px);
    }
    .hero-title {
        margin: 0.65rem 0 0.25rem;
        color: #f8fbff;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.02;
        letter-spacing: -0.03em;
        text-shadow: 0 1px 2px rgba(0,0,0,0.35);
    }
    .hero-subtitle {
        margin: 0;
        max-width: 78ch;
        color: rgba(226, 232, 240, 0.92);
        font-size: 1rem;
        line-height: 1.55;
    }
    .hero-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 1rem;
    }
    .hero-stat {
        padding: 0.55rem 0.8rem;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.11);
        color: #f8fbff;
        border: 1px solid rgba(255, 255, 255, 0.12);
        min-width: 155px;
    }
    .hero-stat .label {
        display: block;
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.8;
    }
    .hero-stat .value {
        display: block;
        margin-top: 0.18rem;
        font-weight: 700;
        font-size: 0.98rem;
    }
    .sim-card {
    border: 1px solid rgba(49, 51, 63, 0.18);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.6rem;
    background: linear-gradient(180deg, rgba(250, 250, 252, 0.98), rgba(244, 247, 255, 0.97));
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
    color: #0b1220; /* texto oscuro para buen contraste en temas claros y oscuros */
    font-size: 0.95rem;
}
.sim-card strong { color: #081027; }
.sim-card .match-meta { color: #334155; font-weight:600; margin-left:0.45rem; }
.sim-pill {
    display: inline-block;
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.sim-pill.ok { background: #176b39; color: #ffffff; }
.sim-pill.warn { background: #fff4d6; color: #8f5a00; }
.sim-pill.dark { background: #e8eefc; color: #1f3a68; }
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 0.55rem 0.95rem;
    background: rgba(15, 23, 42, 0.05);
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0f172a, #1d4ed8) !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero-shell">
        <span class="hero-badge">Scout Mundial 2026 · simulador predictivo</span>
        <h1 class="hero-title">Simulador del Mundial 2026</h1>
        <p class="hero-subtitle">
            Explorá la fase de grupos, las llaves de 16avos y el bracket completo hasta la final.
            La app combina datos históricos, resultados confirmados y un modelo de clasificación para estimar cada cruce.
        </p>
        <div class="hero-stats">
            <div class="hero-stat"><span class="label">Cobertura</span><span class="value">16avos a la final</span></div>
            <div class="hero-stat"><span class="label">Motor</span><span class="value">Random Forest + Monte Carlo</span></div>
            <div class="hero-stat"><span class="label">Datos</span><span class="value">Grupos, llaves y métricas históricas</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Fase de grupos", "Llaves", "Predicciones", "Bracket completo"])

with tabs[0]:
    st.success("Fase de grupos terminada.")
    st.caption("Tablas finales con posiciones, puntos, diferencia de gol y estado de clasificación.")

    group_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    group_cols = st.columns(3)

    for idx, grupo in enumerate(group_order):
        with group_cols[idx % 3]:
            df_grupo = construir_grupo_dataframe(grupo)
            champion = df_grupo.iloc[0]['Equipo']
            runner_up = df_grupo.iloc[1]['Equipo']
            third = df_grupo.iloc[2]['Equipo'] if len(df_grupo) > 2 else None
            third_pill = ""
            if third and third in BEST_THIRD_TEAMS:
                third_pill = f" <span class='sim-pill warn'>3° {third}</span>"

            st.markdown(f"<div class='sim-card'>", unsafe_allow_html=True)
            st.markdown(f"**Grupo {grupo}**")
            st.markdown(
                (
                    f"<span class='sim-pill dark'>1° {champion}</span> "
                    f"<span class='sim-pill ok'>2° {runner_up}</span>"
                    f"{third_pill}"
                ),
                unsafe_allow_html=True,
            )
            st.dataframe(
                df_grupo,
                hide_index=True,
                use_container_width=True,
                height=186,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Mejores terceros")
    terceros = GROUPS_FINAL_DF[GROUPS_FINAL_DF['Pos'] == 3].copy()
    terceros['Clasifica'] = terceros['Equipo'].isin(BEST_THIRD_TEAMS).map({True: 'Sí', False: 'No'})
    st.dataframe(
        terceros[['Grupo', 'Equipo', 'Pts', 'DG', 'GF', 'Clasifica']].sort_values(
            by=['Pts', 'DG', 'GF', 'Equipo'], ascending=[False, False, False, True]
        ),
        hide_index=True,
        use_container_width=True,
    )

with tabs[1]:
    st.markdown("### Bracket de 16avos")
    st.caption("Las llaves están separadas por lado y ordenadas de arriba hacia abajo como en el cuadro oficial.")

    left_col, right_col = st.columns(2)
    with left_col:
        render_branch_side("Lado izquierdo", ROUND_OF_32_LEFT_BRANCHES, KNOCKOUT_LOOKUP)
    with right_col:
        render_branch_side("Lado derecho", ROUND_OF_32_RIGHT_BRANCHES, KNOCKOUT_LOOKUP)

    st.markdown("### Cruces confirmados")
    for equipo_A, equipo_B, marcador, ganador in CRUCES_CONFIRMADOS_16AVOS:
        st.markdown(
            (
                "<div class='sim-card'>"
                "<span class='sim-pill ok'>Confirmado</span> "
                f"<strong>{equipo_A} vs {equipo_B}</strong> "
                f"<span class='match-meta'>— {marcador}</span> para <strong>{ganador}</strong>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

with tabs[2]:
    st.markdown("### Predicción de 16avos")
    st.caption("Se predicen los 16 cruces de la ronda de 32; los ya jugados aparecen como confirmados.")

    if st.button("Simular 16avos con Monte Carlo", type="primary"):
        with st.spinner("Corriendo simulación Monte Carlo..."):
            st.session_state['simulacion_16avos_mc'] = simular_llaves_montecarlo(obtener_fixtures_predictivos(), n_iter=5000)

    fixtures_predictivos = obtener_fixtures_predictivos()
    resultados_16avos = [predecir_partido_eliminatoria(f['equipo_A'], f['equipo_B']) for f in fixtures_predictivos]

    df_16avos = pd.DataFrame(resultados_16avos)
    for columna in ['prob_A', 'prob_empate', 'prob_B']:
        if columna in df_16avos.columns:
            df_16avos[columna] = pd.to_numeric(df_16avos[columna], errors='coerce').round(1)
    df_16avos[['prob_A', 'prob_empate', 'prob_B']] = df_16avos[['prob_A', 'prob_empate', 'prob_B']].fillna('-')

    st.dataframe(
        df_16avos.rename(columns={
            'partido': 'Partido',
            'estado': 'Estado',
            'marcador': 'Marcador',
            'ganador': 'Ganador',
            'prob_A': 'Prob. A %',
            'prob_empate': 'Empate %',
            'prob_B': 'Prob. B %',
        }),
        use_container_width=True,
    )

    confirmados = [r for r in resultados_16avos if r['estado'] == 'Confirmado']
    if confirmados:
        st.markdown("#### Partidos ya jugados")
        for resultado in confirmados:
            st.write(f"{resultado['partido']}: {resultado['marcador']} para {resultado['ganador']}")

    if 'simulacion_16avos_mc' in st.session_state:
        st.markdown("#### Monte Carlo")
        st.caption("Frecuencia de clasificación estimada tras múltiples corridas de la ronda de 16avos.")
        st.dataframe(st.session_state['simulacion_16avos_mc'], use_container_width=True, hide_index=True)

with tabs[3]:
    st.markdown("### Bracket completo: de 16avos a la Final")
    st.caption(
        "Encadena los resultados de 16avos con Octavos, Cuartos, Semifinal y Final. "
        "Los cruces ya jugados respetan el resultado real; el resto se resuelve con el modelo entrenado."
    )
    st.info(
        "La simulación toma como base la fase de grupos ya terminada y las llaves de 16avos aseguradas; "
        "desde ahí corre Monte Carlo hasta definir el campeón."
    )

    col_det, col_mc = st.columns(2)

    with col_det:
        st.markdown("**Predicción puntual**")
        st.caption("Un solo camino: en cada cruce gana el equipo con mayor probabilidad según el modelo.")
        if st.button("Calcular camino más probable"):
            with st.spinner("Encadenando rondas..."):
                st.session_state['bracket_unico'] = simular_bracket_unico(aleatorio=False)

    with col_mc:
        st.markdown("**Simulación Monte Carlo**")
        st.caption("Miles de corridas aleatorias para estimar probabilidades de avance por equipo.")
        n_iter_bracket = st.slider("Iteraciones", min_value=500, max_value=10000, value=2000, step=500)
        if st.button("Simular cuadro 16avos -> Final", type="primary"):
            with st.spinner(f"Corriendo {n_iter_bracket} simulaciones del cuadro completo..."):
                st.session_state['bracket_mc'] = simular_mundial_2026_montecarlo(n_iter=n_iter_bracket)

    if 'bracket_unico' in st.session_state:
        detalle_por_etapa, campeon = st.session_state['bracket_unico']
        st.markdown("#### Camino más probable")
        st.success(f"🏆 Campeón pronosticado: **{campeon}**")

        for etapa in ETAPAS_BRACKET:
            partidos_etapa = detalle_por_etapa[etapa]
            with st.expander(f"{etapa} · {len(partidos_etapa)} partido(s)", expanded=(etapa == 'Final')):
                for fila in partidos_etapa:
                    clase_pill = 'ok' if fila['estado'] == 'Confirmado' else 'dark'
                    st.markdown(
                        f"<div class='sim-card' style='margin-bottom:0.5rem;'>"
                        f"<span class='sim-pill {clase_pill}'>{fila['estado']}</span> "
                        f"<strong>{fila['equipo_A']} vs {fila['equipo_B']}</strong> "
                        f"→ Gana <strong>{fila['ganador']}</strong>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    if 'bracket_mc' in st.session_state:
        st.markdown("#### Probabilidades Monte Carlo por etapa")
        st.caption("Porcentaje de simulaciones (de las corridas elegidas arriba) en que cada equipo alcanzó cada instancia del torneo.")
        st.dataframe(st.session_state['bracket_mc'], use_container_width=True, hide_index=True)