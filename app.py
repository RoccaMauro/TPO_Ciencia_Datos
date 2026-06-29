import streamlit as st
import pandas as pd
import numpy as np
import joblib
import itertools
import kagglehub
import os

# --- Configuración de la App ---
st.set_page_config(page_title="Scout Mundial 2026", layout="wide")
st.title("🏆 Scout Mundial 2026 - Simulador Predictivo")
st.markdown("Simulación estocástica basada en modelos de Machine Learning y datos históricos reales.")

st.sidebar.markdown("🔗 **Motor de Datos:** Conectado a Datasets Oficiales")
st.sidebar.divider()

# --- Carga de Modelos y Datos ---
@st.cache_resource
def load_model():
    rf = joblib.load("rf_model.pkl")
    features = joblib.load("features.pkl")
    return rf, features

@st.cache_data
def load_data():
    # 1. Carga de los equipos y fixture (2026)
    try:
        equipos = pd.read_csv("teams_2026.csv") 
        fixture = pd.read_csv("schedule_2026.csv")
    except FileNotFoundError:
        base_path = kagglehub.dataset_download("areezvisram12/fifa-world-cup-2026-match-data-unofficial")
        equipos = pd.read_csv(os.path.join(base_path, "teams.csv"))
        fixture = pd.read_csv(os.path.join(base_path, "matches.csv"))

# Función de limpieza robusta para asegurar la integridad referencial
    def limpiar_placeholder(nombre):
        nombre_str = str(nombre).strip()
        nombre_lower = nombre_str.lower()
        
        # Normalizamos la cadena eliminando guiones y guiones bajos para evitar fallos de coincidencia
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
            # Normalizamos también la clave del diccionario por seguridad
            p_norm = placeholder.replace('-', ' ').replace('_', ' ')
            if p_norm in nombre_norm:
                return ganador_real
                
        # Modificamos la salvaguarda para que no interfiera con los mapeos correctos
        if 'tbc' in nombre_norm or ('play off' in nombre_norm and not any(k in nombre_norm for k in mapeo_repechajes.keys())):
            return 'Panama'
            
        return nombre_str

    # Normalización de Equipos
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

    # Normalización del Fixture
    fixture.columns = [c.lower().strip() for c in fixture.columns]
    col_home = next((c for c in fixture.columns if 'home' in c or 'team1' in c or 'team_1' in c), 'equipo_local')
    col_away = next((c for c in fixture.columns if 'away' in c or 'team2' in c or 'team_2' in c), 'equipo_visitante')
    col_fixture_group = next((c for c in fixture.columns if 'group' in c), 'grupo')

    fixture = fixture.rename(columns={col_home: 'equipo_local', col_away: 'equipo_visitante', col_fixture_group: 'grupo'})
    if 'grupo' in fixture.columns:
        fixture['grupo'] = fixture['grupo'].astype(str).str.replace('Group ', '', regex=False).str.strip()

    fixture['equipo_local'] = fixture['equipo_local'].apply(limpiar_placeholder)
    fixture['equipo_visitante'] = fixture['equipo_visitante'].apply(limpiar_placeholder)

    # -------------------------------------------------------------------
    # 3. EXTRACCIÓN DE ESTADÍSTICAS REALES (Reemplazando np.random)
    # -------------------------------------------------------------------
    try:
        # Nos conectamos al mismo dataset histórico del ETL
        team_dataset_path = kagglehub.dataset_download("harrachimustapha/fifa-world-cup-team-dataset")
        df_historico = pd.read_csv(os.path.join(team_dataset_path, "train.csv"))
        
        # Filtramos para quedarnos con la versión más reciente de cada equipo
        df_historico = df_historico.sort_values('version').drop_duplicates('team', keep='last')
        
        # Preparamos las columnas que la app va a llenar
        equipos['elo_actual'] = 1500 # Base neutral
        equipos['rank_fifa'] = 50
        equipos['goles_favor_prom'] = 1.0
        equipos['goles_contra_prom'] = 1.0
        equipos['pct_victorias'] = 0.33
        equipos['participaciones'] = 0
        equipos['grupos_superados'] = 0

        # Mapeamos los datos reales iterando sobre cada selección clasificada
        for idx, row in equipos.iterrows():
            pais = row['equipo']
            
            # Búsqueda del país en el historial
            datos_pais = df_historico[df_historico['team'] == pais]
            
            if not datos_pais.empty:
                stats = datos_pais.iloc[0]
                
                # Asignación de datos reales extraídos del CSV
                equipos.at[idx, 'rank_fifa'] = stats.get('fifa_rank_pre_tournament', 50)
                equipos.at[idx, 'goles_favor_prom'] = stats.get('goals_scored_last_4y', 4) / 4.0 
                equipos.at[idx, 'goles_contra_prom'] = stats.get('goals_received_last_4y', 4) / 4.0
                
                # Cálculo de porcentaje de victorias reciente
                wins = stats.get('wins_last_4y', 0)
                losses = stats.get('losses_last_4y', 0)
                draws = stats.get('draws_last_4y', 0)
                total_games = wins + losses + draws
                if total_games > 0:
                    equipos.at[idx, 'pct_victorias'] = round(wins / total_games, 2)
                    
                equipos.at[idx, 'participaciones'] = stats.get('world_cup_participations_before', 0)
                equipos.at[idx, 'grupos_superados'] = stats.get('groups_passed_before', 0)
                
                # Estimación de ELO basado en Ranking FIFA para no requerir un 3er dataset en tiempo real
                # Formula de correlación simplificada: ELO = 2100 - (Rank * 8)
                rank = stats.get('fifa_rank_pre_tournament', 50)
                equipos.at[idx, 'elo_actual'] = max(1000, 2100 - (rank * 8))

    except Exception as e:
        st.warning(f"No se pudieron cargar las estadísticas reales, usando fallback. Error: {e}")

    return equipos, fixture

rf_model, feature_cols = load_model()
equipos_df, fixture_df = load_data()

# --- Funciones Auxiliares ---
def obtener_partidos_reales(grupo_seleccionado):
    equipos_del_grupo = equipos_df[equipos_df['grupo'] == grupo_seleccionado]['equipo'].tolist()
    lista_partidos = []
    
    if 'equipo_local' in fixture_df.columns and 'equipo_visitante' in fixture_df.columns:
        for _, row in fixture_df.iterrows():
            eq_A, eq_B = row['equipo_local'], row['equipo_visitante']
            if eq_A in equipos_del_grupo and eq_B in equipos_del_grupo:
                lista_partidos.append((eq_A, eq_B))
                
    if not lista_partidos:
        return list(itertools.combinations(equipos_del_grupo, 2))
        
    return lista_partidos

def calcular_features_partido(equipo_A, equipo_B, df_stats):
    stats_A = df_stats[df_stats['equipo'] == equipo_A].iloc[0]
    stats_B = df_stats[df_stats['equipo'] == equipo_B].iloc[0]
    
    features = {
        'diff_elo': stats_A['elo_actual'] - stats_B['elo_actual'],
        'diff_ranking_fifa': stats_B['rank_fifa'] - stats_A['rank_fifa'], 
        'diff_goles_favor_prom': stats_A['goles_favor_prom'] - stats_B['goles_favor_prom'],
        'diff_goles_contra_prom': stats_A['goles_contra_prom'] - stats_B['goles_contra_prom'],
        'diff_pct_victorias': stats_A['pct_victorias'] - stats_B['pct_victorias'],
        'diff_participaciones_mundial': stats_A['participaciones'] - stats_B['participaciones'],
        'diff_mejor_instancia': stats_A['grupos_superados'] - stats_B['grupos_superados']
    }
    return pd.DataFrame([features])[feature_cols]

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

# --- Interfaz de Usuario ---
st.sidebar.header("Parámetros de Simulación")

grupos_disponibles = sorted([g for g in equipos_df['grupo'].unique() if g and g != 'nan'])
grupo_seleccionado = st.sidebar.selectbox("Seleccione el Grupo", grupos_disponibles)

n_simulaciones = st.sidebar.slider("Iteraciones Monte Carlo", 1000, 20000, 10000, step=1000)

if st.button("Ejecutar Simulación", type="primary"):
    
    partidos = obtener_partidos_reales(grupo_seleccionado)
    
    if not partidos:
        st.error(f"No se pudieron cargar los partidos para el Grupo {grupo_seleccionado}.")
        st.stop()
        
    probabilidades_partidos = {}
    
    st.subheader(f"Fixture y Probabilidades Base (Grupo {grupo_seleccionado})")
    cols = st.columns(3)
    
    for i, (eq_A, eq_B) in enumerate(partidos):
        if eq_A not in equipos_df['equipo'].values or eq_B not in equipos_df['equipo'].values:
            continue
            
        X_pred = calcular_features_partido(eq_A, eq_B, equipos_df)
        probs = rf_model.predict_proba(X_pred)[0]
        
        probabilidades_partidos[(eq_A, eq_B)] = probs
        
        clases = rf_model.classes_ 
        p_A = probs[list(clases).index('Gana_A')] * 100
        p_E = probs[list(clases).index('Empate')] * 100
        p_B = probs[list(clases).index('Gana_B')] * 100
        
        with cols[i % 3]:
            st.markdown(f"**{eq_A} vs {eq_B}**")
            st.write(f"{eq_A}: {p_A:.1f}% | Empate: {p_E:.1f}% | {eq_B}: {p_B:.1f}%")
            
    st.divider()
    
    with st.spinner(f'Consultando historial y ejecutando {n_simulaciones} iteraciones...'):
        df_resultados = simulacion_monte_carlo(probabilidades_partidos, n_simulaciones)
        
    df_resultados['Clasifica'] = df_resultados['Sale 1.0'] + df_resultados['Sale 2.0']
    df_resultados = df_resultados.sort_values(by='Clasifica', ascending=False)
    
    st.subheader("📊 Tabla Probabilística de Clasificación Final")
    st.dataframe(
        df_resultados[['Sale 1.0', 'Sale 2.0', 'Clasifica', 'Eliminada']].style.format("{:.1f}%").background_gradient(cmap='Blues', subset=['Clasifica']),
        use_container_width=True
    )