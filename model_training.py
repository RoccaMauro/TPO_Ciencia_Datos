import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def cargar_datos():
    # Carga el dataset generado exitosamente por tu ETL
    return pd.read_csv("dataset_integrado.csv")

def analisis_exploratorio(df):
    """Genera visualizaciones básicas del EDA."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='etiqueta', data=df, order=['Gana_A', 'Empate', 'Gana_B'])
    plt.title("Distribución de Resultados Históricos")
    plt.savefig("eda_distribucion.png")
    plt.close()
    print("Grafico EDA generado: eda_distribucion.png")

def calcular_pesos_recencia(fechas, fecha_referencia=None, vida_media_anios=20):
    """Da mas importancia a partidos recientes mediante decaimiento exponencial."""
    if fecha_referencia is None:
        fecha_referencia = fechas.max()

    antiguedad_anios = (fecha_referencia - fechas).dt.days.clip(lower=0) / 365.25
    return pd.Series(
        np.power(0.5, antiguedad_anios / vida_media_anios),
        index=fechas.index,
    )


def entrenar_y_evaluar(df):
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.dropna(subset=['fecha']).sort_values('fecha').reset_index(drop=True)
    
    # Seleccionar features numéricos generados en el ETL
    features = [
        'diff_elo', 'diff_goles_favor_prom',
        'diff_goles_contra_prom', 'diff_pct_victorias', 
        'diff_participaciones_mundial', 'diff_mejor_instancia'
    ]

    df[features] = df[features].fillna(0)
    
    X = df[features]
    y = df['etiqueta']
    
    # División en entrenamiento y prueba
    posicion_corte = int(len(df) * 0.8)
    fecha_corte = df.loc[posicion_corte, 'fecha']
    mascara_train = df['fecha'] < fecha_corte
    mascara_test = ~mascara_train

    X_train, y_train = X.loc[mascara_train], y.loc[mascara_train]
    X_test, y_test = X.loc[mascara_test], y.loc[mascara_test]

    if X_train.empty or X_test.empty:
        raise ValueError("No hay suficientes fechas distintas para la division temporal")

    print("\n--- Division temporal ---")
    print(f"Entrenamiento: {df.loc[mascara_train, 'fecha'].min().date()} a "
          f"{df.loc[mascara_train, 'fecha'].max().date()} ({len(X_train)} partidos)")
    print(f"Prueba: {df.loc[mascara_test, 'fecha'].min().date()} a "
          f"{df.loc[mascara_test, 'fecha'].max().date()} ({len(X_test)} partidos)")

    pesos_train = calcular_pesos_recencia(
        df.loc[mascara_train, 'fecha'],
        fecha_referencia=df.loc[mascara_train, 'fecha'].max(),
    )
    print(f"Peso temporal del partido mas antiguo: {pesos_train.min():.3f}")
    print(f"Peso temporal del partido mas reciente: {pesos_train.max():.3f}")
    
    print("\n--- Evaluando Árbol de Decisión ---")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train, sample_weight=pesos_train)
    y_pred_dt = dt.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
    
    print("\n--- Evaluando Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
    )
    rf.fit(X_train, y_train, sample_weight=pesos_train)
    y_pred_rf = rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    
    # Importancia de las variables (Útil para la presentación del TPO)
    importancia = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nVariables más influyentes del modelo:")
    print(importancia)
    
    # Tras la evaluacion, el modelo de la app aprovecha todos los partidos.
    pesos_finales = calcular_pesos_recencia(df['fecha'])
    rf_final = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
    )
    rf_final.fit(X, y, sample_weight=pesos_finales)

    joblib.dump(rf_final, "rf_model.pkl")
    joblib.dump(features, "features.pkl")
    print(f"\nModelo final reentrenado con {len(X)} partidos")
    print("\nModelo predictivo guardado como 'rf_model.pkl'")
    print("Variables guardadas como 'features.pkl'")

if __name__ == "__main__":
    df = cargar_datos()
    analisis_exploratorio(df)
    entrenar_y_evaluar(df)
