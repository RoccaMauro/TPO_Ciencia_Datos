import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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
    print("✅ Gráfico EDA generado: eda_distribucion.png")

def entrenar_y_evaluar(df):
    # Rellenamos los datos faltantes con 0 para no perder los partidos
    df = df.fillna(0)
    
    # Seleccionar features numéricos generados en el ETL
    features = [
        'diff_elo', 'diff_ranking_fifa', 'diff_goles_favor_prom', 
        'diff_goles_contra_prom', 'diff_pct_victorias', 
        'diff_participaciones_mundial', 'diff_mejor_instancia'
    ]
    
    X = df[features]
    y = df['etiqueta']
    
    # División en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\n--- Evaluando Árbol de Decisión ---")
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.2f}")
    
    print("\n--- Evaluando Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    
    # Importancia de las variables (Útil para la presentación del TPO)
    importancia = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("\nVariables más influyentes del modelo:")
    print(importancia)
    
    # Guardar el modelo Random Forest y la lista de features para la app
    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(features, "features.pkl")
    print("\n✅ Modelo predictivo guardado como 'rf_model.pkl'")
    print("✅ Variables guardadas como 'features.pkl'")

if __name__ == "__main__":
    df = cargar_datos()
    analisis_exploratorio(df)
    entrenar_y_evaluar(df)