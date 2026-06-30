import unicodedata
import re

ALIAS_EQUIPOS = {
    "Brasil": "Brazil",
    "Sudáfrica": "South Africa",
    "Canadá": "Canada",
    "Japón": "Japan",
}

def normalizar_equipo(nombre: str) -> str:
    nombre = str(nombre).strip()
    nombre = ALIAS_EQUIPOS.get(nombre, nombre)

    nombre = unicodedata.normalize("NFKD", nombre)
    nombre = nombre.encode("ascii", "ignore").decode("ascii")
    nombre = nombre.lower()
    nombre = re.sub(r"[^a-z0-9]+", " ", nombre)
    return nombre.strip()