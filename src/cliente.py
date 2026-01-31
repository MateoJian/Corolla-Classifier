import requests
import time
import sys

API_URL = "http://localhost:8080"

def enviar_imagen(ruta_imagen):
    with open(ruta_imagen, "rb") as f:
        files = {"file": f}
        resp = requests.post(f"{API_URL}/lista/tareas", files=files)

    if resp.status_code != 201:
        print("Error al enviar la imagen:", resp.json())
        return None

    data = resp.json()
    job_id = data["job_id"]
    print(f"Tarea creada con ID: {job_id}")
    return job_id


def consultar_resultado(job_id):
    url = f"{API_URL}/lista/tarea/{job_id}"

    while True:
        resp = requests.get(url)

        if resp.status_code == 404:
            print("La tarea no existe.")
            return None

        data = resp.json()["tarea"]
        estado = data["status"]

        if estado == "pendiente":
            print("Procesando...", end="\r")
            time.sleep(1)
            continue

        if estado == "completado":
            print("\nResultado listo:")
            print(f"Clase: {data['clase']}")
            print(f"Probabilidad: {data['probabilidad']}")
            return data

ruta = sys.argv[1]

job_id = enviar_imagen(ruta)

if job_id:
    consultar_resultado(job_id)

# Llamar con python3 cliente.py 'ruta_imagen'