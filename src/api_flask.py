# coding: utf-8
from flask import Flask, jsonify, abort, make_response, request
import uuid
import redis
import os
import base64
import json

app = Flask(__name__)

redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_client = redis.Redis(host=redis_host, port=6379, decode_responses=True)

def obtener_tarea(job_id):
    data = redis_client.get(f"trabajo:{job_id}")
    if not data:
        return None
    return json.loads(data)

def guardar_tarea(tarea):
    redis_client.set(f"trabajo:{tarea['job_id']}", json.dumps(tarea))

def muestra_tarea(tarea):
    respuesta = {
        "job_id": tarea["job_id"],
        "status": tarea["status"]
    }
    if tarea["status"] == "completado":
        respuesta["clase"] = tarea["clase"]
        respuesta["probabilidad"] = tarea["probabilidad"]
    return respuesta

@app.route('/lista/tarea/<string:id>', methods=["GET"])
def get_status(id):
    tarea = obtener_tarea(id)
    if tarea is None:
        abort(404)
    return jsonify({"tarea": muestra_tarea(tarea)})

@app.route("/lista/tareas", methods=["POST"])
def create_tarea():
    if 'file' not in request.files:
        return make_response(jsonify({'error': 'No se envió ninguna imagen'}), 400)

    file = request.files['file']
    if file.filename == '':
        return make_response(jsonify({'error': 'Nombre de archivo vacío'}), 400)

    job_id = str(uuid.uuid4())
    img_bytes = file.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    tarea = {
        "job_id": job_id,
        "status": "pendiente",
        "clase": None,
        "probabilidad": None
    }
    
    guardar_tarea(tarea)

    redis_client.rpush("trabajos", 
        json.dumps({
            "job_id": job_id,
            "image": img_base64
        })
    )
    return jsonify(muestra_tarea(tarea)), 201


@app.errorhandler(404)
def no_encontrado(error):
    return make_response(jsonify({'error': 'Tarea inexistente'}), 404)

@app.errorhandler(400)
def solicitud_incorrecta(error):
    return make_response(jsonify({'error': 'Solicitud incorrecta'}), 400)