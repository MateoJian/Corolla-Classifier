import torch
import sys
import os
import redis
import io
import json
import base64
from PIL import Image
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.abspath(os.path.join('..')))
from models.modelo import Modelo

# 2. Carga del modelo (Fuera del bucle para que solo ocurra UNA VEZ)
modelo = Modelo()
modelo.load_state_dict(torch.load("../models/modelo_corolla.pth", map_location=device))
modelo.to(device)
modelo.eval()

# 3. Transformaciones (Nota: Para inferencia no usamos RandomResizedCrop, usamos CenterCrop o solo Resize)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Configuración de Redis
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
indice_clases = {0:'Windshield Fluid', 1:'Engine', 2:'Battery', 3:'Engine Cover', 4:'Air Filter', 5:'Reservoir Cap', 6:'Coolant Reservoir'}

print(f"Worker listo en {device}. Esperando imágenes...")

while True:
    try:
        _, raw_data = redis_client.blpop('trabajos')
        
        data = json.loads(raw_data)
        img_id = data["id"]
        
        # Decodificar imagen
        img_bytes = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image = transform(image).to(device).unsqueeze(0)

        with torch.no_grad():
            output = modelo(image)
            pred_idx = output.argmax(1).item()
            clase_final = indice_clases[pred_idx]
            confianza = torch.nn.functional.softmax(output, dim=1).max().item()

        resultado = {
            "clase": clase_final,
            "probabilidad": f"{confianza:.2f}",
            "status": "completado"
        }
        redis_client.set(img_id, json.dumps(resultado), ex=3600)
        
        print(f"Procesado ID {img_id}: {clase_final}")

    except Exception as e:
        print(f"Error procesando trabajo: {e}")