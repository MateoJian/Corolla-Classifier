import torch
import numpy as np
import random
from PIL import Image 
from torchvision import transforms 

def set_seed(seed=7556):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def preprocesado_imagen(ruta_imagen):
    transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(ruta_imagen).convert("RGB")
    image = transform(image)
    return image