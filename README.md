# Corolla-Classifier
Car parts classifier using ResNet18.

Clasificación de piezas de Toyota Corolla con ResNet18, Autoencoder y despliegue Dockerizado

Este proyecto desarrolla un sistema completo de clasificación de imágenes para identificar diferentes partes de un Toyota Corolla, utilizando el dataset Toyota Corolla de Kaggle creado por StevenAlbert15. Además del modelo principal, incluye un pipeline de despliegue con Docker, Redis, Nginx y Flask, así como herramientas de visualización y explicabilidad del modelo.

Toyota Corolla Car Parts Dataset  
Kaggle: https://www.kaggle.com/datasets/stevenalbert15/toyota-corolla-car-parts
contiene imágenes de múltiples componentes del vehículo.

Se ha utilizado una ResNet18 preentrenada (transfer learning) con las siguientes características:
Fine-tuning de las últimas capas.
Entrenamiento con PyTorch.

```
Estructura:  
data/  
    Toyota Corolla Dataset/  //Carpeta con el dataset descargado  
models/  
    modelo_corolla.pth  
    modelo.py  
notebooks/  
    01_Entrenamiento_y_Evaluacion.ipynb  
src/  
    compose/  
        docker-compose.yml  
        nginx/  
            default.conf  
    __init__.py  
    apiflask.py  
    cliente.py  
    data.py  
    dataset.py  
    Dockerfile_api  
    Dockerfile_inference  
    inference.py  
    utils.py  
```