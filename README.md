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

Para explorar la estructura del dataset, se ha entrenado un autoencoder que reduce las imágenes a un espacio latente de 2 dimensiones.
Esto permite entender mejor la distribución del dataset.

El sistema de inferencia está completamente dockerizado y compuesto por:
Flask API
Expone un endpoint /predict para recibir imágenes y devolver la clase predicha.
Gestiona peticiones de forma ligera y eficiente.
Redis
Actúa como cola de tareas para desacoplar la API del proceso de inferencia.
Permite manejar múltiples peticiones concurrentes.
Nginx
Servidor reverse proxy.
Maneja el tráfico entrante y distribuye las peticiones hacia Flask.
Aporta seguridad y escalabilidad.
Docker Compose
Orquesta todos los servicios.
Facilita la reproducibilidad del entorno.
