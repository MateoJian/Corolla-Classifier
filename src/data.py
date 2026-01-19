import kagglehub

# Download latest version
path = kagglehub.dataset_download("stevenalbert15/toyota-corolla-car-parts")

print("Path to dataset files:", path)