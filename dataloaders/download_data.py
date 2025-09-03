import kagglehub

# Download latest version
path = kagglehub.dataset_download("orvile/bus-uc-breast-ultrasound")

print("Path to dataset files:", path)