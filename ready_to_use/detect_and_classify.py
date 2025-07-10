import os
import cv2
import torch
import pandas as pd
import torch.nn as nn
import sys
from tqdm import tqdm
from torchvision import models, transforms
from torch.nn import functional as F
from ultralytics import YOLO

# Rutas a los modelos, ajustar según el entorno y rutas a los modelos
detection_model_path = r"F:\Universidad\Curso 2024-25\Segundo Semestre\TFG\detection_and_classification_palms\detections\fixed_normalization\yolov11m\bbox_45\train_yolov11m_usar\weights\best.pt"
yolo_cls_model_path = r"F:\Universidad\Curso 2024-25\Segundo Semestre\TFG\Desarrollo\dataset\classification_per_type\palm_type_classification_dataset\yolov11m_cls\resultado_final\palm_type_classification_model_45px_final_try\weights\best.pt"
resnet_env_model_path = r"cls_models\cls_env\resnet50_best.pth"
output_csv = r".\detect_and_classify_results.csv"

# Verificar argumento de la consola para el directorio de imágenes
if len(sys.argv) != 2:
    print(f"Uso: python {sys.argv[0]} [ruta_al_directorio]")
    sys.exit(1)

# Se captura el directorio de las imágenes desde el argumento de la consola
img_dir = sys.argv[1]

# Cargar modelos YOLO (detección y clasificación de tipos de palmeras))
detector = YOLO(detection_model_path)
yolo_cls = YOLO(yolo_cls_model_path)

# Cargar modelo ResNet50 para clasificación de ambiente
resnet_env = models.resnet50(weights=None)
resnet_env.fc = nn.Sequential(
    nn.Dropout(p=0.6),
    nn.Linear(resnet_env.fc.in_features, 7)
)

# Cargar pesos del modelo ResNet50
resnet_env.load_state_dict(torch.load(resnet_env_model_path, map_location='cpu'))
# Configurar el modelo ResNet50 para evaluación
resnet_env.eval()

# Verificar si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_env.to(device)

# Transformación para la entrada de imágenes con el modelo cargado con arquitectura ResNet50
transform_resnet = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


rows = []

# Se recorre el directorio de imágenes cargando barra de progreso para ayuda visual
for filename in tqdm(os.listdir(img_dir), desc="Procesando imágenes"):
    # Formatos de archivos válidos
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    
    # Ruta completa a la imagen
    image_path = os.path.join(img_dir, filename)
    # Se lee la imagen
    image = cv2.imread(image_path)

    # Se verifican errores al leer la imagen
    if image is None:
        print(f"Error leyendo {filename}")
        continue

    # Se captura el alto y ancho de la imagen
    img_height, img_width = image.shape[:2]

    # Se predice la detección de palmeras con el modelo cargado con YOLO
    results = detector.predict(source=image, conf=0.2, device=0, imgsz=960)[0]

    # Se recorren las detecciones de palmeras
    for box in results.boxes:
        # Se extraen las coordenadas de la caja delimitadora mapeandolas a entero y la confianza de detección
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detection_confidence = float(box.conf[0])

        # Se define el RoI de la palmera, para la clasificación de tipo de palmera
        palm_crop = image[y1:y2, x1:x2]

        # Se guarda temporalmente la imagen recortada para pasarla al clasificador
        temp_path = "temp_crop.jpg"
        cv2.imwrite(temp_path, palm_crop)
        
        # Se predice la clasificación del tipo de palmera con el modelo de YOLO cargado previamente
        palm_cls_result = yolo_cls.predict(source=temp_path, conf=0.0, device=0, imgsz=96)[0]
        os.remove(temp_path)

        # Se extrae la clase y la confianza de la clasificación de tipo de palmera        
        palm_class = int(palm_cls_result.probs.top1)
        palm_confidence = float(palm_cls_result.probs.top1conf)

        # Se calcula el centro del RoI del tipo de palmera para producir el RoI del ambiente sobre el centro de la misma
        x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
        roi_size = 256
        # Se calculan las coordenadas del RoI del ambiente, asegurando que no se salga de los límites de la imagen
        env_x1 = max(0, x_center - roi_size // 2)
        env_y1 = max(0, y_center - roi_size // 2)
        env_x2 = min(img_width, x_center + roi_size // 2)
        env_y2 = min(img_height, y_center + roi_size // 2)
        env_crop = image[env_y1:env_y2, env_x1:env_x2]

        # Se le pasa la transformación a la imagen para el modelo ResNet50
        env_tensor = transform_resnet(env_crop).unsqueeze(0).to(device)
        # Modo de inferencia (predicción)
        with torch.no_grad():
            logits = resnet_env(env_tensor)
            probs = F.softmax(logits, dim=1)
            # Clase predicha y su confianza
            env_class = int(torch.argmax(probs, dim=1).item())
            env_conf = float(torch.max(probs).item())

        # Guardar resultados
        rows.append({
            "image": filename,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "det_confidence": detection_confidence,
            "palm_class": palm_class,
            "palm_confidence": palm_confidence,
            "environment_class": env_class,
            "environment_confidence": env_conf
        })

# Guardar CSV
df = pd.DataFrame(rows)
df.to_csv(output_csv, index=False)
print(f"Guardado: {output_csv}")