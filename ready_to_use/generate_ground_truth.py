import os
import json
import pandas as pd

# Cambiar por el directorio donde están los archivos JSON de tu dataset
json_dir = r"F:\Universidad\Curso 2024-25\Segundo Semestre\TFG\Desarrollo\dataset\fixed_normalization\Val\json_annotations"
output_csv = "ground_truth.csv"

rows = []

# Se itera sobre todos los archivos en el directorio que contengan anotaciones en formato JSON
for file in os.listdir(json_dir):

    # Abrir y cargar el archivo JSON
    with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
        data = json.load(f)

    # Obtener el nombre de la imagen asociada a las anotaciones
    image_name = data["image"]
    # Obtener el idebtificador de la imagen
    palm_id = data.get("id", "unknown")

    # Recorrer todas las palmeras anotadas en el archivo
    for palm in data.get("palms", []):
        # Obtener las coordenadas del bounding box en píxeles
        bbox = palm["bbox_pixel"]
        xc = bbox["x_center"]
        yc = bbox["y_center"]
        w = bbox["width"]
        h = bbox["height"]

        # Convertir el bounding box de formato (centro, ancho, alto) a (x1, y1, x2, y2)
        x1 = int(xc - w / 2)
        y1 = int(yc - h / 2)
        x2 = int(xc + w / 2)
        y2 = int(yc + h / 2)

        # Obtener las etiquetas de ambiente y tipo de palmera desde los atributos
        environment = palm["attributes"].get("Tip_amb")
        palm_type = palm["attributes"].get("Hibrid")

        # Ignorar palmeras con clase desconocida ("?")
        if environment == "?" or palm_type == "?":
            continue

        # Mapear clases de ambiente a índices numéricos (deben coincidir con los usados en entrenamiento)
        environment_class = {"na": 0, "ot": 1, "rd": 2, "ru": 3, "ud": 4, "uj": 5, "vi": 6}[environment.lower()]

        # Mapear clases de tipo de palmera a índices numéricos
        palm_class = {"bdc": 0, "da": 1, "hme": 2, "pc": 3, "pme": 4, "tdi": 5}[palm_type.lower()]

        # Agregar la información extraída a la lista de filas
        rows.append({
            "image": image_name,
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "palm_class": palm_class,
            "environment_class": environment_class
        })

# Crear un DataFrame de pandas con todas las filas recogidas
df = pd.DataFrame(rows)

# Guardar el DataFrame en formato CSV
df.to_csv(output_csv, index=False)

print(f"ground_truth.csv guardado en: {output_csv}")