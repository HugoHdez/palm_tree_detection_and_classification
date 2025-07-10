import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Función que calcula el IoU (Intersection over Union) entre dos contenedores
def calculate_iou(first_box, second_box):
    fb_x = max(first_box[0], second_box[0])
    fb_y = max(first_box[1], second_box[1])
    sb_x = min(first_box[2], second_box[2])
    sb_y = min(first_box[3], second_box[3])
    intersection_area = max(0, sb_x - fb_x + 1) * max(0, sb_y - fb_y + 1)
    first_box_area = (first_box[2] - first_box[0] + 1) * (first_box[3] - first_box[1] + 1)
    second_box_area = (second_box[2] - second_box[0] + 1) * (second_box[3] - second_box[1] + 1)
    return intersection_area / float(first_box_area + second_box_area - intersection_area + 1e-6)

# Directorios de entrada y salida
gt_csv_path = r".\detect_and_classify_results\ground_truth_val.csv"
pred_csv_path = r".\detect_and_classify_results\yolo_detect_yolo_cls_resnet_env.csv"
save_dir = r".\detect_and_classify_results\evaluation_metrics_validation"
os.makedirs(save_dir, exist_ok=True)

# Nombres de clases
class_names_env = ['na', 'ot', 'rd', 'ru', 'ud', 'uj', 'vi']
class_names_palm = ['bcd', 'da', 'hme', 'pc', 'pme', 'tdi']

# Se cargan los datos de ground truth y predicciones
gt_df = pd.read_csv(gt_csv_path)
pred_df = pd.read_csv(pred_csv_path)

# Variables de Verdaderos Positivos (TP), Falsos Positivos (FP) y Falsos Negativos (FN)
tp, fp, fn = 0, 0, 0
correct_env_cls, correct_palm_cls = 0, 0
total_env_cls, total_palm_cls = 0, 0
all_true_env, all_pred_env = [], []
all_true_palm, all_pred_palm = [], []

# Se evaluan las predicciones del csv en cuestión
for _, pred in pred_df.iterrows():
    # Coordenadas del contenedor predicho
    pred_box = [pred["bbox_x1"], pred["bbox_y1"], pred["bbox_x2"], pred["bbox_y2"]]
    image = pred["image"]

    gt_image_df = gt_df[gt_df["image"] == image]
    matched = False

    # Se evalua si algún contenedor real fue predicho
    for _, gt in gt_image_df.iterrows():
        gt_box = [gt["bbox_x1"], gt["bbox_y1"], gt["bbox_x2"], gt["bbox_y2"]]
        # Se calcula el IoU y se verifica si coincide con el 50% mínimo requerido
        if calculate_iou(pred_box, gt_box) >= 0.5:
            matched = True
            tp += 1
            total_env_cls += 1
            total_palm_cls += 1

            all_true_env.append(gt["environment_class"])
            all_pred_env.append(pred["environment_class"])
            all_true_palm.append(gt["palm_class"])
            all_pred_palm.append(pred["palm_class"])

            if pred["environment_class"] == gt["environment_class"]:
                correct_env_cls += 1
            if pred["palm_class"] == gt["palm_class"]:
                correct_palm_cls += 1
            break
    # Si no se encontró coincidencia, es un falso positivo
    if not matched:
        fp += 1

# Se evaluan los falsos negativos, es decir, los contenedores reales que no fueron detectados
for _, gt in gt_df.iterrows():
    gt_box = [gt["bbox_x1"], gt["bbox_y1"], gt["bbox_x2"], gt["bbox_y2"]]
    image = gt["image"]
    pred_image_df = pred_df[pred_df["image"] == image]

    matched = any(calculate_iou(gt_box, [p["bbox_x1"], p["bbox_y1"], p["bbox_x2"], p["bbox_y2"]]) >= 0.5 for _, p in pred_image_df.iterrows())
    if not matched:
        fn += 1

# Métricas de evaluación
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
acc_env = correct_env_cls / total_env_cls if total_env_cls > 0 else 0
acc_palm = correct_palm_cls / total_palm_cls if total_palm_cls > 0 else 0

# Se imprime el resumen de resultados por pantalla y se guarda en un archivo
summary = f"""

Resultados de Detección y Clasificación:
----------------------------------------
True Positives (TP): {tp}
False Positives (FP): {fp}
False Negatives (FN): {fn}

Precisión de Detección: {precision:.2f}
Recall de Detección: {recall:.2f}

Accuracy Clasificación Ambiente: {correct_env_cls}/{total_env_cls} = {acc_env:.2f}
Accuracy Clasificación Tipo Palmera: {correct_palm_cls}/{total_palm_cls} = {acc_palm:.2f}

"""
print(summary)

with open(os.path.join(save_dir, "metricas.txt"), "w", encoding="utf-8") as f:
    f.write(summary)

# Gráfico de barras de TP, FP y FN
plt.bar(["TP", "FP", "FN"], [tp, fp, fn], color=["green", "red", "orange"])
plt.title("Resultados de Detección")
plt.ylabel("Cantidad")
plt.grid(axis="y")
plt.savefig(os.path.join(save_dir, "barras_tp_fp_fn.png"))
plt.close()

# Matriz de confusión de tipos de palmera, absoluta y normalizada
if all_true_palm:
    cm_palm = confusion_matrix(all_true_palm, all_pred_palm, labels=range(len(class_names_palm)))
    cm_palm_norm = cm_palm.astype("float") / cm_palm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm_palm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names_palm, yticklabels=class_names_palm, ax=axes[0])
    axes[0].set_title("Matriz de Confusión Absoluta - Tipo de Palmera")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Real")

    sns.heatmap(cm_palm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names_palm, yticklabels=class_names_palm, ax=axes[1])
    axes[1].set_title("Matriz de Confusión Normalizada - Tipo de Palmera")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Real")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "matrices_confusion_palm.png"))
    plt.close()

# Matriz de confusión de ambientes, absoluta y normalizada
if all_true_env:
    cm_env = confusion_matrix(all_true_env, all_pred_env, labels=range(len(class_names_env)))
    cm_env_norm = cm_env.astype("float") / cm_env.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cm_env, annot=True, fmt="d", cmap="Blues", xticklabels=class_names_env, yticklabels=class_names_env, ax=axes[0])
    axes[0].set_title("Matriz de Confusión Absoluta - Ambiente")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Real")

    sns.heatmap(cm_env_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names_env, yticklabels=class_names_env, ax=axes[1])
    axes[1].set_title("Matriz de Confusión Normalizada - Ambiente")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Real")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "matrices_confusion_env.png"))
    plt.close()