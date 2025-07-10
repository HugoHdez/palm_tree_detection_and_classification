import os
import cv2
import json
import glob

# Rutas
path = r"F:\Universidad\Curso 2024-25\Segundo Semestre\TFG\Desarrollo\dataset\fixed_normalization\Train"
labels_binaria = f'{path}/labels_binaria'
labels_ambiente = f'{path}/labels_ambiente'
labels_palmera = f'{path}/labels_palmera'
jsons_path = f'{path}/json_annotations'
worldfiles_path = f'{path}/worldfiles'


# Variables globales para mouse callback
clicked_point = None
click_done = False

# Función para dibujar bbox
def draw_bbox(image, bbox, color=(0, 255, 0)):
    h, w = image.shape[:2]
    _, x_c, y_c, bw, bh = map(float, bbox.strip().split())
    x1 = int((x_c - bw / 2) * w)
    y1 = int((y_c - bh / 2) * h)
    x2 = int((x_c + bw / 2) * w)
    y2 = int((y_c + bh / 2) * h)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    return image

def mouse_callback(event, x, y, flags, param):
    global clicked_point, click_done
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        click_done = True

# Recorrer los labels
label_files = sorted(glob.glob(os.path.join(labels_binaria, '*.txt')))

# A partir del pimer archivo.
for label_file in label_files[52:]:
    base_name = os.path.splitext(os.path.basename(label_file))[0]
    image_file = os.path.join(images_path, base_name + '.png')
    json_file = os.path.join(jsons_path, base_name + '.json')
    worldfile = os.path.join(worldfiles_path, base_name + '.pgw')  # archivo .pgw
    labels_palmera_file = os.path.join(labels_palmera, base_name + '.txt')
    labels_ambiente_file = os.path.join(labels_ambiente, base_name + '.txt')

    if not (os.path.exists(image_file) and os.path.exists(json_file) and os.path.exists(worldfile)):
        print(f"Faltan archivos necesarios para {base_name}, se omite.")
        continue
    
    # Leer worldfile (.pgw)
    with open(worldfile, 'r') as f:
        A = float(f.readline())  # pixel size X
        D = float(f.readline())  # rotation Y (0)
        B = float(f.readline())  # rotation X (0)
        E = float(f.readline())  # pixel size Y (negativo)
        C = float(f.readline())  # top-left X
        F = float(f.readline())  # top-left Y

    # Leer imagen y JSON
    image = cv2.imread(image_file)
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Leer labels
    with open(label_file, 'r') as f:
        bboxes = f.readlines()


    palmera_bboxes = []
    ambiente_bboxes = []
    if os.path.exists(labels_palmera_file):
        with open(labels_palmera_file, 'r') as f:
            palmera_bboxes = f.readlines()
    if os.path.exists(labels_ambiente_file):
        with open(labels_ambiente_file, 'r') as f:
            ambiente_bboxes = f.readlines()

    index = 0
    while index < len(bboxes):
        temp_image = image.copy()
        draw_bbox(temp_image, bboxes[index])
        window_name = f'{base_name} - BBox {index+1}/{len(bboxes)}'
        cv2.imshow(window_name, temp_image)
        cv2.setMouseCallback(window_name, mouse_callback)
        key = cv2.waitKey(0)

        pending_update = False

        # Si se hace clic izquierdo → actualizar posición y mostrar bbox corregido
        if click_done:
            h, w = image.shape[:2]
            cx, cy = clicked_point
            class_id, _, _, bw, bh = map(float, bboxes[index].strip().split())
            x_c_norm = cx / w
            y_c_norm = cy / h
            new_line = f"{int(class_id)} {x_c_norm:.6f} {y_c_norm:.6f} {bw:.6f} {bh:.6f}\n"

            # Mostrar visualmente la corrección
            temp_image = image.copy()
            draw_bbox(temp_image, new_line, color=(0, 165, 255))
            cv2.imshow(window_name, temp_image)

            # Guardar la corrección en memoria, pero no se escribe en disco aún
            bboxes[index] = new_line
            json_data['palms'][index]['bbox_pixel']['x_center'] = int(cx)
            json_data['palms'][index]['bbox_pixel']['y_center'] = int(cy)

            # Recalcular coordenadas reales con .pgw
            x_real = A * cx + C
            y_real = E * cy + F
            json_data['palms'][index]['real_coordinates']['x'] = x_real
            json_data['palms'][index]['real_coordinates']['y'] = y_real

            print(f"BBox reposicionado → centro: ({cx},{cy}), reales: ({x_real:.2f}, {y_real:.2f})")
            click_done = False
            clicked_point = None
            pending_update = True  # hay bbox actualizado pendiente de confirmar con ENTER
        
        # Si se presiona ENTER, confirmar actualización
        elif key == 13:
            index += 1

        # Si se presiona Q, eliminar bbox y json entry
        elif key == ord('q'):
            print(f'Eliminado bbox {index+1} de {base_name}')
             # Eliminar del archivo principal
            del bboxes[index]

            # Eliminar del JSON
            if index < len(json_data['palms']):
                del json_data['palms'][index]

            # Eliminar del archivo de palmera
            if index < len(palmera_bboxes):
                del palmera_bboxes[index]

            # Eliminar del archivo de ambiente
            if index < len(ambiente_bboxes):
                del ambiente_bboxes[index]
            
        # ESC para salir
        elif key == 27:
            print("Saliendo.")
            cv2.destroyAllWindows()
            exit()

        # Guardar archivos modificados
        with open(label_file, 'w') as f:
            f.writelines(bboxes)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        if palmera_bboxes:
            with open(labels_palmera_file, 'w') as f:
                f.writelines(palmera_bboxes)

        if ambiente_bboxes:
            with open(labels_ambiente_file, 'w') as f:
                f.writelines(ambiente_bboxes)

        cv2.destroyAllWindows()