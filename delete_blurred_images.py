import os
def delete_blurred_images_data(name_list, base_path):
    """
    Elimina imágenes, etiquetas, worldfiles y archivos auxiliares asociados a una lista de nombres base.
    
    :param name_list: Lista de nombres base (sin extensión) de los archivos a eliminar.
    :param base_path: Ruta raíz del proyecto donde están las carpetas de imágenes, labels, etc.
    """
    for name in name_list:
        file_paths = [
            os.path.join(base_path, 'images', f"{name}.png"),
            os.path.join(base_path, 'labels_binaria', f"{name}.txt"),
            os.path.join(base_path, 'labels_palmera', f"{name}.txt"),
            os.path.join(base_path, 'labels_ambiente', f"{name}.txt"),
            os.path.join(base_path, 'worldfiles', f"{name}.pgw"),
            os.path.join(base_path, 'json_annotations', f"{name}.json"),
            os.path.join(base_path, 'with_bbox', f"{name}.png"),
        ]

        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Eliminado: {file_path}")
                else:
                    print(f"No encontrado: {file_path}")
            except Exception as e:
                print(f"Error eliminando {file_path}: {e}")

to_delete = [
    "palmera_192157",
    "palmera_192557",
    "palmera_194957",
    "palmera_215357",
    "palmera_287357"
]

delete_blurred_images_data(to_delete, r'F:\Universidad\Curso 2024-25\Segundo Semestre\TFG\Desarrollo\dataset\fixed_normalization\Train')