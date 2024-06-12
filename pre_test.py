import cv2
import os

def crop_icon_region(image_path, output_path, x, y, width, height):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return
    roi = image[y:y+height, x:x+width]
    cv2.imwrite(output_path, roi)

x, y, = 920, 5
width, height = 80, 35

input_dirs = {
    'train': {
        'com_c4': 'imagens_originais/train/com_c4', 
        'sem_c4': 'imagens_originais/train/sem_c4'
    },
    'validation': {
        'com_c4': 'imagens_originais/validation/com_c4', 
        'sem_c4': 'imagens_originais/validation/sem_c4'
    }
}

output_base_dir = 'dataset'

for subset, classes in input_dirs.items():
    for label, input_dir in classes.items():
        output_dir = os.path.join(output_base_dir, subset, label)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not os.path.exists(input_dir):
            print(f"Diretório não encontrado: {input_dir}")
            continue

        for filename in os.listdir(input_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                crop_icon_region(input_path, output_path, x, y, width, height)