
import os
import re
import numpy as np
import imageio
from tqdm import tqdm
from pathlib import Path
from skimage.io import imsave, imread
import nibabel as nib

def dias_pre_processing(input_dir, output_dir):
    """Fusionne les images par minimum pixel à pixel."""
    pattern = re.compile(r'image_s(\d+)_i\d+\.png')
    images_dict = {}

    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(1))
            images_dict.setdefault(idx, []).append(os.path.join(input_dir, fname))

    os.makedirs(output_dir, exist_ok=True)

    for idx, file_list in images_dict.items():
        img_path = os.path.join(output_dir, f'image_{idx}.png')
        if os.path.exists(img_path):
            print(f"Image déjà traitée : {img_path}, on saute.")
            continue
        imgs = []
        shapes = []
        for f in file_list:
            arr = imread(f)
            imgs.append(arr)
            shapes.append((f, arr.shape))
        # On garde seulement les images de la même taille que la première
        ref_shape = imgs[0].shape
        imgs_same_shape = [img for img in imgs if img.shape == ref_shape]
        if len(imgs_same_shape) < len(imgs):
            for (fname, shape) in shapes:
                if shape != ref_shape:
                    print(f"Taille différente ignorée pour {fname} : {shape} au lieu de {ref_shape}")
        if len(imgs_same_shape) == 0:
            print(f"Aucune image valide pour le groupe {idx}, on saute.")
            continue
        img = np.minimum.reduce(imgs_same_shape)
        imsave(img_path, img)
        print(f'Saved {img_path}')


def hhv_pre_processing():
    input_dir = 'data/datasets/HHV/train/kidney_1_dense/'
    output_dir = 'data/3d'
    
    folders = ['images', 'labels']
     
    for folder in folders: 
        folder_path = Path(input_dir+folder)
        # Lister tous les fichiers .tif
        image_files = sorted([
            f for f in os.listdir(folder_path) if f.endswith('.tif')
        ])

        # Charger la première image pour obtenir les dimensions
        first_image_path = os.path.join(folder_path, image_files[0])
        first_image = imageio.imread(first_image_path)
        height, width = first_image.shape

        # Initialiser le volume 3D
        volume = np.zeros((len(image_files), height, width), dtype=first_image.dtype)

        # Charger toutes les images dans le volume
        print("Stacking images into volume...")
        for i, filename in enumerate(tqdm(image_files)):
            image_path = os.path.join(folder_path, filename)
            volume[i] = imageio.imread(image_path)

        # Sauvegarde du volume 3D au format .npy (rapide et efficace)
        np.save(output_dir+f'kidney_{folder}.npy', volume)
        print("Volume 3D sauvegardé sous 'kidney_volume.npy'")


def hhv_split(name):
    input_path = f'data/3d/kidney_{name}s.npy'
    output_dir = f'data/3d/{name}s'
    os.makedirs(output_dir, exist_ok=True)
    split_x = 2
    split_y = 2
    split_z = 2

    image = np.load(input_path)
    if image.ndim != 3:
        raise ValueError("L'image chargée n'est pas 3D.")

    Z, Y, X = image.shape

    # Coupes flexibles
    z_splits = np.array_split(np.arange(Z), split_z)
    y_splits = np.array_split(np.arange(Y), split_y)
    x_splits = np.array_split(np.arange(X), split_x)

    count = 1
    for z_idxs in z_splits:
        for y_idxs in y_splits:
            for x_idxs in x_splits:
                sub_volume = image[
                    z_idxs[0]:z_idxs[-1]+1,
                    y_idxs[0]:y_idxs[-1]+1,
                    x_idxs[0]:x_idxs[-1]+1
                ]
                output_path = os.path.join(output_dir, f"{name}_{count}.npy")
                np.save(output_path, sub_volume)
                count += 1

    print(f"{count - 1} sous-volumes sauvegardés dans : {output_dir}")


if __name__ == "__main__":
    # input_dir = 'data/datasets/DIAS/raw/full/images' 
    # output_dir = 'data/datasets/DIAS/clean/images'  
    # dias_pre_processing(input_dir, output_dir)
    
    # hhv_pre_processing()
    hhv_split("label")
