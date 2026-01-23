import os
import shutil

source_root = "./data/Autism emotion recogition dataset/test"
output_dir = "./data/stylegan_images"

os.makedirs(output_dir, exist_ok=True)

for emotion in os.listdir(source_root):
    emotion_path = os.path.join(source_root, emotion)
    if not os.path.isdir(emotion_path):
        continue

    for img in os.listdir(emotion_path):
        if not img.lower().endswith((".jpg", ".jpeg")):
            continue

        name, ext = os.path.splitext(img)
        new_name = f"{name}__{emotion}{ext}"

        src = os.path.join(emotion_path, img)
        dst = os.path.join(output_dir, new_name)

        shutil.copy(src, dst)

print("Images copied and renamed successfully.")
