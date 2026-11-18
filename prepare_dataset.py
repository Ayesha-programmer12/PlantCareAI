import os
import random
import shutil

SOURCE_DIR = r"D:\PlantCareAI\dataset\PlantVillage\PlantVillage"
OUTPUT_DIR = r"D:\PlantCareAI\dataset\prepared"

MAX_IMAGES_PER_CLASS = 150
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

random.seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)

total_images = 0

for category in os.listdir(SOURCE_DIR):

    category_path = os.path.join(SOURCE_DIR, category)

    if not os.path.isdir(category_path):
        continue

    images = [
        file for file in os.listdir(category_path)
        if file.lower().endswith(IMAGE_EXTENSIONS)
    ]

    random.shuffle(images)

    selected_images = images[:MAX_IMAGES_PER_CLASS]

    output_category = os.path.join(OUTPUT_DIR, category)
    os.makedirs(output_category, exist_ok=True)

    for image in selected_images:
        source = os.path.join(category_path, image)
        destination = os.path.join(output_category, image)

        shutil.copy2(source, destination)

    total_images += len(selected_images)

    print(f"{category}: {len(selected_images)} images prepared")

print("\nDataset preparation completed!")
print("Total selected images:", total_images)
print("Output location:", OUTPUT_DIR)