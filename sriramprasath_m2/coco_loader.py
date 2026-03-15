import json
import os

os.makedirs("coco_data", exist_ok=True)

annotations = {
    "images": [
        {"id": 1, "file_name": "bus.jpg", "width": 640, "height": 480}
    ],
    "annotations": [
        {"id": 1, "image_id": 1, "category_id": 6, "bbox": [100, 150, 200, 300]},
        {"id": 2, "image_id": 1, "category_id": 1, "bbox": [300, 100, 80, 200]},
        {"id": 3, "image_id": 1, "category_id": 1, "bbox": [400, 120, 70, 190]},
        {"id": 4, "image_id": 1, "category_id": 1, "bbox": [500, 130, 75, 180]}
    ],
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 6, "name": "bus"}
    ]
}

with open("coco_data/annotations.json", "w") as f:
    json.dump(annotations, f)

with open("coco_data/annotations.json", "r") as f:
    data = json.load(f)

print("COCO Annotation Loader!")
print("Total images: " + str(len(data["images"])))
print("Total annotations: " + str(len(data["annotations"])))
print("Categories:")
for cat in data["categories"]:
    print("  -> " + cat["name"])
print("Annotation loader done!")