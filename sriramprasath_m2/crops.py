from ultralytics import YOLO
from PIL import Image
import os

model = YOLO("yolov8n.pt")
results = model.predict(source="bus.jpg", conf=0.5)
os.makedirs("crops", exist_ok=True)
image = Image.open("bus.jpg")
count = {}

for box in results[0].boxes:
    class_id = int(box.cls)
    class_name = model.names[class_id]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = image.crop((x1, y1, x2, y2))
    if class_name not in count:
        count[class_name] = 0
    filename = "crops/crop_" + class_name + "_" + str(count[class_name]) + ".jpg"
    crop.save(filename)
    count[class_name] += 1
    print("Saved: " + filename)

print("All crops saved!")