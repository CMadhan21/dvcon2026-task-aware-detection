from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    conf=0.5
)

print("Detection Complete!")
print("Objects detected: " + str(len(results[0].boxes)))

for box in results[0].boxes:
    class_id = int(box.cls)
    class_name = model.names[class_id]
    confidence = float(box.conf)
    print("Found: " + class_name + " confidence: " + str(round(confidence, 2)))