from ultralytics import YOLO
import clip
import torch
from PIL import Image

# Load models
clip_model, preprocess = clip.load("ViT-B/32")
yolo_model = YOLO("yolov8n.pt")

def run_pipeline(image_path, task):

    image = Image.open(image_path)

    # YOLO detection
    results = yolo_model(image)
    boxes = results[0].boxes.xyxy

    crops = []

    for box in boxes:
        x1,y1,x2,y2 = map(int, box)
        crop = image.crop((x1,y1,x2,y2))
        crops.append(crop)

    scores = []

    for crop in crops:

        img = preprocess(crop).unsqueeze(0)
        text = clip.tokenize([task])

        with torch.no_grad():
            img_feat = clip_model.encode_image(img)
            txt_feat = clip_model.encode_text(text)

            similarity = (img_feat @ txt_feat.T).item()

        scores.append(similarity)

    best_index = scores.index(max(scores))

    print("Best object index:", best_index)
