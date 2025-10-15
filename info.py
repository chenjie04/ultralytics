
from ultralytics import YOLO


model = YOLO("yolo115n.yaml")

print(model)

model.info(detailed=False)