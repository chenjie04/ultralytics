
from ultralytics import YOLO

model = YOLO("yolo113n.yaml")
print(model)

model.info(detailed=False)