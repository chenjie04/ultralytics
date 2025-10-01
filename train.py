from ultralytics import YOLO

# Load a model
model = YOLO("yolo113n.yaml")

# Train the model
train_results = model.train(
    # resume=True,
    data="VOC.yaml",  # path to dataset YAML
    # data="DUO.yaml",
    # data="Brackish.yaml", # 这个数据集需要将下面几个数据增强注释掉
    # data="TrashCAN_material.yaml",
    # data="coco.yaml",
    # data="coco8.yaml",
    epochs=50,  # number of training epochs
    batch=64,
    imgsz=640,  # training image size
    scale=0.5,  # N:0.5, S:0.9; M:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.0,  # N:0.0, S:0.05; M:0.15; L:0.15; X:0.2
    copy_paste=0.1,  # N:0.1, S:0.15; M:0.4; L:0.5; X:0.6
    device=[0, 1],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    cache="disk",
    project="runs/yolo113_VOC",
    name="n"
)