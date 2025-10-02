from ultralytics import YOLO

# Load a model
<<<<<<< HEAD
<<<<<<< HEAD
model = YOLO("yolo115n.yaml")
# model = YOLO("runs/yolo114_VOC/n2/weights/last.pt")
=======
model = YOLO("yolo113n.yaml")
>>>>>>> df3b7fa0f (开发DA3NetV2)
=======
# model = YOLO("yolo114n.yaml")
model = YOLO("runs/yolo114_VOC/n2/weights/last.pt")
>>>>>>> 52fe3ad13 (测试特征融合模块)

# Train the model
train_results = model.train(
    resume=True,
    data="VOC.yaml",  # path to dataset YAML
    # data="DUO.yaml",
    # data="Brackish.yaml", # 这个数据集需要将下面几个数据增强注释掉
    # data="TrashCAN_material.yaml",
    # data="coco.yaml",
    # data="coco8.yaml",
<<<<<<< HEAD
<<<<<<< HEAD
    # data="TT100K-2016.yaml",
    epochs=500,  # number of training epochs
=======
    epochs=50,  # number of training epochs
>>>>>>> df3b7fa0f (开发DA3NetV2)
=======
    epochs=500,  # number of training epochs
>>>>>>> 52fe3ad13 (测试特征融合模块)
    batch=64,
    imgsz=640,  # training image size
    scale=0.5,  # N:0.5, S:0.9; M:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.0,  # N:0.0, S:0.05; M:0.15; L:0.15; X:0.2
    copy_paste=0.1,  # N:0.1, S:0.15; M:0.4; L:0.5; X:0.6
    device=[0, 1],  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    cache="disk",
<<<<<<< HEAD
<<<<<<< HEAD
    project="runs/yolo115_VOC",
    name="n_sa_pe"
=======
    project="runs/yolo113_VOC",
=======
    project="runs/yolo114_VOC",
>>>>>>> 52fe3ad13 (测试特征融合模块)
    name="n"
>>>>>>> df3b7fa0f (开发DA3NetV2)
)