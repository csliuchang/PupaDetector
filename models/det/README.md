# RotateDetection 


## Performance

### MSRA-TD500
| Model |    Backbone    |    precision    |    recall    |    mAP  | GPU | Image/GPU | FPS | Loss| lr schd | Data Augmentation | Configs |       
|:------------:|:------------:|:------------:|:---------:|:-----------:|:----------:|:-----------:|:---------:|:---------:|:---------:|:---------:|:---------:|     
| [RRetinaNet](https://arxiv.org/pdf/1707.06484.pdf)| yolov5s | 0.6647 | 0.7100 | 0.4956 | **1X** GeForce RTX 1660 Ti | 2 | 52 | GWD loss | 5e-5 1x | No | [dla_resnet18.json](./configs/rretinanet/models/rretinanet_yolov5_backbone.json) |