# Duck-Detection

오리 농장에서 오리를 검출하기 위한 object detection project입니다. **Detectron2**을 이용한 **RetinaNet**과 **YOLOX**를 사용했습니다.

## 본 프로젝트에 사용된 네트워크

### RetinaNet

- **[Duck-Detection에 적용한 코드](https://github.com/Koowater/Duck-Detection/tree/master/DD-Detectron2)**

- [original paper](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)

### YOLOX

- **[Duck-Detection에 적용한 코드](https://github.com/Koowater/Duck-Detection/tree/master/DD-YOLOX)**

- [original github repository](https://github.com/Megvii-BaseDetection/YOLOX)
## About the project

오리 농장의 오리 위치를 bounding box로 추론하는 task입니다.

- Input data : a single RGB image.
- Output data : Bounding boxes(confidence, class, x, y, width, height)

오리는 총 세 가지 상태(class)로 존재합니다.

- 살아있는 일반 오리(duck)
- 뒤집어진 오리(fallen)
- 죽은 오리(dead)

_**Model training과 evaluation은 [DD-Detectron2](https://github.com/Koowater/Duck-Detection/tree/master/DD-Detectron2), [DD-YOLOX](https://github.com/Koowater/Duck-Detection/tree/master/DD-YOLOX) 폴더를 참조해주세요.**_

## Evaluation

### 1. RetinaNet

- Training hyperparameter
    - From pretrained weights
    - batch size : 4
    - Max iteration : 3000
    - Base learning rate : 0.001
- epoch 1999
    
    ```bash
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.492
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.642
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.579
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.207
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.492
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.622
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.794
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.797
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.797
    
    [04/05 01:50:16 d2.evaluation.coco_evaluation]: Evaluation results for bbox: 
    |   AP   |  AP50  |  AP75  |  APs  |  APm   |  APl   |
    |:------:|:------:|:------:|:-----:|:------:|:------:|
    | 49.163 | 64.219 | 57.906 | 0.000 | 20.678 | 49.234 |
    [04/05 01:50:16 d2.evaluation.coco_evaluation]: Per-category bbox AP: 
    | category   | AP     | category   | AP    | category   | AP     |
    |:-----------|:-------|:-----------|:------|:-----------|:-------|
    | duck       | 61.687 | slapped    | 8.058 | dead       | 77.744 |
    ```
    

### 2. YOLOX-s

- Training hyperparameter
    - From pretrained weights
    - bach size : 8
    - max epoch : 50
    - no aug epochs : 15 (augmentation 없이 15 epoch까지 학습)
    - scheduler : yolox warmup cosine scheduler
    - weight decay = 5e-4
    - 자세한 사항은 [`DD-YOLOX/yolox/exp/my_yolox_base.py`](https://github.com/Koowater/Duck-Detection/blob/master/DD-YOLOX/yolox/exp/my_yolox_base.py)의 training config을 참고해주시기 바랍니다.
- best epoch
    
    ```bash
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.743
    Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.886
    Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.853
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
    Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.743
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.678
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.779
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.779
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
    Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.779
    per class AP:
    | class   | AP     | class   | AP     | class   | AP     |
    |:--------|:-------|:--------|:-------|:--------|:-------|
    | duck    | 51.824 | slapped | 90.000 | dead    | 80.971 |
    per class AR:
    | class   | AR     | class   | AR     | class   | AR     |
    |:--------|:-------|:--------|:-------|:--------|:-------|
    | duck    | 60.069 | slapped | 90.000 | dead    | 83.600 |
## Dataset labeling Rule

Dataset labeling 시, 다음과 같은 규칙을 정하였습니다.

- 오리의 몸체 모두가 드러나는 오리만을 labeling한다. 머리, 부리, 꼬리가 드러나는 경우, 대부분 이 조건을 만족합니다.
- 오리의 다리는 최소 하나는 드러나야 합니다.
- 너무 멀리 있어 작게 보이는 오리는 labeling 대상에서 제외하였다. 이미지 상에서 최대한 카메라 쪽에 가까운 오리들을 대상으로 labeling을 수행했습니다.

이러한 Rule을 정한 이유는, 이 task의 목적이 살아있는 오리를 모두 판별하는 것이 아닌 뒤집어졌거나(fallen) 죽은(dead) 오리를 탐지하기 위함이기 때문입니다.