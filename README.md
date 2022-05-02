# Duck-Detection

오리 농장에서 오리를 검출하기 위한 object detection project입니다. [Detectron2](https://github.com/facebookresearch/detectron2)을 이용한 [RetinaNet](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)과 [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)를 사용했습니다.

## Network that used for this project
---
### [RetinaNet](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)
### [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
## About the project
---
오리 농장의 오리 위치를 bounding box로 추론하는 task입니다.

- Input data : a single RGB image.
- Output data : Bounding boxes(confidence, class, x, y, width, height)

오리는 총 세 가지 상태(class)로 존재합니다.

- 살아있는 일반 오리(duck)
- 뒤집어진 오리(fallen)
- 죽은 오리(dead)

Model training과 evaluation은 [DD-Detectron2](), [DD-YOLOX]() 폴더를 참조해주세요.

## Dataset labeling Rule
---
Dataset labeling 시, 다음과 같은 규칙을 정하였습니다.

- 오리의 몸체 모두가 드러나는 오리만을 labeling한다. 머리, 부리, 꼬리가 드러나는 경우, 대부분 이 조건을 만족합니다.
- 오리의 다리는 최소 하나는 드러나야 합니다.
- 너무 멀리 있어 작게 보이는 오리는 labeling 대상에서 제외하였다. 이미지 상에서 최대한 카메라 쪽에 가까운 오리들을 대상으로 labeling을 수행했습니다.

이러한 Rule을 정한 이유는, 이 task의 목적이 살아있는 오리를 모두 판별하는 것이 아닌 뒤집어졌거나(fallen) 죽은(dead) 오리를 탐지하기 위함이기 때문입니다.