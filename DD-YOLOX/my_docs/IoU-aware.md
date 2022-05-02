# IoU-aware Single-stage Object Detector for Accurate Localization

https://arxiv.org/abs/1912.05992

# Abstract

### Single-stage object detector의 문제점

- low correlation between the classification score and localization accuracy in detection results severely hurts the average precision of the detection model.
    
    → classification score와 localization accuracy 사이의 낮은 상관관계가 detection model의 AP에 극도로 부정적인 영향을 준다.
    

이 문제를 해결하기 위해 IoU-aware single-stage object detector를 제안한다. IoU-aware single-stage object detector는 각각의 detected box의 IoU를 예측한다. 예측된 IoU는 classification score에 곱해져서 final detection confidence를 계산한다. 이는 localization accuracy에 더욱 관련성이 생긴다. detection confidence는 subsequent NMS와 COCO AP computation의 입력으로 사용된다. 

# Method

$$
\begin{equation} L_{IoU}=\frac1{N_{Pos}}\sum^N_{i∈Pos}BCE(IoU_i,\hat{IoU}_i) \end{equation}
$$

$IoU_i$: predicted IoU, $\hat{IoU_i}$: target IoU computed between $b_i$ and $\hat{b_i}$
$b_i$: positive example, $\hat{b_i}$: ground truth box

$$
\begin{equation} 
\hat{IoU}_i=overlap(b_i,\hat{b}_i)
\end{equation}
$$

$$
\begin{equation} \frac{∂BCE(IoU_i,\hat{IoU_i})}{∂\hat{IoU_i}}=log\frac{1-IoU_i}{IoU_i} \end{equation}
$$

$$
\begin{equation} L_{total} = L_{cls}+L_{loc}+L_{IoU} \end{equation}
$$

$$
\begin{equation} S_{det}=p^α_iIoU^{(1-α)}\end{equation}
$$

$p_i$: classification score, $α$: parameter to control the contribution the $p_i$ and predicted IoU [0, 1]
