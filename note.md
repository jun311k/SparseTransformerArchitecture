# CIFAR-10

## 주요 파라미터
비록 직접적인 표 형태로는 나타나 있지 않지만, 논문 본문에 따르면 **OpenAI의 Sparse Transformer가 CIFAR-10 실험에서 사용한 주요 하이퍼파라미터**는 다음과 같습니다:


## 📊 CIFAR-10 실험에 사용된 파라미터 (논문에서 발췌)

| 항목               | 값                            | 설명                           |
| ---------------- | ---------------------------- | ---------------------------- |
| 입력 시퀀스 길이        | 3072                         | 32×32 픽셀 × 3채널 (RGB)         |
| 레이어 수            | **128**                      | 매우 깊은 모델                     |
| Attention Head 수 | **2개**                       | head 별로 stride 적용            |
| Embedding 차원 (d) | **256**                      | 각 토큰의 표현 차원                  |
| Feedforward 비율   | **half-size FFN**            | FFN 차원이 d보다 작음 (효율성)         |
| 드롭아웃 비율          | **0.25**                     | 과적합 방지                       |
| 학습률              | **0.00035**                  | 상대적으로 낮은 학습률                 |
| Optimizer        | Adam                         | 논문에 명시되어 있지 않지만 일반적으로 사용됨    |
| Epoch 수          | **120 epochs**               | CIFAR-10 전체 학습 반복 횟수         |
| Attention 종류     | **Strided Sparse Attention** | head마다 stride와 offset 다르게 설정 |


### 🧠 추가 설명

* **Strided Sparse Attention**은 head마다 stride/offset 조합이 다르게 구성되어 시퀀스 전체를 효율적으로 커버합니다.
* CIFAR-10은 이미지 크기가 작기 때문에(3072 길이), 이 모델은 비교적 긴 시퀀스 실험 중에는 “작은 편”에 속합니다.
* 128 레이어라는 깊이는 상당히 깊은 편이며, 실험을 통해 sparse 구조가 수렴 속도나 최종 loss 면에서 더 우수함을 보였습니다.



## 3072 토큰의 구성

🔢 3072 시퀀스 길이의 의미
CIFAR-10: 이미지 크기 = 32×32 = 1024 픽셀

각 픽셀당 RGB 3채널이 존재 → 총 1024 × 3 = 3072 값

이 값들은 모두 **8-bit 정수 (0~255)**로 표현되므로, 3072 바이트 시퀀스로 처리 가능합니다.🔢 3072 시퀀스 길이의 의미
CIFAR-10: 이미지 크기 = 32×32 = 1024 픽셀

각 픽셀당 RGB 3채널이 존재 → 총 1024 × 3 = 3072 값

이 값들은 모두 **8-bit 정수 (0~255)**로 표현되므로, 3072 바이트 시퀀스로 처리 가능합니다.


논문에 명확히 언급되어 있지는 않지만, 일반적으로는 다음 두 가지 방식이 가능합니다:

✅ 방식 1: [픽셀 기준 interleaved] → (가장 일반적 추정)
csharp
Copy
Edit
[pixel_0_R, pixel_0_G, pixel_0_B, pixel_1_R, pixel_1_G, pixel_1_B, ...]
즉, 픽셀 단위로 RGB를 순차적으로 나열

이 경우 공간 순서도 보존되고, Transformer가 지역 패턴을 더 잘 포착할 수 있음

🚫 방식 2: 채널별 나누어 나열 (덜 일반적)
css
Copy
Edit
[R_채널 1024개], [G_채널 1024개], [B_채널 1024개]
이 방식은 채널 간 구조를 잘 반영하지 못함 (Transformer는 상대적 위치가 중요하기 때문)

✅ 결론
각 픽셀마다 R, G, B 순으로 값을 넣고, 이를 전체 픽셀 순서에 따라 나열하는
즉,
[R₀, G₀, B₀, R₁, G₁, B₁, ..., R₁₀₂₃, G₁₀₂₃, B₁₀₂₃]
형태로 3072개의 토큰을 구성한 것으로 보는 것이 가장 합리적입니다.

## Stride 파라미터

| 시퀀스 길이 (n) | √n 값       | 보통 사용하는 window/stride 값 |
| ---------- | ---------- | ----------------------- |
| 1024       | 32         | 32                      |
| 2048       | 45.25      | 32 또는 64               |
| 3072       | 55.45      | 보통 64 또는 56 정도       |
| 4096       | 64         | 64                      |


##  Recommended command options
```bash
$ python sparse_transformer_mask.py --size 3072 --window_size 64 --stride 64 --full_png --skip_pattern_print
```
- token size: 3072
- local_attention: 64
- stride period: 64
- full size attention image
- skip pattern print