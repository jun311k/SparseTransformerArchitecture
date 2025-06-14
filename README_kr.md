# CIFAR-10을 위한 Sparse Transformer Attention Masks

**For English version, please refer to [README.md](README.md).**

이 프로젝트는 CIFAR-10 이미지 데이터(32x32 픽셀 = 1024 토큰)를 위한 sparse attention 패턴을 구현하고 시각화합니다. Sparse attention 패턴은 모델의 지역적 및 전역적 의존성 포착 능력을 유지하면서 계산 복잡도와 메모리 사용량을 크게 줄입니다.

## 스크립트 개요

이 프로젝트는 두 개의 주요 스크립트를 포함합니다:

1. `sparse_transformer_mask.py`: 다양한 어텐션 패턴을 구현하고 시각화
2. `mask_resource_calculator.py`: 마스크 연산의 리소스 요구사항 분석

### 1. Sparse Transformer Mask 생성기

`sparse_transformer_mask.py` 스크립트는 다양한 어텐션 패턴을 생성하고 시각화하는 데 중점을 둡니다.

```bash
python sparse_transformer_mask.py --help
usage: sparse_transformer_mask.py [-h] [--size SIZE] [--window_size WINDOW_SIZE] [--stride STRIDE] 
                                 [--sample_size SAMPLE_SIZE] [--full_png] [--no_graphic] 
                                 [--order {row_first,column_first}] [--skip_pattern_print]

Sparse Transformer Attention Mask Implementation

options:
  -h, --help            도움말 메시지를 표시하고 종료
  --size SIZE           마스크 행렬의 크기 (기본값: 1024)
  --window_size WINDOW_SIZE
                        로컬 어텐션 윈도우의 크기 (기본값: 32)
  --stride STRIDE       어텐션 포인트 간의 스트라이드 (기본값: 32)
  --sample_size SAMPLE_SIZE
                        시각화할 샘플의 크기 (기본값: 128)
  --full_png            샘플 대신 전체 크기 PNG 이미지 저장
  --no_graphic          그래픽을 건너뛰고 스파스 패턴만 출력
  --order {row_first,column_first}
                        마스크의 0이 아닌 요소의 순서 (기본값: row_first)
  --skip_pattern_print  마스크의 0이 아닌 요소 출력 건너뛰기
```

### 2. 마스크 리소스 계산기

`mask_resource_calculator.py` 스크립트는 마스크 연산의 리소스 요구사항을 분석합니다:
- 필요한 고유 행과 열의 수
- 연속된 연산 간의 리소스 변화
- 최대 리소스 사용 케이스

```bash
python mask_resource_calculator.py --help
usage: mask_resource_calculator.py [-h] [--mask_size MASK_SIZE] [--num_multiplications NUM_MULTIPLICATIONS]
                                   [--window_size WINDOW_SIZE] [--stride STRIDE] [--file] [--read_limit READ_LIMIT]

Calculate mask resources for sparse matrix multiplications.

options:
  -h, --help            도움말 메시지를 표시하고 종료
  --mask_size MASK_SIZE
                        마스크 행렬의 크기 (기본값: 1024)
  --num_multiplications NUM_MULTIPLICATIONS
                        동시 곱셈의 총 수 (기본값: 64)
  --window_size WINDOW_SIZE
                        strided/fixed 마스크의 로컬 어텐션 윈도우 크기 (기본값: 32)
  --stride STRIDE       strided 어텐션의 스트라이드 (기본값: 32)
  --file                결과를 파일로 출력
  --read_limit READ_LIMIT
                        파일에서 읽을 라인 수 제한 (기본값: 1000)
```

## 어텐션 패턴

다섯 가지 어텐션 패턴이 구현되어 있습니다:

### 1. 일반(Full) 어텐션

일반 어텐션 패턴은 표준 causal 어텐션을 구현합니다:
- **대각선 자기-어텐션(값 1)**: 각 토큰이 자기 자신에 attend
- **하삼각 어텐션(값 2)**: 각 토큰이 이전 모든 토큰에 attend

이 패턴의 희소성은 **49.6%** 입니다(상삼각 부분만 마스킹).

![일반 어텐션](images/normal_mask_128x128.png)

### 2. Strided 패턴

Strided 패턴은 세 가지 어텐션을 결합합니다:
- **대각선 자기-어텐션(값 1)**: 각 토큰이 자기 자신에 attend
- **로컬 윈도우 어텐션(값 2)**: 각 토큰이 윈도우 내 인접 토큰에 attend
- **Strided 어텐션(값 3)**: 각 토큰이 일정 간격의 이전 토큰에 attend

윈도우 크기와 스트라이드가 32일 때 약 **95.4% 희소성**을 가집니다.

![Strided 패턴](images/strided_mask_128x128.png)

### 3. Fixed 패턴

Fixed 패턴은 다음과 같습니다:
- **대각선 자기-어텐션(값 1)**: 각 토큰이 자기 자신에 attend
- **블록 단위 로컬 어텐션(값 2)**: 각 토큰이 같은 블록 내 이전 토큰에 attend
- **고정 컬럼 어텐션(값 3)**: 각 토큰이 각 이전 블록의 마지막 토큰에 attend

블록 크기 32일 때 약 **96.9% 희소성**을 가집니다.

![Fixed 패턴](images/fixed_mask_128x128.png)

### 4. Sliding Window 패턴

각 토큰이 고정 크기 윈도우 내 이웃 토큰에 attend (자기 자신 제외):
- **대각선 자기-어텐션(값 1)**
- **슬라이딩 윈도우 어텐션(값 2)**

윈도우 크기 32일 때 약 **96.8% 희소성**을 가집니다.

![Sliding Window 패턴](images/sliding_window_mask_128x128.png)

### 5. Dilated Sliding Window 패턴

각 토큰이 dilation 간격으로 떨어진 위치에 attend:
- **대각선 자기-어텐션(값 1)**
- **dilated sliding window 어텐션(값 2)**

윈도우 크기 32, dilation 2일 때 약 **96.8% 희소성**을 가집니다.

![Dilated Sliding Window 패턴](images/dilated_sliding_window_mask_128x128.png)

## 파일 출력

`mask_resource_calculator.py`를 `--file` 옵션과 함께 실행하면 `generated/` 디렉토리에 두 가지 유형의 파일이 생성됩니다:

1. 계산 포인트 파일 (`*_mask_{NUM_MULTIPLICATIONS}_read_limit_{READ_LIMIT}.txt`):
   - 각 시간 단계의 계산 포인트 포함
   - 마스크 생성에 사용된 파라미터 포함
   - 형식: `generated/{mask_type}_mask_{NUM_MULTIPLICATIONS}_read_limit_{READ_LIMIT}.txt`

2. 분석 파일 (`*_mask_{NUM_MULTIPLICATIONS}_read_limit_{READ_LIMIT}_analysis.txt`):
   - 계산 포인트에 대한 상세 분석 포함
   - 최대 케이스 정보와 변화 분석 포함
   - 형식: `generated/{mask_type}_mask_{NUM_MULTIPLICATIONS}_read_limit_{READ_LIMIT}_analysis.txt`

예시:
```bash
# 어텐션 패턴 생성 및 시각화
python sparse_transformer_mask.py --size 1024 --window_size 32

# 리소스 요구사항 분석
python mask_resource_calculator.py --file --num_multiplications 64 --read_limit 1000
```

## 컬러 매핑

시각화는 다음과 같은 custom colormap을 사용합니다:
- **회색 (값 0)**: 마스킹(No Attention)
- **진한 파랑 (값 1)**: 대각선/자기-어텐션
- **로얄 블루 (값 2)**: 로컬 어텐션(또는 normal 패턴의 하삼각)
- **스카이 블루 (값 3)**: Strided/고정 컬럼 어텐션

## 이론적 효율성

시퀀스 길이 1024(32x32 CIFAR-10 이미지) 기준:

- **표준 어텐션**: O(n²) = 1,048,576 연산
- **희소 어텐션**(95% sparsity): O((1-s)·n²) = 52,428 연산
- **이론적 속도 향상**: 약 20배

## 요구사항

- Python 3.6+
- NumPy
- Matplotlib

## 참고문헌

- [Sparse Transformer (Child et al., 2019)](https://arxiv.org/abs/1904.10509)
- [Longformer (Beltagy et al., 2020)](https://arxiv.org/abs/2004.05150)
- [BigBird (Zaheer et al., 2020)](https://arxiv.org/abs/2007.14062)
