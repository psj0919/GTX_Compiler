현재 개발 중인 `GTX_Compiler`의 `pth -> gguf` 변환 파이프라인에 Netron 시각화 호환성을 위한 '텐서 네이밍 정규화(Tensor Renaming)' 기능을 추가하려고 해. 

모델의 가중치 데이터는 그대로 유지하되, GGUF 파일로 Write할 때 메타데이터와 텐서 이름만 Netron이 파싱할 수 있는 표준 규칙으로 매핑해 주는 것이 핵심 목표야. 아래 요구사항과 규칙을 반영해서 변환 모듈 코드를 업데이트해 줘.

### 1. 핵심 요구사항
* GGUF 변환 시, `general.architecture` 메타데이터 값을 기존 고유 아키텍처 이름을 쓰되 텐서 이름을 `blk.N.` 패턴으로 강제 변환해야 해.
* PyTorch(`pth`)의 원본 텐서 key 값을 GGUF에 쓸 때, 아래의 '네이밍 변환 규칙'을 거친 문자열로 치환하여 `GGUFWriter`에 전달해야 해.
* 데이터 복사나 구조 변경 없이, 단순히 딕셔너리 key(텐서 이름)를 정규식이나 문자열 조작으로 치환하는 매핑 레이어(Mapping Layer)만 추가할 것.

### 2. 텐서 네이밍 변환 규칙 (Netron `blk` 패턴)
Netron은 텐서 이름에 `blk.[숫자].`가 포함되어 있으면 이를 레이어 블록으로 자동 그룹화하여 시각화해. 따라서 기존 비전 모델이나 커스텀 모델의 블록 구조를 아래와 같이 매핑해 줘.

* **규칙 1 (Layer Blocks):** 중간의 반복되는 레이어 블록은 `blk.[블록인덱스].[연산자명].[가중치타입]` 형태로 치환해.
  * (예시) `image_encoder.blocks.0.attn.qkv.weight` -> `blk.0.attn_qkv.weight`
  * (예시) `image_encoder.blocks.5.mlp.fc2.bias` -> `blk.5.mlp_fc2.bias`
* **규칙 2 (Prefix/Suffix 정리):** 하위 경로에 온점(`.`)이 너무 많으면 Netron 파서가 오작동할 수 있으므로, 블록 인덱스 이후의 온점은 언더바(`_`)로 합쳐줘. 
* **규칙 3 (Input/Output 레이어):** 블록에 속하지 않는 임베딩이나 최종 출력 레이어는 직관적인 이름으로 바꿔줘.
  * (예시) `patch_embed.proj.weight` -> `token_embd.weight`
  * (예시) `neck.conv1.weight` -> `output_norm.weight` (또는 의미에 맞는 단일 레이어명)

### 3. 구현 지침 가이드
* Python의 `gguf` 라이브러리를 사용하는 변환 스크립트 내 함수에서 `blk.\1.` 형태로 치환하는 로직을 작성해 줘.
