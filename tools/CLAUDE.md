# 작업: GTX_Compiler의 DOT 출력이 Netron에서 제대로 렌더되도록 개선

## 배경
- GTX_Compiler(SuperGate)는 컴파일된 모델 그래프를 Graphviz .dot 로 내보내는 기능이 이미 있다.
- 이 .dot 를 Netron(네이티브 source/dot.js 리더)에서 열면 노드 op·attribute가 제대로 안 보인다.
- 현재 증상: 노드 type/name 이 op 가 아니라 무의미한 id(n0, n1)로 뜨고,
  op·하이퍼파라미터·shape 가 label="conv2d\n...\n[1,64,112,112]" 처럼 한 덩어리로만 들어가
  개별 attribute 로 분해되지 않는다.

## 확인된 렌더 규칙 (반드시 실제 소스로 재검증할 것)
- Netron 은 DOT 노드 id 를 type/name 으로, DOT 노드 attribute 를 클릭 가능한 metadata 필드로 쓴다.
- 따라서 label 한 덩어리에 몰아넣으면 op 가 살아나지 않는다.
- 먼저 https://github.com/lutzroeder/netron 의 source/dot.js 를 읽어,
  "노드 label/id 가 어떤 형식일 때 type 으로 승격되는지, attribute 가 어떻게 metadata 로
  매핑되는지"의 정확한 규칙을 코드로 확인한 뒤, 그 규칙에 맞춰 이미터를 고쳐라.
  (내 요약을 그대로 믿지 말고 소스로 검증)

## 목표
GTX_Compiler 가 내보내는 .dot 를 Netron 에서 열었을 때:
1) 각 노드의 type 이 op(CONV2D, RELU, ELEMWISE_ADD, MAXPOOL, ADAPTIVE_AVG_POOL2D, FLATTEN, DENSE 등)로 표시
2) in_channels/out_channels/kernel/stride/padding/out_shape 등 파라미터가 각각 별도 필드로 표시
3) 엣지(데이터 의존성)가 정확히 연결
되게 한다.

## 준수 규칙
- digraph(방향 그래프) + 엣지는 -> 로 유지.
- graphviz 예약 attribute(shape 는 노드 모양)를 텐서 shape 로 덮어쓰지 말 것 → out_shape 등 다른 키 사용.
- 노드 id 와 파라미터는 컴파일러 내부 op-IR(모듈 테이블: op + kwargs, forward 의 데이터 흐름)에서 그대로 가져와 1:1 매핑. 값을 새로 추정하지 말 것.
- 기존 백엔드(Python/C++/GGUF 이미터)와 다른 출력은 절대 건드리지 말 것. DOT 이미터만 수정.

## 작업 절차
1. 저장소에서 현재 DOT 를 생성하는 코드 위치를 찾아 구조를 요약(어떤 IR 을 도는지, 노드/엣지/라벨을 어떻게 뽑는지).
2. Netron source/dot.js 의 type/attribute 매핑 규칙을 읽고, 우리 출력이 왜 id 로 폴백되는지 원인을 특정.
3. 최소 변경으로 이미터를 수정: (a) 노드 id/type 이 op 로 뜨도록, (b) 파라미터를 개별 attribute 로.
4. ResNet 같은 샘플 모델로 .dot 를 재생성.
5. 변경 전/후 .dot 스니펫과, 무엇이 어떻게 바뀌는지 diff 로 보고.

## 완료 기준 (검증 가능해야 함)
- 생성된 .dot 를 Netron 에서 열면 CONV2D/RELU/ADD 등 op 가 노드 type 으로 보인다.
- Conv 노드 클릭 시 kernel/stride/padding/out_shape 가 개별 필드로 보인다.
- residual 분기(add 의 두 입력)와 downsample 경로가 그래프에 올바르게 연결된다.
- 기존 Python/C++/GGUF 출력은 변경되지 않는다(회귀 없음).

## 가정 보고
작업 시작 전, DOT 이미터 위치와 내부 op-IR 접근 방식에 대한 가정을 먼저 한 줄로 정리해 보고하고 진행하라.
