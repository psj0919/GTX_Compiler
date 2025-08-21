`graph → (exporter) → Python 파일 → 파일에서 로드한 단일 nn.Module(안에 모든 op가 submodules) 이 핵심이며, 이 파이프라인은 prepare_quantizable_module / recreate_gtx_module → module_template.py(per-op 모듈 생성) → ModuleHooker(연결/훅) 순으로 동작


전체 흐름은 모델(파이토치) → GTX 그래프(파싱/최적화/quant 준비) → (선택적으로) exporter로 Python 클래스 생성 → 다시 로드한 단일 nn.Module(quant_module)과 그래프를 서로 연결 → 그래프(또는 패턴 그래프)를 GtxCompiler가 XGraph/XIR로 변환 → xmodel 생성/컴파일 순으로 진행됩니다. GtxCompiler는 이 흐름에서 그래프를 실제 하드웨어용 XIR/XModel로 변환하는 최종 변환기 역할을 합니다.

단계별 상세 매핑

1. PyTorch 모델 실행/파싱
   - 파일/함수: 파서(parse.TorchParser)
   - 출력: GTX Graph (노드, 텐서, op.attrs, op.params 포함)

2. prepare_quantizable_module / recreate_gtx_module
    - 파일/함수: qproc.utils.prepare_quantizable_module / recreate_gtx_module
    - 동작:
        - 그래프를 기반으로 quantizable한 quant_module(하나의 nn.Module로 재구성) 생성.
        - (옵션) script-writer/exporter가 그래프를 Python 파일로 쓰고, 그 파일을 동적으로 로드하여 단일 클래스 인스턴스 생성 (recreate_gtx_module).
        - ModuleHooker.hook_module_with_node로 노드 ↔ 서브모듈 연결, 출력 훅 등록 등 수행.
        - 산출: (quant_module, graph) — 둘은 서로 연결되어 블롭(출력 값) 및 파라미터 동기화가 가능.

3. ModuleHooker와 데이터 동기화
    - 파일/함수: qproc.ModuleHooker
    - 동작:
        - 각 서브모듈에 .node 참조 부여
        - forward 훅으로 노드 출력/shape/데이터를 캡처
        - update_gtx_blob_data / update_parameters 등으로 graph의 텐서·파라미터 정보를 채움
    - 목적: GTX 그래프의 노드 정보(출력 shape, 값, 파라미터)를 런타임에서 보강 -> 이후 변환에 필요한 shape/const 확보

4. Graph 최적화 / DevGraph 변환
    - 파일/함수: gtx_shared.compile.DevGraphOptimizer 등
    - 동작: redundant op 제거, layout 변환, constant folding 등으로 배포용 dev_graph 생성

5. GTX 변환 (GtxCompiler)
    - 파일/함수: gtx_shared.compile.gtx_compiler.GtxCompiler.do_compile
    - 입력:
        - compile_graph (GTX Graph 또는 Pattern Graph) — 노드, out_tensors.shape, op.params(데이터) 필요
        - quant_config_info (옵션) — 노드/파라미터별 quant 정보(정수 xmodel 생성시)
    - 주요 동작:
        - graph의 파라미터(상수)를 XGraph에 fixed const로 복사(필요시 dtype/axis 조정 — 예: convtranspose flip)
        - 각 노드를 ISS_CONVERTOR에 등록된 변환기 함수로 XGraph 상의 iss_op로 변환
    - (옵션) xmodel 파일로 export (float 또는 int 접미)
        - 출력: XGraph 인스턴스 (그리고 파일 출력 시 xmodel)
        - 검증: verify_xmodel(shape 일치 검증), verify_gtx_graph(지원 여부 검사)

- 관계도(간단)
  - PyTorch model --(parse)--> GTX Graph
  - GTX Graph --(prepare_quantizable_module)--> quant_module + Graph (hooked)
  - quant_module (forward + hooks) → update blobs/shapes in Graph
  - Graph / dev_graph --(GtxCompiler.do_compile, quant_config_info)--> XGraph (xmodel)
  - XGraph --(xcompiler)--> 하드웨어 타깃용 compiled graph

핵심 의존성/전제조건 (GtxCompiler가 제대로 동작하려면)
- Graph의 각 노드에 대해 출력 shape가 정확히 존재해야 함(verify_xmodel에서 비교). shape 누락 시 변환 실패 또는 불일치.
- op.params에 실제 numpy 데이터가 있어야 함(상수를 XGraph에 넣기 위해). ModuleHooker.update_blobs_once 또는 update_gtx_blob_data로 채워야 함.
- 변환기(ISS_CONVERTOR)에 해당 op 타입의 변환 함수가 등록되어 있어야 함. 커스텀 op는 매핑을 추가해야 함.
- quantized xmodel을 원하면 quant_config_info(노드별 bn/fp 정보 등)를 정확히 전달해야 함.

실무적 주의점(문제 발생 시 확인 순서)
- Graph에 out_tensors.shape가 채워져 있는가? (없으면 Module을 forward 해 hook으로 채워야 함)
- op.params(가중치 등)가 Graph에 포함되어 있는가? (prepare / ModuleHooker로 동기화)
- 변환기 매핑(ISS_CONVERTOR)에 op 지원이 있는가? (커스텀 · 복합 op 점검)
- quant 모드 사용 시 quant_config_info의 포맷과 노드 키가 맞는가?
- 변환 중 예외(AddXopError) 발생 위치 — 에러 메시지에서 노드명/타입 확인

디버깅 팁
- 먼저 prepare_quantizable_module로 (quant_module, graph)를 얻고, ModuleHooker.update_blobs_once로 graph의 tensor 값/shape를 채운 뒤 GtxCompiler.do_compile(graph, output_file_name=None)을 호출해보면 변환 실패 지점을 빠르게 찾을 수 있습니다.
- verify_xmodel을 호출해 shape mismatch 로그를 확인.
- 작은 서브그래프(예: conv→bn→relu)로 먼저 end-to-end 실행해 성공 케이스를 만든 뒤 복잡 모델로 확장.

테스트/실험 제안
- 현재 assemble_example.py에 있는 fake pipeline 대신 실제 prepare_quantizable_module(혹은 스텁으로 만든 버전)으로 quant_module과 graph를 얻고, 그 graph를 GtxCompiler.do_compile에 넘겨보면 전체 흐름을 시뮬레이션할 수 있습니다. (테스트 스텁에서는 XGraph와 iss_op 변환자들이 어떻게 동작하는지 부분적으로만 확인됩니다.)

요약
- prepare_quantizable_module → module_template(per-op modules) → ModuleHooker는 GTX graph와 실행 가능한 파이토치 모듈을 동기화하여 graph가 변환(특히 상수/shape 정보) 가능한 상태가 되도록 준비합니다.
- GtxCompiler는 그 준비된 graph를 받아 XGraph/XIR로 변환하고 xmodel을 생성하는 단계입니다. 즉, prepare_quantizable_module이 주는 graph/quant 정보가 없으면 GtxCompiler는 필요한 데이터(파라미터/shape/quant config)를 얻지 못해 실패하거나 잘못된 xmodel을 만들 수 있습니다.