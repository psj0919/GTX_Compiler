"""_pass_changes diff 로직 자체검증 (torch 불요, 순수 함수).

그래프 최적화 Pass별 로그(PDF p5)의 핵심: 전후 스냅샷 diff 로 제거/타입변환 노드 추출.
    uv run python test/test_graph_opt_log.py
"""
from shared.compile.pipeline import _pass_changes


def test_removed_and_converted():
    before = {"conv1": "conv2d", "bn1": "batch_norm", "shp": "shape"}
    after = {"conv1": "conv2d", "shp": "const"}   # bn1 제거, shp 타입변환

    rows = _pass_changes(before, after, "fold_conv_bn")
    by_layer = {r[0]: r for r in rows}

    # 제거된 bn1 → pass 기본 action(FUSED)
    assert by_layer["bn1"][2] == "FUSED", by_layer["bn1"]
    # 타입변환된 shp → CONVERTED + old→new note
    assert by_layer["shp"][2] == "CONVERTED", by_layer["shp"]
    assert by_layer["shp"][3] == "shape → const", by_layer["shp"]
    # 변경 없는 conv1 은 빠진다
    assert "conv1" not in by_layer
    print("ok: removed=FUSED, converted=CONVERTED, unchanged 제외")


if __name__ == "__main__":
    test_removed_and_converted()
