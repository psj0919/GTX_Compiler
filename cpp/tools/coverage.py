#!/usr/bin/env python
"""C++ IR 의 op/shape 커버리지 리포트.

yolo 처럼 Python 이 OptPass 그래프 재작성을 적용하는 모델은 노드 정렬 기반 1:1 대조가
불가하다(노드 수·구조 상이). 대신 C++ 파서의 커버리지를 보고한다:
  - op_type 히스토그램(파싱 op vs UNPARSED placeholder)
  - 출력 텐서 shape 추론 비율

사용: python cpp/tools/coverage.py cpp/assets/yolo11n.ir.cpp.json
"""
import collections
import json
import sys


def main():
    if len(sys.argv) != 2:
        print("usage: coverage.py <ir.cpp.json>")
        sys.exit(2)
    d = json.load(open(sys.argv[1]))
    nodes = d["nodes"]
    hist = collections.Counter(n["op_type"] for n in nodes)
    unparsed = sum(v for k, v in hist.items() if k.startswith("UNPARSED"))
    shp_known = sum(1 for n in nodes for t in n["out_tensors"] if t["shape"] is not None)
    shp_tot = sum(1 for n in nodes for t in n["out_tensors"])

    print(f"model={d['model']}  nodes={len(nodes)}  input_shape={d['input_shape']}")
    print(f"파싱: {len(nodes) - unparsed}/{len(nodes)} op (UNPARSED {unparsed})")
    print(f"shape 추론: {shp_known}/{shp_tot} 출력텐서")
    print("op_type 히스토그램:")
    for k, v in sorted(hist.items(), key=lambda x: -x[1]):
        mark = "  ⚠" if k.startswith("UNPARSED") else ""
        print(f"  {v:4d}  {k}{mark}")


if __name__ == "__main__":
    main()
