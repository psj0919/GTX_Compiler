#!/usr/bin/env python
"""Python TorchParser IR(JSON) vs C++ gtxc-parse IR(JSON) 1:1 대조.

저장된 trace 기반 C++ 경로는 활성 텐서의 value 이름(ret.1 / 3217 등)이 Python 저수준
트레이서와 다르다 → 이름은 무시하고 **구조·shape·config/attr/param** 으로 대조한다.
파라미터 텐서 이름(conv1.weight 등)은 state_dict 키에서 오므로 엄격 비교한다.

사용: python cpp/tools/compare_ir.py cpp/assets/resnet18.ir.py.json cpp/assets/resnet18.ir.cpp.json
"""
import json
import sys

FLOAT_EPS = 1e-6


def norm_val(v):
    """tensor ref 는 이름 제거 후 shape 만, float 는 라운딩."""
    if isinstance(v, dict):
        if "__tensor__" in v:
            return ("<T>", tuple(v.get("shape") or ()))
        # tensor desc {name,shape,dtype}
        if set(v.keys()) >= {"name", "shape"}:
            return ("<T>", tuple(v.get("shape") or ()), v.get("dtype"))
        return {k: norm_val(x) for k, x in v.items()}
    if isinstance(v, list):
        return [norm_val(x) for x in v]
    if isinstance(v, float):
        return round(v, 8)
    return v


def shapes(tensors):
    return [tuple(t["shape"]) if t["shape"] else None for t in tensors]


def param_sig(params, keep_name):
    out = {}
    for k, v in params.items():
        if isinstance(v, list):
            out[k] = [(t["name"] if keep_name else None, tuple(t["shape"] or ())) for t in v]
        else:
            out[k] = (v["name"] if keep_name else None, tuple(v["shape"] or ()))
    return out


def cfg_sig(d):
    return {k: norm_val(v) for k, v in d.items()}


def main():
    if len(sys.argv) != 3:
        print("usage: compare_ir.py <py.json> <cpp.json>")
        sys.exit(2)
    py = json.load(open(sys.argv[1]))
    cpp = json.load(open(sys.argv[2]))
    pn = sorted(py["nodes"], key=lambda n: n["idx"])
    cn = sorted(cpp["nodes"], key=lambda n: n["idx"])

    errors = []
    if len(pn) != len(cn):
        errors.append(f"node count: py={len(pn)} cpp={len(cn)}")

    name_match = 0
    pname_total = pname_match = 0
    n = min(len(pn), len(cn))
    for i in range(n):
        a, b = pn[i], cn[i]
        tag = f"[{i}] {a['op_type']}"
        if a["op_type"] != b["op_type"]:
            errors.append(f"{tag}: op_type py={a['op_type']} cpp={b['op_type']}")
            continue
        if a["name"] == b["name"]:
            name_match += 1
        # in/out tensor shapes
        if shapes(a["in_tensors"]) != shapes(b["in_tensors"]):
            errors.append(f"{tag}: in_tensor shapes py={shapes(a['in_tensors'])} cpp={shapes(b['in_tensors'])}")
        if shapes(a["out_tensors"]) != shapes(b["out_tensors"]):
            errors.append(f"{tag}: out_tensor shapes py={shapes(a['out_tensors'])} cpp={shapes(b['out_tensors'])}")
        # params: 이름+shape 엄격
        pa, pb = param_sig(a["params"], True), param_sig(b["params"], True)
        if pa != pb:
            # shape 만이라도 맞는지 구분
            if param_sig(a["params"], False) == param_sig(b["params"], False):
                errors.append(f"{tag}: param NAMES differ py={pa} cpp={pb}")
            else:
                errors.append(f"{tag}: params differ py={pa} cpp={pb}")
        for _, v in a["params"].items():
            vs = v if isinstance(v, list) else [v]
            for t in vs:
                pname_total += 1
        for (k, va), (kb, vb) in zip(a["params"].items(), b["params"].items()):
            vas = va if isinstance(va, list) else [va]
            vbs = vb if isinstance(vb, list) else [vb]
            for ta, tb in zip(vas, vbs):
                if ta["name"] == tb["name"]:
                    pname_match += 1
        # configs / attrs
        if cfg_sig(a["configs"]) != cfg_sig(b["configs"]):
            errors.append(f"{tag}: configs py={cfg_sig(a['configs'])} cpp={cfg_sig(b['configs'])}")
        if cfg_sig(a["attrs"]) != cfg_sig(b["attrs"]):
            errors.append(f"{tag}: attrs py={cfg_sig(a['attrs'])} cpp={cfg_sig(b['attrs'])}")

    print(f"nodes: py={len(pn)} cpp={len(cn)}")
    print(f"node-name exact match: {name_match}/{n} (활성텐서 이름차이는 정상)")
    print(f"param-name exact match: {pname_match}/{pname_total} (state_dict 키 — 일치해야 함)")
    if errors:
        print(f"\n❌ {len(errors)} 구조 불일치:")
        for e in errors[:60]:
            print("  -", e)
        sys.exit(1)
    print("\n✅ 구조·shape·config/attr/param 완전 일치 (이름 제외 1:1 parity)")


if __name__ == "__main__":
    main()
