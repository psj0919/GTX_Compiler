"""ggml_profiler.py — GTX 런타임 op 프로파일러 (PDF p8–12).

`GgmlModule.__call__`(각 연산자의 forward dispatch 단일 지점)에 데코레이터를 씌워
op 별 **latency + 메모리(activation/weight/peak) + Time Share%** 를 계측한다.

켜기(오버헤드 때문에 옵션):
    GTX_PROFILE=1        python output/<Model>.py     # 계측 on
    GTX_PROFILE_REPS=20  ...                          # N회 반복(1회차 warmup 제외) avg/max/min
    GTX_PROFILE_OUT=dir  ...                          # 출력 디렉토리(기본: 모델 .py 옆)

출력(3-3): 콘솔 표 + CSV + JSON + Perfetto(Chrome trace) JSON. GTX 는 전면 NPU/fp16
구동이라 메모리는 원소수×2바이트(fp16)로 산출, Target=NPU(GTX).
"""
import atexit
import os
import sys
import time
# Rich 라이브러리로 콘솔에 컬러 출력 및 테이블 형식으로 프로파일 결과를 표시한다.
import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

_console = Console()

_FP16_BYTES = 2   # GTX 전면 fp16 → 텐서 메모리 = 원소수 × 2

_ENABLED = os.environ.get("GTX_PROFILE", "").lower() in ("1", "true", "on", "yes")
_REPS = max(1, int(os.environ.get("GTX_PROFILE_REPS", "1") or "1"))

_cur = []      # 진행 중 rep 의 op 기록 [{idx, op, ms, act, weight, out}]
_reps = []     # 완료된 rep 들: List[List[record]]
_order = 0


def enabled():
    return _ENABLED


def _arr(v):
    """wrapped/param → ndarray (없으면 None)."""
    if hasattr(v, "detach") and hasattr(v, "numpy"):
        v = v.detach()
    try:
        a = np.asarray(v)
    except Exception:
        return None
    return a if a.dtype != object and a.ndim >= 1 else None


def _nbytes(v):
    a = _arr(v)
    return int(a.size) * _FP16_BYTES if a is not None else 0


def _in_bytes(args, kwargs):
    """activation 입력 바이트(텐서 인자만; dim/end 등 스칼라 attr 는 제외)."""
    b = sum(_nbytes(a) for a in args)
    for k in ("input", "other"):
        if k in kwargs:
            b += _nbytes(kwargs[k])
    if isinstance(kwargs.get("tensors"), (list, tuple)):
        b += sum(_nbytes(t) for t in kwargs["tensors"])
    return b


def profile(call):
    """`GgmlModule.__call__` 데코레이터. GTX_PROFILE 미설정이면 원본 그대로(무오버헤드)."""
    if not _ENABLED:
        return call

    def wrapped(self, *args, **kwargs):
        global _order
        # rep 경계: 각 forward 는 INPUT op 으로 시작 → 새 rep 로 분리.
        if self.type == "INPUT" and _cur:
            _reps.append(_cur.copy())
            _cur.clear()
            _order = 0
        t0 = time.perf_counter()
        out = call(self, *args, **kwargs)
        ms = (time.perf_counter() - t0) * 1e3
        out_b = _nbytes(out)
        _cur.append({
            "idx": _order, "op": self.type, "ms": ms,
            "act": _in_bytes(args, kwargs) + out_b, "out": out_b,
            "weight": sum(_nbytes(v) for v in getattr(self, "weights", {}).values()),
        })
        _order += 1
        return out

    return wrapped


def _aggregate(reps):
    """rep 들(warmup 제외 후) → op 별 avg/max/min latency + 메모리. 순수 함수(테스트용).

    반환: (rows, totals). rows[i] = {idx, op, ms_avg/max/min, act, weight, out}.
    """
    if not reps:
        return [], {}
    use = reps[1:] if len(reps) > 1 else reps   # 1회차 warmup(로딩·캐시) 제외
    n = len(use[0])
    rows = []
    for i in range(n):
        lat = [r[i]["ms"] for r in use if i < len(r)]
        base = use[0][i]
        rows.append({
            "idx": i, "op": base["op"],
            "ms_avg": sum(lat) / len(lat), "ms_max": max(lat), "ms_min": min(lat),
            "act": base["act"], "out": base["out"], "weight": base["weight"],
        })
    tot_ms = sum(r["ms_avg"] for r in rows)
    weight_total = sum(r["weight"] for r in rows)
    peak = weight_total + (max((r["act"] for r in rows), default=0))  # 상주 weight + 최대 activation
    return rows, {"ms": tot_ms, "weight": weight_total, "peak": peak,
                  "reps": len(use), "warmup": len(reps) - len(use)}


def _kb(b):
    return f"{b / 1024:.1f}KB" if b < 1024 * 1024 else f"{b / (1024 * 1024):.2f}MB"


def _out_base():
    d = os.environ.get("GTX_PROFILE_OUT")
    argv0 = sys.argv[0] if sys.argv and sys.argv[0] else "gtx"
    stem = os.path.splitext(os.path.basename(argv0))[0] or "gtx"
    return os.path.join(d or os.path.dirname(os.path.abspath(argv0)) or ".", stem + ".profile")


def _share_cell(share):
    """Time Share% 를 색상 막대와 함께 표시 (병목 op 강조)."""
    color = "red" if share >= 25 else "yellow" if share >= 10 else "green"
    bars = int(round(share / 10.0))   # 10%당 블록 1개(최대 10)
    bar = "█" * bars
    return f"[{color}]{share:5.1f}%[/] [{color}]{bar}[/]"


def _print_console(rows, tot):
    _console.rule(f"[bold cyan]GTX RUNTIME PROFILE[/]  "
                  f"[dim](reps={tot['reps']}, warmup={tot['warmup']})[/]", align="left")
    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, header_style="bold")
    table.add_column("Idx", justify="right", style="dim")
    table.add_column("Op", style="cyan", no_wrap=True)
    table.add_column("Target")
    table.add_column("DType")
    table.add_column("Lat(ms)", justify="right")
    table.add_column("Share", justify="left", no_wrap=True)
    table.add_column("Activation", justify="right")
    table.add_column("Peak", justify="right")
    for r in rows:
        share = (100.0 * r["ms_avg"] / tot["ms"]) if tot["ms"] else 0.0
        peak = tot["weight"] + r["act"]
        table.add_row(
            str(r["idx"]), r["op"][:17], "[green]NPU(GTX)[/]", "[cyan]FP16[/]",
            f"{r['ms_avg']:.3f}", _share_cell(share), _kb(r["act"]), _kb(peak))
    _console.print(table)
    summary = Table(box=box.MINIMAL, show_header=False, show_edge=False, pad_edge=False)
    summary.add_column(style="bold")
    summary.add_column(justify="right")
    summary.add_row("TOTAL INFERENCE TIME", f"[bold]{tot['ms']:.3f} ms[/]  ({len(rows)} ops)")
    summary.add_row("WEIGHT MEMORY", _kb(tot["weight"]))
    summary.add_row("GLOBAL PEAK MEMORY", f"[bold]{_kb(tot['peak'])}[/]")
    _console.print(summary)
    _console.print("  [dim](Scratch: ggml 내부 mem-pool 비노출 → 생략. "
                   "Activation=입력+출력 fp16.)[/]")


def _write_csv(path, rows, tot):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "op", "target", "dtype", "lat_ms_avg", "lat_ms_max",
                    "lat_ms_min", "share_pct", "activation_bytes", "peak_bytes"])
        for r in rows:
            share = (100.0 * r["ms_avg"] / tot["ms"]) if tot["ms"] else 0.0
            w.writerow([r["idx"], r["op"], "NPU(GTX)", "FP16",
                        f"{r['ms_avg']:.6f}", f"{r['ms_max']:.6f}", f"{r['ms_min']:.6f}",
                        f"{share:.3f}", r["act"], tot["weight"] + r["act"]])


def _write_json(path, rows, tot):
    import json
    data = {"summary": {"total_ms": tot["ms"], "weight_bytes": tot["weight"],
                        "peak_bytes": tot["peak"], "reps": tot["reps"],
                        "warmup": tot["warmup"], "n_ops": len(rows)},
            "ops": [{"idx": r["idx"], "op": r["op"], "target": "NPU(GTX)", "dtype": "FP16",
                     "lat_ms": {"avg": r["ms_avg"], "max": r["ms_max"], "min": r["ms_min"]},
                     "share_pct": (100.0 * r["ms_avg"] / tot["ms"]) if tot["ms"] else 0.0,
                     "activation_bytes": r["act"], "weight_bytes": r["weight"],
                     "peak_bytes": tot["weight"] + r["act"]} for r in rows]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_perfetto(path, rows, tot):
    """Chrome trace 포맷(perfetto.dev): op Duration(ph=X) + Peak Memory Counter(ph=C)."""
    import json
    ev = [
        {"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "GTX NPU"}},
        {"name": "thread_name", "ph": "M", "pid": 0, "tid": 0, "args": {"name": "op"}},
    ]
    ts = 0.0   # µs, 누적
    for r in rows:
        dur = r["ms_avg"] * 1e3
        ev.append({"name": f"{r['op']}#{r['idx']}", "cat": "op", "ph": "X",
                   "ts": ts, "dur": dur, "pid": 0, "tid": 0,
                   "args": {"activation_bytes": r["act"]}})
        ev.append({"name": "Peak Memory", "cat": "mem", "ph": "C", "ts": ts, "pid": 0,
                   "args": {"Allocated Bytes": tot["weight"] + r["act"]}})
        ts += dur
    with open(path, "w") as f:
        json.dump({"traceEvents": ev, "displayTimeUnit": "ms"}, f, indent=2)


@atexit.register
def _dump():
    if not _ENABLED:
        return
    if _cur:                       # 마지막 rep flush
        _reps.append(_cur.copy())
    rows, tot = _aggregate(_reps)
    if not rows:
        return
    _print_console(rows, tot)
    base = _out_base()
    try:
        _write_csv(base + ".csv", rows, tot)
        _write_json(base + ".json", rows, tot)
        _write_perfetto(base + ".perfetto.json", rows, tot)
        print(f"  → 저장: {base}.csv / .json / .perfetto.json (perfetto.dev 로 타임라인)",
              flush=True)
    except OSError as e:
        print(f"  ⚠ 프로파일 파일 저장 실패: {e}", flush=True)


def demo():
    """_aggregate 자체검증: 2 rep(warmup 1 제외), op 별 avg/max/min."""
    reps = [
        [{"idx": 0, "op": "INPUT", "ms": 9.0, "act": 100, "out": 100, "weight": 0}],  # warmup
        [{"idx": 0, "op": "INPUT", "ms": 1.0, "act": 100, "out": 100, "weight": 0}],
        [{"idx": 0, "op": "INPUT", "ms": 3.0, "act": 100, "out": 100, "weight": 0}],
    ]
    rows, tot = _aggregate(reps)
    assert len(rows) == 1 and rows[0]["op"] == "INPUT"
    assert rows[0]["ms_avg"] == 2.0 and rows[0]["ms_max"] == 3.0 and rows[0]["ms_min"] == 1.0
    assert tot["warmup"] == 1 and tot["reps"] == 2
    print("ok: warmup 제외 + avg/max/min 집계")


if __name__ == "__main__":
    demo()
