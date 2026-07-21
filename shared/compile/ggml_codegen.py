#
# Copyright 2025 Supergate.cc, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""vision.cpp(ggml) 백엔드 코드 생성기 (thin dispatcher).

각 op 의 렌더링은 `nn/modules/<op>.py` 의 ``render(node, ctx)`` 함수에 있고
``@register_render(OP.X)`` 로 등록된다 (shared/compile/render_api.py). 이 generator 는
Graph IR 를 순회하며 `RENDERERS` 로 디스패치하고, 구조적 op(INPUT/FLATTEN/RETURN)만
내장으로 처리한 뒤 visp arch 컨벤션의 `.cpp/.h/.weights.txt` 를 emit 한다.
설계·op매핑: docs/ggml_codegen.md
"""

import os

from shared.compile.render_api import RENDERERS, RenderContext, op_value, out_shape


class _FusedActNode:
    """ctx.attr(node, ...) 호환 래퍼 — fused activation op 의 attr(leaky slope/clamp min·max) 조회용."""

    def __init__(self, op):
        self.op = op


class VispCodeGenerator:
    def __init__(self, graph, output_dir, model_name="model", arch=None, quant_plan=None,
                 baked=None, skip=None):
        self.graph = graph
        self.output_dir = output_dir
        self.model_name = model_name
        self.arch = arch or model_name
        self.quant_plan = quant_plan or {}
        # const-fold 결과: baked={id(tensor):(gguf_key,arr)} 는 m.weights 로 로드,
        # skip={id(node)} 상수영역 노드는 emit 생략. (shared/compile/const_fold.py)
        self.baked = baked or {}
        self.skip = skip or set()

    # ------------------------------------------------------------ const baking
    def _bind_baked_inputs(self, node, ctx):
        """노드 입력 중 const-fold 로 baked 된 텐서를 m.weights("key") 로 로드해 바인딩."""
        for t in (node.in_tensors or []):
            if t is None:
                continue
            tid = id(t)
            if tid in self.baked and tid not in ctx._var:
                key, _ = self.baked[tid]
                var = ctx.new_var("cf")
                ctx.line(f'    tensor {var} = m.weights("{key}");')
                ctx._var[tid] = var

    # ----------------------------------------------------------------- driver
    def _walk(self, ctx):
        # op 렌더러 등록 트리거 (nn/modules/*.py 의 @register_render)
        import nn.modules  # noqa: F401

        # op 시퀀스 → 단일 fused ggml op 융합 (SwiGLU/RMS_NORM/FLASH_ATTN 등)
        from shared.compile.ggml_fusion import detect_fusions

        anchor_emit, skip = detect_fusions(self.graph)

        # NMS-free(v10) 후처리(top-k/gather/index `//`·`%`)는 ggml 그래프로 표현 불가
        # (정수 div/mod·값기반 top-k·gather 의미 mismatch) — CPU 후처리 영역(vision.cpp 도
        # NMS 를 host 에서 수행). 그래프에 그런 op 이 있으면 **자동으로** dense 예측 (1,C,A)
        # 까지만 출력한다(런타임 크래시 방지). top-k/NMS 는 harness 등 CPU 에서.
        # env GTX_DENSE_OUT=1 로 강제도 가능, GTX_DENSE_OUT=0 로 비활성도 가능.
        import os as _os
        _PP_OPS = {"max", "argmax", "topk", "aten::topk", "gather", "aten::gather", "index"}
        _env = _os.environ.get("GTX_DENSE_OUT")
        if _env == "0":
            dense_out = False
        elif _env:
            dense_out = True
        else:
            dense_out = any(op_value(n) in _PP_OPS for n in self.graph.nodes)
        if dense_out:
            print("[g2c] NMS-free postprocess detected → emitting dense output "
                  "(top-k/NMS is CPU-side)", flush=True)
        dense_var = None

        # 중간 텐서 탭(env GTX_DEBUG_TAPS="all" | "12,45,..."): 노드별 출력을
        # compute_graph_output 으로 추가 표시 → harness 가 dump, eager 백엔드와 노드별 대조.
        # tap 이름 tap{node_idx} = 생성 .py 의 module_{node_idx} 와 정렬(동일 IR 노드 순서).
        _taps = _os.environ.get("GTX_DEBUG_TAPS")
        _tap_set = (None if not _taps or _taps == "all"
                    else {int(i) for i in _taps.split(",") if i.strip().isdigit()})

        def _tap(idx, var):
            if not _taps or var in (None, "x"):
                return
            if _tap_set is not None and idx not in _tap_set:
                return
            ctx.line(f'    compute_graph_output(m, ggml_cont(m, {var}), "tap{idx}");')

        last_var = "x"
        for _idx, node in enumerate(self.graph.nodes):
            nid = id(node)
            if dense_out and dense_var is not None and op_value(node) in _PP_OPS:
                break  # 후처리 시작 → dense 까지만
            # --- const-fold: 상수영역 노드는 emit 생략(값은 baked weight 로 로드) ---
            if nid in self.skip:
                continue
            # --- 입력 중 baked 상수 텐서는 m.weights 로 lazy 로드 후 바인딩 ---
            self._bind_baked_inputs(node, ctx)
            # --- fused 패턴: 내부 노드는 스킵, anchor 에서 단일 ggml 호출 emit ---
            if nid in skip:
                continue
            if nid in anchor_emit:
                spec, roles = anchor_emit[nid]
                last_var = ctx.out(node, spec.emit(ctx, roles), hint=spec.name)
                _tap(_idx, last_var)
                continue

            ot = op_value(node)

            # --- 구조적 op (전용 모듈 파일 없음) ---
            if ot == "input":
                ctx.bind(node, "x")
                continue
            if ot == "return":
                continue
            if ot == "flatten":
                # flatten 은 정적 출력 shape 로 reshape 한다(start_dim 보존). 예:
                #   flatten(1): (B,C,H,W)→(B,C*H*W)   flatten(2): (B,C,H,W)→(B,C,H*W).
                # ggml_reshape 는 contiguous 요구 → cont 로 감싼다(torch reshape 와 동치).
                a = ctx.inp(node)
                sh = out_shape(node)
                if sh and len(sh) <= 4:
                    ne = list(reversed([int(d) for d in sh]))
                    args = ", ".join(str(d) for d in ne)
                    last_var = ctx.out(
                        node, f"ggml_reshape_{len(ne)}d(m, ggml_cont(m, {a}), {args})",
                        hint="flat",
                    )
                else:
                    # 폴백: 배치(ne[3]) 유지하고 W*H*C 를 dim0 으로 (start_dim=1 가정).
                    last_var = ctx.out(
                        node,
                        f"ggml_reshape_2d(m, {a}, "
                        f"{a}->ne[0] * {a}->ne[1] * {a}->ne[2], {a}->ne[3])",
                        hint="flat",
                    )
                _tap(_idx, last_var)
                continue

            render = RENDERERS.get(node.op.type) or RENDERERS.get(ot)
            if render is None:
                a = ctx.inp(node)
                var = ctx.new_var("op")
                ctx.line(
                    f"    tensor {var} = {a}; "
                    f"// TODO(ggml): unhandled op '{node.op.type}' ({node.name})"
                )
                ctx.bind(node, var)
                last_var = var
                continue

            res = render(node, ctx)
            # Conv+Activation fuse(--Opt≥1): producer 에 흡수된 활성화를 여기서 emit
            # (FuseConvActivation 이 relu 노드를 제거 → 마커로 재현, ggml 은 conv 다음 relu 라 동일).
            act_op = getattr(node, "fused_activation", None)
            if res is not None and act_op is not None:
                res = self._emit_fused_act(ctx, act_op, res)
                ctx.bind(node, res)      # 하류가 활성화 적용된 출력을 쓰게 재바인딩
            if res is not None:
                last_var = res
                _tap(_idx, res)
            # dense 후보: head 의 concat (1,C,A) (A=anchor 총수). 마지막 것이 최종 dense.
            if dense_out and res is not None and ot == "concat":
                sh = out_shape(node)
                if sh and len(sh) == 3 and sh[0] == 1 and sh[1] >= 5 and sh[2] >= 64:
                    dense_var = res
        if dense_out and dense_var is not None:
            last_var = dense_var
        return last_var

    # ------------------------------------------------------- fused activation
    def _emit_fused_act(self, ctx, act_op, var):
        """producer 에 흡수된 활성화 op 을 var 에 적용해 ggml 로 emit.

        nn/modules 의 활성화 render 와 **동일한 ggml 매핑**(relu→ggml_relu, relu6/clamp→
        ggml_clamp, leaky_relu→ggml_leaky_relu(slope) 등) → fused 여도 unfused 와 수치 동일.
        미지원 활성화는 그대로 통과(안전)."""
        t = str(getattr(act_op, "type", "")).lower()
        n = _FusedActNode(act_op)
        if t == "relu":
            expr = f"ggml_relu(m, {var})"
        elif t == "relu6":
            expr = f"ggml_clamp(m, {var}, 0.0f, 6.0f)"
        elif t == "sigmoid":
            expr = f"ggml_sigmoid(m, {var})"
        elif t == "gelu":
            expr = f"ggml_gelu(m, {var})"
        elif t == "tanh":
            expr = f"ggml_tanh(m, {var})"
        elif t in ("silu", "aten::silu_", "aten::silu"):
            # inplace config(dispatcher set) 로 커널 선택. fused 라 producer 출력의 단독
            # 소비자 → inplace 덮어쓰기 안전. config 없으면 aten 접미사 폴백.
            inplace = getattr(act_op, "inplace", None)
            if inplace is None:
                inplace = t.endswith("_")
            expr = f"ggml_silu_inplace(m, {var})" if inplace else f"ggml_silu(m, {var})"
        elif t == "leaky_relu":
            slope = ctx.attr(n, "negative_slope", 0.01)
            expr = f"ggml_leaky_relu(m, {var}, {float(slope)}f, false)"
        elif t == "clamp":
            lo = ctx.attr(n, "min", 0.0)
            hi = ctx.attr(n, "max", 6.0)
            expr = f"ggml_clamp(m, {var}, {float(lo)}f, {float(hi)}f)"
        else:
            return var
        v2 = ctx.new_var("act")
        ctx.line(f"    tensor {v2} = {expr};")
        return v2

    # --------------------------------------------------------------- emitters
    # 양자(quantized) conv 커널을 소비하는 헬퍼. vision.cpp 의 conv_2d 는 WHCN 에서
    # ggml_conv_2d_direct(F16 커널 전제)를 쓰므로 양자 커널을 못 받는다. conv = im2col +
    # mul_mat 이고 ggml_mul_mat 은 양자 가중치를 소비하므로, 커널을 2D [IC*KH*KW, OC] 로
    # 보고 직접 im2col→mul_mat 한다. 가중치는 GGUF 에 그 2D 양자 형태로 저장된다.
    # (※ vision.cpp 빌드로 수치 검증 필요 — 이 저장소엔 depend/ggml 미체크아웃)
    _CONV_2D_Q_HELPER = '''\
// --- quantized conv (conv = im2col + mul_mat; mul_mat consumes quantized weights) ---
// weight: quantized 2D [IC*KH*KW, OC]. x: WHCN [W,H,C,N]. kh,kw: kernel spatial size.
static tensor conv_2d_q(model_ref m, tensor x, int stride, int pad, int kh, int kw) {
    tensor w = m.weights("weight");                  // quantized 2D [IC*KH*KW, OC]
    int64_t ic = x->ne[2];                           // WHCN channels
    // im2col 은 spatial 차원(KW,KH,IC)을 위해 커널 텐서가 필요 — 양자 2D 엔 없으므로
    // 모양 전용 더미(데이터 미사용)로 전달.
    tensor kshape = ggml_new_tensor_4d(m, GGML_TYPE_F16, kw, kh, ic, 1);
    tensor cols = ggml_im2col(m, kshape, x, stride, stride, pad, pad, 1, 1, true, GGML_TYPE_F32);
    int64_t OW = cols->ne[1], OH = cols->ne[2], N = cols->ne[3];
    tensor cols2d = ggml_reshape_2d(m, cols, cols->ne[0], OW * OH * N);  // [K, OW*OH*N]
    tensor out = ggml_mul_mat(m, w, cols2d);         // [OC, OW*OH*N] (w=양자 → 첫 인자)
    out = ggml_reshape_4d(m, out, w->ne[1], OW, OH, N);     // [OC, OW, OH, N]
    out = ggml_cont(m, ggml_permute(m, out, 2, 0, 1, 3));  // → WHCN [OW, OH, OC, N]
    if (tensor bias = m.find("bias")) {
        bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
        out = ggml_add_inplace(m, out, bias);
    }
    return out;
}
'''

    # grid_sample 은 ggml 에 커널이 없다(임의 위치 보간). ggml_custom_4d 로 직접 구현해
    # 그래프에 넣는다 — DCN/deformable attention 공용. bilinear/nearest + zeros padding.
    _GRID_SAMPLE_HELPER = '''\
// --- grid_sample_2d: ggml 에 대응 커널이 없어 custom op 으로 구현 ---
// a: WHCN [W,H,C,N].  g: [2,GW,GH,N] (torch (N,GH,GW,2) 의 ggml 역순), 정규화 좌표 [-1,1].
// dst: [GW,GH,C,N].  flags: bit0=nearest, bit1=align_corners.  범위 밖은 0 (zeros padding).
#define GTX_GS_AT(t, i0, i1, i2, i3) \\
    (*(float const *)((char const *)(t)->data + (i0)*(t)->nb[0] + (i1)*(t)->nb[1] \\
                      + (i2)*(t)->nb[2] + (i3)*(t)->nb[3]))

static void grid_sample_2d_op(ggml_tensor * dst, int ith, int nth, void * userdata) {
    ggml_tensor const * a = dst->src[0];
    ggml_tensor const * g = dst->src[1];
    int const flags = (int)(intptr_t)userdata;
    bool const nearest = (flags & 1) != 0;
    bool const align   = (flags & 2) != 0;
    int64_t const W = a->ne[0], H = a->ne[1];
    int64_t const GW = dst->ne[0], GH = dst->ne[1], C = dst->ne[2], N = dst->ne[3];
    int64_t const rows = GH * C * N;                 // (gh, c, n) 행 단위로 스레드 분할
    for (int64_t r = ith; r < rows; r += nth) {
        int64_t const gh = r % GH, c = (r / GH) % C, n = r / (GH * C);
        for (int64_t gw = 0; gw < GW; ++gw) {
            float const * gp = (float const *)((char const *)g->data
                + gw * g->nb[1] + gh * g->nb[2] + n * g->nb[3]);
            float fx = gp[0], fy = gp[1];            // torch grid 마지막 축 = (x, y)
            fx = align ? (fx + 1.f) * (W - 1) * 0.5f : ((fx + 1.f) * W - 1.f) * 0.5f;
            fy = align ? (fy + 1.f) * (H - 1) * 0.5f : ((fy + 1.f) * H - 1.f) * 0.5f;
            float v = 0.f;
            if (nearest) {
                int64_t const xi = (int64_t)std::nearbyint(fx);
                int64_t const yi = (int64_t)std::nearbyint(fy);
                if (xi >= 0 && xi < W && yi >= 0 && yi < H) v = GTX_GS_AT(a, xi, yi, c, n);
            } else {
                int64_t const x0 = (int64_t)std::floor(fx), y0 = (int64_t)std::floor(fy);
                float const dx = fx - (float)x0, dy = fy - (float)y0;
                for (int64_t j = 0; j < 2; ++j) {
                    for (int64_t i = 0; i < 2; ++i) {
                        int64_t const xi = x0 + i, yi = y0 + j;
                        if (xi < 0 || xi >= W || yi < 0 || yi >= H) continue;
                        v += GTX_GS_AT(a, xi, yi, c, n)
                             * (i ? dx : 1.f - dx) * (j ? dy : 1.f - dy);
                    }
                }
            }
            *(float *)((char *)dst->data + gw * dst->nb[0] + gh * dst->nb[1]
                       + c * dst->nb[2] + n * dst->nb[3]) = v;
        }
    }
}

static tensor grid_sample_2d(model_ref m, tensor a, tensor g, int flags) {
    ggml_tensor * args[2] = { a, g };
    return ggml_custom_4d(m, GGML_TYPE_F32, g->ne[1], g->ne[2], a->ne[2], a->ne[3],
                          args, 2, grid_sample_2d_op, GGML_N_TASKS_MAX,
                          (void *)(intptr_t)flags);
}
'''

    # deformable conv (DCN v1/v2). ggml_conv_2d_deform 은 bias/dilation/groups 를 안 받으므로
    # bias 만 여기서 더한다(dilation/groups≠1 은 render 가 TODO 로 표시).
    _CONV_2D_DEFORM_HELPER = '''\
// --- deformable conv: ggml_conv_2d_deform + optional bias ---
// weight(GGUF, OIHW) → ggml ne [KW,KH,IC,OC].  x: WHCN.
// offset: [OW,OH,2*KH*KW,N],  mask: [OW,OH,KH*KW,N] (v1 이면 nullptr).
static tensor conv_2d_deform(model_ref m, tensor x, tensor offset, tensor mask,
                             int stride, int pad) {
    tensor out = ggml_conv_2d_deform(m, m.weights("weight"), x, offset, mask,
                                     stride, stride, pad, pad);
    if (tensor bias = m.find("bias")) {
        bias = ggml_reshape_4d(m, bias, 1, 1, bias->ne[0], 1);
        out = ggml_add_inplace(m, out, bias);
    }
    return out;
}
'''

    def _emit_quant_helpers(self, ctx):
        plan = getattr(ctx, "quant_plan", None) or {}
        ops = {op_value(n) for n in self.graph.nodes}
        out = ""
        if any(e.get("kind") == "conv" for e in plan.values()):
            out += "\n" + self._CONV_2D_Q_HELPER
        if "grid_sample" in ops:
            out += "\n" + self._GRID_SAMPLE_HELPER
        if "deform_conv2d" in ops:
            out += "\n" + self._CONV_2D_DEFORM_HELPER
        return out

    def _emit_source(self, ctx, last_var):
        body = "\n".join(ctx.lines)
        arch_id = self.arch.lower()  # GGUF general.architecture 규약(소문자)
        quant_helpers = self._emit_quant_helpers(ctx)
        # grid_sample custom 커널만 <cmath> 필요 — 안 쓰는 모델의 생성물은 그대로 둔다.
        extra_inc = ("#include <cmath>\n"
                     if any(op_value(n) == "grid_sample" for n in self.graph.nodes) else "")
        # RNN(LSTM/GRU)은 입력이 3D 시퀀스 [feat,seq,batch] 라 이미지용 cwhn↔whcn 레이아웃
        # 변환을 적용하면 안 된다 → RNN arch 에선 입력/출력 래핑을 생략하고 x 를 그대로 흘린다.
        is_rnn = any(op_value(n) in ("aten::lstm", "aten::gru")
                     for n in self.graph.nodes)
        if is_rnn:
            in_wrap = "    // RNN: 시퀀스 입력 [feat,seq,batch] — cwhn 변환 생략"
            out_wrap = f"    x = {last_var};"
        else:
            in_wrap = ("    // 입력 레이아웃 변환 (TODO: cwhn/whcn 자동 판별)\n"
                       "    x = cwhn_to_contiguous_2d(m, x);")
            out_wrap = f"    x = contiguous_2d_to_cwhn(m, {last_var});"
        return f"""// GENERATED BY SuperGate GTX Compiler (ggml/vision.cpp backend), DO NOT EDIT!
#include "visp/arch/{self.model_name}.h"
#include "visp/ml.h"
#include "visp/nn.h"
#include "visp/vision.h"
#include "util/string.h"

{extra_inc}#include <string_view>

namespace visp {{
{quant_helpers}
tensor {self.arch}_forward(model_ref m, tensor x, {self.arch}_params const& p) {{
    (void)p;
{in_wrap}

{body}

{out_wrap}
    return compute_graph_output(m, x, "result");
}}

{self.arch}_params {self.arch}_detect_params(model_file const& f) {{
    {self.arch}_params p{{}};
    // GGUF general.architecture 검증 (모델은 정적 unroll 이라 추가 하이퍼파라미터 없음).
    if (std::string_view arch = f.arch(); arch != "{arch_id}") {{
        throw except(
            "Architecture expected to be '{arch_id}', but was '{{}}' ({{}})",
            arch, f.path);
    }}
    return p;
}}

}} // namespace visp
"""

    def _emit_header(self):
        return f"""// GENERATED BY SuperGate GTX Compiler (ggml/vision.cpp backend), DO NOT EDIT!
#pragma once
#include "visp/ml.h"

namespace visp {{

struct {self.arch}_params {{
    // TODO(ggml): 모델 하이퍼파라미터 (block_count 등)
}};

tensor {self.arch}_forward(model_ref m, tensor x, {self.arch}_params const& p);
{self.arch}_params {self.arch}_detect_params(model_file const& f);

}} // namespace visp
"""

    def _emit_weight_manifest(self, ctx):
        lines = [
            '# GGUF weight manifest (key.suffix) — .cpp 의 m["key"] 와 일치해야 함',
            f"# arch={self.arch}",
        ]
        for key, suffixes in ctx._weights:
            for suf in suffixes:
                lines.append(f"{key}.{suf}")
        return "\n".join(lines) + "\n"

    def generate(self):
        os.makedirs(self.output_dir, exist_ok=True)
        ctx = RenderContext()
        ctx.quant_plan = self.quant_plan      # conv render 가 conv_2d_q emit 판단에 사용
        last_var = self._walk(ctx)

        src = self._emit_source(ctx, last_var)
        hdr = self._emit_header()
        man = self._emit_weight_manifest(ctx)

        src_path = os.path.join(self.output_dir, f"{self.model_name}.cpp")
        hdr_path = os.path.join(self.output_dir, f"{self.model_name}.h")
        man_path = os.path.join(self.output_dir, f"{self.model_name}.weights.txt")
        with open(src_path, "w") as f:
            f.write(src)
        with open(hdr_path, "w") as f:
            f.write(hdr)
        with open(man_path, "w") as f:
            f.write(man)

        return {
            "source": src_path,
            "header": hdr_path,
            "weights_manifest": man_path,
        }
