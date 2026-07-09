# deprecated — 보관용 graveyard. nn/modules/__init__.py 에서 import 하지 않는다(inert).
#
# 여기의 render 는 카테고리 폴더로 분리·이관되었다:
#   head_render.py       → shape/render.py, indexing/render.py, activations/render.py(SILU),
#                          layers/render.py(DEPTHWISE_CONV2D), math/render.py(floor_div/remainder)
#   vision_ops_render.py → math/render.py, activations/render.py(ELU/SOFTPLUS/HSIGMOID/HSWISH/XIELU),
#                          layers/render.py(CONV3D/CONVTRANSPOSE1D·2D), pooling/render.py(MAX/AVG_POOL1D)
#
# 원본은 참조용으로만 남긴다. import 시 @register_render 가 재실행되어 중복 등록되므로
# 절대 활성 경로에서 import 하지 말 것.
