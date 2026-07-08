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
"""Conv/Dense/Add → Activation fuse 핸들러 (추론 최적화용 구조적 fusion).

Conv-BN fold 와 달리 ReLU 등 활성화는 **비선형**이라 conv 가중치에 수학적으로 흡수할 수
없다. 대신 추론 시 단일 fused 커널(Conv+ReLU)로 실행되는 것을 그래프에 표현한다:
producer 노드에 `fused_activation` 표식을 달고 activation 노드를 제거한다
(`base_graph.remove_node` 가 producer 출력을 activation 소비자로 rewire).
"""
from shared.base import OP

# 활성화를 흡수할 수 있는 선행(producer) op.
FUSE_ACT_PRODUCERS = [OP.CONV2D, OP.DEPTHWISE_CONV2D, OP.DENSE, OP.ADD]
# 흡수 대상 활성화 op (선형 fold 불가, 구조적으로만 fuse).
FUSE_ACTIVATIONS = [OP.RELU, OP.RELU6, OP.LEAKY_RELU, OP.SIGMOID, OP.GELU, OP.TANH, OP.CLAMP,
                    OP.SILU]


class ConvActivationHandler(object):
    """[producer, activation] 패턴을 fuse.

    producer 출력이 activation 의 **단독 소비자**일 때만 fuse한다 — 다분기(예: conv 출력이
    activation 과 다른 노드로 동시에 감)면 activation 제거 시 그 분기가 활성화를 잃으므로 skip.
    """

    def __call__(self, *args, **kwargs):
        _, node_set = args
        producer, actv = node_set[0], node_set[1]
        # producer 출력이 activation 단독 소비자인지(out_nodes 는 use 당 소비자명 리스트).
        if len(producer.out_nodes) != 1:
            return
        # activation 의 op 을 producer 에 보관 → codegen 이 type·attr(leaky slope/clamp min·max)로
        # 그대로 ggml activation 을 emit(relu 유실 없음). 시각화는 op.type 만 표시.
        producer.fused_activation = actv.op
        actv.merged = True
