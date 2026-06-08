from shared.graph import Graph
from typing import Mapping, List
from utils import TorchGraphSymbol, logging


def extract_scope_name(node_name: str) -> str:
    return node_name.rsplit(TorchGraphSymbol.NODE_NAME_SEPERATOR, 1)[0]


def extract_scope_name_node_name_map(graph: Graph) -> Mapping[str, List[str]]:
    ret: Mapping[str, str] = {}
    for node in graph.nodes:
        scope_name = extract_scope_name(node.name)
        if scope_name in ret:
            ret[scope_name].append(node.name)
        else:
            ret[scope_name] = [node.name]
    return ret

