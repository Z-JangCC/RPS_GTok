"""Stateful interpreter for GPTok graph programs."""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from gptok2.data.schema import GraphRecord, graph_to_record
from gptok2.program.actions import Action, GraphProgram, parse_token
from gptok2.vq.codebook import PrototypeVQCodebook


@dataclass
class RuntimePort:
    node: int
    role: str
    mode: str
    capacity: int
    used: int = 0
    code_id: int = -1
    local_index: int = -1
    block_id: int = 0

    @property
    def open(self) -> bool:
        return self.used < self.capacity


@dataclass
class RuntimeAnchor:
    kind: str
    nodes: tuple[int, ...]
    role: str
    subtype: int
    code_id: int = -1
    local_index: int = -1


@dataclass
class InterpreterState:
    graph: nx.Graph = field(default_factory=nx.Graph)
    ports: list[RuntimePort] = field(default_factory=list)
    anchors: list[RuntimeAnchor] = field(default_factory=list)
    edge_anchors: list[RuntimeAnchor] = field(default_factory=list)
    current_ports: list[int] = field(default_factory=list)
    current_anchors: list[int] = field(default_factory=list)
    current_edge_anchors: list[int] = field(default_factory=list)
    current_base_node: int = 0
    current_num_nodes: int = 0
    current_code_id: int = -1
    block_stack: list[int] = field(default_factory=list)
    illegal_actions: int = 0
    runtime_errors: int = 0
    invalid_refs: int = 0
    type_mismatches: int = 0
    duplicate_edges: int = 0
    capacity_violations: int = 0
    executed_actions: int = 0
    graph_started: bool = False
    graph_ended: bool = False
    stopped: bool = False

    def recent_open_ports(self, window: int, exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        return [i for i in range(len(self.ports) - 1, -1, -1) if i not in exclude and self.ports[i].open][:window]

    def recent_ports(self, window: int, exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        return [i for i in range(len(self.ports) - 1, -1, -1) if i not in exclude][:window]

    def recent_anchors(self, window: int, kind: str = "node", exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        source = self.edge_anchors if kind == "edge" else self.anchors
        return [i for i in range(len(source) - 1, -1, -1) if i not in exclude][:window]


def execute_program(program: GraphProgram, codebook: PrototypeVQCodebook, config: dict | None = None) -> tuple[GraphRecord, InterpreterState]:
    cfg = (config or {}).get("program", {})
    state = InterpreterState()
    for action in program.actions:
        execute_action(state, action, codebook, int(cfg.get("ref_window", 16)), bool(cfg.get("close_cycle_consumes_all", False)))
    record = graph_to_record(state.graph, program.graph_id + "_recon", family="gptok_reconstruction")
    return record, state


def execute_action(state: InterpreterState, action: Action, codebook: PrototypeVQCodebook, ref_window: int = 16, close_cycle_consumes_all: bool = False) -> None:
    state.executed_actions += 1
    try:
        if state.stopped:
            state.illegal_actions += 1
            return
        if action.op == "BEGIN_GRAPH":
            if state.graph_started:
                state.illegal_actions += 1
                return
            state.graph_started = True
            return
        if action.op == "END_GRAPH":
            if not state.graph_started or state.graph_ended or state.block_stack:
                state.illegal_actions += 1
                return
            state.graph_ended = True
            return
        if action.op == "STOP":
            if not state.graph_ended:
                state.illegal_actions += 1
                return
            state.stopped = True
            return
        if action.op == "BEGIN_BLOCK":
            if not state.graph_started or state.graph_ended:
                state.illegal_actions += 1
                return
            state.block_stack.append(len(state.block_stack) + 1)
            return
        if action.op == "END_BLOCK":
            if not state.graph_started or state.graph_ended:
                state.illegal_actions += 1
                return
            if state.block_stack:
                state.block_stack.pop()
            else:
                state.illegal_actions += 1
            return
        if not state.graph_started or state.graph_ended:
            state.illegal_actions += 1
            return
        if action.op == "EMIT":
            _emit(state, codebook.get(int(action.args[0])))
            return
        if action.op == "INTERFACE":
            _apply_interface(state, codebook.get_interface(int(action.args[0])))
            return
        if action.op == "ATTACH":
            _attach(state, int(action.args[0]), int(action.args[1]), ref_window, close_cycle=False)
            return
        if action.op == "CLOSE_CYCLE":
            _attach(state, int(action.args[0]), int(action.args[1]), ref_window, close_cycle=close_cycle_consumes_all)
            return
        if action.op == "MERGE_NODE":
            _merge_node(state, int(action.args[0]), int(action.args[1]), ref_window)
            return
        if action.op == "MERGE_EDGE":
            _merge_edge(state, int(action.args[0]), int(action.args[1]), ref_window)
            return
        if action.op == "GLOBAL_LINK":
            _global_link(state, int(action.args[0]), int(action.args[1]))
            return
        state.illegal_actions += 1
    except (TypeError, ValueError, IndexError):
        state.runtime_errors += 1
        state.illegal_actions += 1


def legal_action_tokens(state: InterpreterState, codebook: PrototypeVQCodebook, config: dict | None = None) -> set[str]:
    cfg = (config or {}).get("program", {})
    w = int(cfg.get("ref_window", 16))
    if state.stopped:
        return set()
    if not state.graph_started:
        return {"BEGIN_GRAPH"}
    if state.graph_ended:
        return {"STOP"}
    tokens = {"END_GRAPH"}
    if cfg.get("begin_blocks", True):
        tokens.add("BEGIN_BLOCK")
    if state.block_stack:
        tokens.add("END_BLOCK")
    tokens.update(f"EMIT({c.code_id})" for c in codebook.codes)
    tokens.update(f"INTERFACE({s.interface_id})" for s in codebook.interfaces)
    current_ports = set(state.current_ports)
    current_anchors = set(state.current_anchors)
    current_edge_anchors = set(state.current_edge_anchors)
    for pi in range(len(state.current_ports)):
        for ri, _ in enumerate(state.recent_open_ports(w, exclude=current_ports), start=1):
            tokens.add(f"ATTACH({pi},{ri})")
            tokens.add(f"CLOSE_CYCLE({pi},{ri})")
    for ai in range(len(state.current_anchors)):
        for ri, _ in enumerate(state.recent_anchors(w, "node", exclude=current_anchors), start=1):
            tokens.add(f"MERGE_NODE({ai},{ri})")
    for ai in range(len(state.current_edge_anchors)):
        for ri, _ in enumerate(state.recent_anchors(w, "edge", exclude=current_edge_anchors), start=1):
            tokens.add(f"MERGE_EDGE({ai},{ri})")
    return tokens


def is_action_token_legal(state: InterpreterState, token: str, codebook: PrototypeVQCodebook, config: dict | None = None) -> bool:
    try:
        action = parse_token(token)
    except ValueError:
        return False
    return is_action_legal(state, action, codebook, config)


def is_action_legal(state: InterpreterState, action: Action, codebook: PrototypeVQCodebook, config: dict | None = None) -> bool:
    cfg = (config or {}).get("program", {})
    w = int(cfg.get("ref_window", 16))
    if state.stopped:
        return False
    if not state.graph_started:
        return action.op == "BEGIN_GRAPH"
    if state.graph_ended:
        return action.op == "STOP"
    if action.op == "BEGIN_GRAPH" or action.op == "STOP":
        return False
    if action.op == "END_GRAPH":
        return not state.block_stack
    if action.op == "BEGIN_BLOCK":
        return bool(cfg.get("begin_blocks", True))
    if action.op == "END_BLOCK":
        return bool(state.block_stack)
    if action.op == "EMIT":
        return len(action.args) == 1 and _as_int(action.args[0]) in {c.code_id for c in codebook.codes}
    if action.op == "INTERFACE":
        return len(action.args) == 1 and _as_int(action.args[0]) in {s.interface_id for s in codebook.interfaces}
    if action.op in {"ATTACH", "CLOSE_CYCLE"}:
        if len(action.args) != 2:
            return False
        port_idx, ref_idx = _as_int(action.args[0]), _as_int(action.args[1])
        refs = state.recent_open_ports(w, exclude=set(state.current_ports))
        return port_idx is not None and ref_idx is not None and 0 <= port_idx < len(state.current_ports) and 1 <= ref_idx <= len(refs)
    if action.op == "MERGE_NODE":
        if len(action.args) != 2:
            return False
        anchor_idx, ref_idx = _as_int(action.args[0]), _as_int(action.args[1])
        refs = state.recent_anchors(w, "node", exclude=set(state.current_anchors))
        return anchor_idx is not None and ref_idx is not None and 0 <= anchor_idx < len(state.current_anchors) and 1 <= ref_idx <= len(refs)
    if action.op == "MERGE_EDGE":
        if len(action.args) != 2:
            return False
        anchor_idx, ref_idx = _as_int(action.args[0]), _as_int(action.args[1])
        refs = state.recent_anchors(w, "edge", exclude=set(state.current_edge_anchors))
        return anchor_idx is not None and ref_idx is not None and 0 <= anchor_idx < len(state.current_edge_anchors) and 1 <= ref_idx <= len(refs)
    if action.op == "GLOBAL_LINK":
        if len(action.args) != 2:
            return False
        ref_a, ref_b = _as_int(action.args[0]), _as_int(action.args[1])
        refs = state.recent_ports(10**9)
        return ref_a is not None and ref_b is not None and ref_a != ref_b and 1 <= ref_a <= len(refs) and 1 <= ref_b <= len(refs)
    return False


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _emit(state: InterpreterState, code) -> None:
    base = state.graph.number_of_nodes()
    state.graph.add_nodes_from(range(base, base + code.prototype_num_nodes))
    for u, v in code.prototype_edges:
        state.graph.add_edge(base + int(u), base + int(v))
    state.current_ports = []
    state.current_anchors = []
    state.current_edge_anchors = []
    state.current_base_node = base
    state.current_num_nodes = int(code.prototype_num_nodes)
    state.current_code_id = int(code.code_id)
    block_id = state.block_stack[-1] if state.block_stack else 0
    for i, schema in enumerate(code.port_schema):
        if len(schema) == 5:
            anchor, role, mode, capacity, subtype = schema
        else:
            role, mode, capacity, subtype = schema
            anchor = i
        node = base + min(int(anchor), max(0, code.prototype_num_nodes - 1))
        port = RuntimePort(node, str(role), str(mode), int(capacity), 0, code.code_id, i, block_id)
        state.ports.append(port)
        state.current_ports.append(len(state.ports) - 1)
    for i, schema in enumerate(code.anchor_schema):
        kind, role, subtype, nodes_or_size = schema
        if isinstance(nodes_or_size, (list, tuple)):
            nodes = tuple(base + min(int(n), max(0, code.prototype_num_nodes - 1)) for n in nodes_or_size)
        else:
            nodes = tuple(base + min(i + j, max(0, code.prototype_num_nodes - 1)) for j in range(int(nodes_or_size)))
        state.anchors.append(RuntimeAnchor(str(kind), nodes, str(role), int(subtype), code.code_id, i))
        state.current_anchors.append(len(state.anchors) - 1)
    for i, schema in enumerate(code.edge_anchor_schema):
        kind, role, subtype, nodes_or_size = schema
        if isinstance(nodes_or_size, (list, tuple)):
            nodes = tuple(base + min(int(n), max(0, code.prototype_num_nodes - 1)) for n in nodes_or_size)
        elif i < len(code.prototype_edges):
            u, v = code.prototype_edges[i]
            nodes = (base + int(u), base + int(v))
        else:
            nodes = tuple(base + min(j, max(0, code.prototype_num_nodes - 1)) for j in range(int(nodes_or_size)))
        state.edge_anchors.append(RuntimeAnchor(str(kind), tuple(nodes), str(role), int(subtype), code.code_id, i))
        state.current_edge_anchors.append(len(state.edge_anchors) - 1)
    if not code.port_schema and not code.anchor_schema and not code.edge_anchor_schema:
        _add_default_shape_anchors(state, base, code)


def _add_default_shape_anchors(state: InterpreterState, base: int, code) -> None:
    for i in range(int(code.prototype_num_nodes)):
        state.anchors.append(RuntimeAnchor("node", (base + i,), "generic-port", 0, code.code_id, i))
        state.current_anchors.append(len(state.anchors) - 1)
    for i, (u, v) in enumerate(code.prototype_edges):
        state.edge_anchors.append(RuntimeAnchor("edge", (base + int(u), base + int(v)), "generic-port", 0, code.code_id, i))
        state.current_edge_anchors.append(len(state.edge_anchors) - 1)


def _apply_interface(state: InterpreterState, schema) -> None:
    state.current_ports = []
    state.current_anchors = []
    state.current_edge_anchors = []
    base = state.current_base_node
    num_nodes = max(1, state.current_num_nodes)
    code_id = state.current_code_id
    block_id = state.block_stack[-1] if state.block_stack else 0
    for i, row in enumerate(schema.port_schema):
        if len(row) == 5:
            anchor, role, mode, capacity, subtype = row
        else:
            role, mode, capacity, subtype = row
            anchor = i
        node = base + min(int(anchor), num_nodes - 1)
        state.ports.append(RuntimePort(node, str(role), str(mode), int(capacity), 0, code_id, i, block_id))
        state.current_ports.append(len(state.ports) - 1)
    for i, row in enumerate(schema.anchor_schema):
        kind, role, subtype, nodes_or_size = row
        if isinstance(nodes_or_size, (list, tuple)):
            nodes = tuple(base + min(int(n), num_nodes - 1) for n in nodes_or_size)
        else:
            nodes = tuple(base + min(i + j, num_nodes - 1) for j in range(int(nodes_or_size)))
        state.anchors.append(RuntimeAnchor(str(kind), nodes, str(role), int(subtype), code_id, i))
        state.current_anchors.append(len(state.anchors) - 1)
    for i, row in enumerate(schema.edge_anchor_schema):
        kind, role, subtype, nodes_or_size = row
        if isinstance(nodes_or_size, (list, tuple)):
            nodes = tuple(base + min(int(n), num_nodes - 1) for n in nodes_or_size)
        else:
            nodes = tuple(base + min(j, num_nodes - 1) for j in range(int(nodes_or_size)))
        state.edge_anchors.append(RuntimeAnchor(str(kind), tuple(nodes), str(role), int(subtype), code_id, i))
        state.current_edge_anchors.append(len(state.edge_anchors) - 1)


def _attach(state: InterpreterState, current_port_index: int, ref_index: int, ref_window: int, close_cycle: bool) -> None:
    if current_port_index >= len(state.current_ports):
        state.invalid_refs += 1
        return
    refs = state.recent_open_ports(ref_window, exclude=set(state.current_ports))
    if ref_index <= 0 or ref_index > len(refs):
        state.invalid_refs += 1
        return
    a = state.ports[state.current_ports[current_port_index]]
    b = state.ports[refs[ref_index - 1]]
    if not a.open or not b.open:
        state.capacity_violations += 1
        return
    if a.node == b.node:
        state.duplicate_edges += 1
        return
    if state.graph.has_edge(a.node, b.node):
        state.duplicate_edges += 1
    state.graph.add_edge(a.node, b.node)
    a.used += 1
    b.used += 1
    if close_cycle:
        a.used = a.capacity
        b.used = b.capacity


def _merge_node(state: InterpreterState, current_anchor_index: int, ref_index: int, ref_window: int) -> None:
    if current_anchor_index >= len(state.current_anchors):
        state.invalid_refs += 1
        return
    refs = state.recent_anchors(ref_window, "node", exclude=set(state.current_anchors))
    if ref_index <= 0 or ref_index > len(refs):
        state.invalid_refs += 1
        return
    a = state.anchors[state.current_anchors[current_anchor_index]]
    b = state.anchors[refs[ref_index - 1]]
    if a.role != b.role and "generic-port" not in {a.role, b.role}:
        state.type_mismatches += 1
    _contract_nodes(state, b.nodes[0], a.nodes[0])


def _merge_edge(state: InterpreterState, current_anchor_index: int, ref_index: int, ref_window: int) -> None:
    if current_anchor_index >= len(state.current_edge_anchors):
        state.invalid_refs += 1
        return
    refs = state.recent_anchors(ref_window, "edge", exclude=set(state.current_edge_anchors))
    if ref_index <= 0 or ref_index > len(refs):
        state.invalid_refs += 1
        return
    a = state.edge_anchors[state.current_edge_anchors[current_anchor_index]]
    b = state.edge_anchors[refs[ref_index - 1]]
    if len(a.nodes) < 2 or len(b.nodes) < 2:
        state.invalid_refs += 1
        return
    _contract_nodes(state, b.nodes[0], a.nodes[0])
    _contract_nodes(state, b.nodes[1], a.nodes[1])


def _global_link(state: InterpreterState, ref_a: int, ref_b: int) -> None:
    refs = state.recent_ports(10**9)
    if ref_a <= 0 or ref_b <= 0 or ref_a > len(refs) or ref_b > len(refs) or ref_a == ref_b:
        state.invalid_refs += 1
        return
    a = state.ports[refs[ref_a - 1]]
    b = state.ports[refs[ref_b - 1]]
    if a.node != b.node:
        if state.graph.has_edge(a.node, b.node):
            state.duplicate_edges += 1
        state.graph.add_edge(a.node, b.node)
        if a.open:
            a.used += 1
        if b.open:
            b.used += 1


def _contract_nodes(state: InterpreterState, keep: int, drop: int) -> None:
    if keep == drop or keep not in state.graph or drop not in state.graph:
        return
    for nb in list(state.graph.neighbors(drop)):
        if nb != keep:
            state.graph.add_edge(keep, nb)
    state.graph.remove_node(drop)
    mapping = {old: i for i, old in enumerate(sorted(state.graph.nodes()))}
    state.graph = nx.relabel_nodes(state.graph, mapping, copy=True)
    keep_new = mapping.get(keep, keep)
    for port in state.ports:
        port.node = keep_new if port.node in {keep, drop} else mapping.get(port.node, port.node)
    for anchor in state.anchors + state.edge_anchors:
        anchor.nodes = tuple(keep_new if n in {keep, drop} else mapping.get(n, n) for n in anchor.nodes)

