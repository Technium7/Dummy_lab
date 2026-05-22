"""RRTConnect bidirectional planner (OMPL RRTConnect algorithm, Python implementation)."""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .scene import PlanningScene


def _wrap(q: Sequence[float]) -> np.ndarray:
    return np.asarray(q, dtype=np.float64)


def joint_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(_wrap(a) - _wrap(b)))


def steer(from_q: Sequence[float], to_q: Sequence[float], step_size: float) -> np.ndarray:
    a = _wrap(from_q)
    b = _wrap(to_q)
    delta = b - a
    dist = float(np.linalg.norm(delta))
    if dist <= step_size:
        return b.copy()
    return a + (step_size / dist) * delta


def interpolate_path_dense(path: List[np.ndarray], samples_per_segment: int = 10) -> np.ndarray:
    if len(path) == 1:
        return np.vstack([path[0]])
    rows: List[np.ndarray] = []
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=(i == len(path) - 2))
        for t in ts:
            rows.append(a + t * (b - a))
    return np.vstack(rows)


class _TreeNode:
    __slots__ = ("q", "parent")

    def __init__(self, q: np.ndarray, parent: Optional["_TreeNode"] = None) -> None:
        self.q = q
        self.parent = parent


class RRTConnectPlanner:
    """Bidirectional RRT-Connect (same connect logic as OMPL RRTConnect)."""

    def __init__(
        self,
        scene: PlanningScene,
        is_valid: Callable[[Sequence[float]], bool],
        step_size: float = 0.12,
        goal_bias: float = 0.08,
        goal_tolerance: float = 0.10,
        max_iterations: int = 8000,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.scene = scene
        self.is_valid = is_valid
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.goal_tolerance = goal_tolerance
        self.max_iterations = max_iterations
        self.rng = rng or random.Random()
        self._low = np.asarray(scene.lower_limits, dtype=np.float64)
        self._high = np.asarray(scene.upper_limits, dtype=np.float64)

    def _sample(self, goal: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.goal_bias:
            return goal.copy()
        return self._low + np.array([self.rng.random() for _ in range(len(self._low))]) * (self._high - self._low)

    def _nearest(self, nodes: List[_TreeNode], q: np.ndarray) -> _TreeNode:
        return min(nodes, key=lambda n: float(np.linalg.norm(n.q - q)))

    def _extend(self, tree: List[_TreeNode], q_target: np.ndarray) -> Tuple[Optional[_TreeNode], bool]:
        nearest = self._nearest(tree, q_target)
        q_new = steer(nearest.q, q_target, self.step_size)
        if float(np.linalg.norm(q_new - nearest.q)) < 1e-9 or not self.is_valid(q_new):
            return None, False
        node = _TreeNode(q_new, nearest)
        tree.append(node)
        reached = float(np.linalg.norm(q_new - q_target)) < 1e-6
        return node, reached

    def _connect(self, tree: List[_TreeNode], q_target: np.ndarray) -> Optional[_TreeNode]:
        node = self._nearest(tree, q_target)
        while True:
            q_new = steer(node.q, q_target, self.step_size)
            if float(np.linalg.norm(q_new - node.q)) < 1e-9 or not self.is_valid(q_new):
                break
            node = _TreeNode(q_new, node)
            tree.append(node)
            if float(np.linalg.norm(q_new - q_target)) < 1e-6:
                return node
        if joint_distance(node.q, q_target) <= self.goal_tolerance:
            return node
        return None

    @staticmethod
    def _path_to_root(node: _TreeNode) -> List[np.ndarray]:
        chain: List[np.ndarray] = []
        cur: Optional[_TreeNode] = node
        while cur is not None:
            chain.append(cur.q.copy())
            cur = cur.parent
        chain.reverse()
        return chain

    def plan(self, start: Sequence[float], goal: Sequence[float]) -> Optional[List[np.ndarray]]:
        start_q = _wrap(start)
        goal_q = _wrap(goal)
        if not self.is_valid(start_q) or not self.is_valid(goal_q):
            return None

        tree_a: List[_TreeNode] = [_TreeNode(start_q.copy())]
        tree_b: List[_TreeNode] = [_TreeNode(goal_q.copy())]
        goal_ref = goal_q

        for _ in range(self.max_iterations):
            if len(tree_a) > len(tree_b):
                tree_a, tree_b = tree_b, tree_a

            q_rand = self._sample(goal_ref)
            node_a, _ = self._extend(tree_a, q_rand)
            if node_a is None:
                tree_a, tree_b = tree_b, tree_a
                continue

            node_b = self._connect(tree_b, node_a.q)
            if node_b is not None and joint_distance(node_b.q, node_a.q) <= self.goal_tolerance:
                path_a = self._path_to_root(node_a)
                path_b = self._path_to_root(node_b)
                path_b.reverse()
                return path_a + path_b[1:]

            tree_a, tree_b = tree_b, tree_a

        return None
