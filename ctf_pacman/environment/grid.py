"""Procedural map generation with guaranteed connectivity and symmetry."""

from __future__ import annotations

from collections import deque
from typing import List, Set, Tuple

import numpy as np


# Action index -> (dx, dy) mapping
_ACTION_DELTAS = {
    0: (0, -1),   # North
    1: (0, 1),    # South
    2: (1, 0),    # East
    3: (-1, 0),   # West
    4: (0, 0),    # Stop
}

_NUM_ACTIONS = 5


class Grid:
    """Procedurally generated, horizontally symmetric CTF game map.

    The map is generated as follows:

    1. The entire border is marked as wall.
    2. A vertical divider at ``width // 2`` is added with three passage gaps.
    3. Interior walls are placed randomly with probability ``wall_density``
       while maintaining horizontal symmetry.
    4. Connectivity is verified via BFS; the map is regenerated if any
       non-wall cell is unreachable.

    Args:
        width:        Total number of columns.
        height:       Total number of rows.
        wall_density: Fraction of interior non-border, non-divider cells
                      that are walls.
        seed:         RNG seed for reproducibility.
    """

    def __init__(
        self,
        width: int,
        height: int,
        wall_density: float,
        seed: int,
    ) -> None:
        self.width = width
        self.height = height
        self.wall_density = wall_density
        self._rng = np.random.default_rng(seed)
        self.walls: np.ndarray = self._generate()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self) -> np.ndarray:
        """Generate a connected, symmetric map. Retries until connectivity holds."""
        for attempt in range(1000):
            walls = self._attempt_generate()
            if walls is not None:
                return walls
        # Fallback: open map (should never happen with reasonable wall_density)
        walls = np.zeros((self.height, self.width), dtype=bool)
        self._add_border(walls)
        self._add_divider(walls)
        return walls

    def _attempt_generate(self) -> np.ndarray | None:
        """One generation attempt; returns None if connectivity fails."""
        w, h = self.width, self.height
        walls = np.zeros((h, w), dtype=bool)

        # 1. Border
        self._add_border(walls)

        # 2. Divider with gaps
        self._add_divider(walls)

        # 3. Symmetric random interior walls
        mid = w // 2
        bound = mid - 1 if w % 2 == 0 else mid
        for y in range(1, h - 1):
            for x in range(1, bound):  # left half only
                if walls[y, x]:
                    continue
                if self._rng.random() < self.wall_density:
                    walls[y, x] = True
                    walls[y, w - 1 - x] = True  # mirror

        # 4. Connectivity check
        if not self._is_connected(walls):
            return None
        return walls

    def _add_border(self, walls: np.ndarray) -> None:
        walls[0, :] = True
        walls[self.height - 1, :] = True
        walls[:, 0] = True
        walls[:, self.width - 1] = True

    def _add_divider(self, walls: np.ndarray) -> None:
        w, h = self.width, self.height
        mid = w // 2
        # Place divider for entire column
        for y in range(h):
            walls[y, mid] = True
            if w % 2 == 0:
                walls[y, mid - 1] = True
        # Three passage gaps
        gap_rows = [h // 4, h // 2, 3 * h // 4]
        for gy in gap_rows:
            gy = max(1, min(h - 2, gy))
            walls[gy, mid] = False
            if w % 2 == 0:
                walls[gy, mid - 1] = False

    def _is_connected(self, walls: np.ndarray) -> bool:
        """BFS from the first non-wall cell to verify all non-wall cells are reachable."""
        h, w = walls.shape
        free: List[Tuple[int, int]] = [
            (x, y) for y in range(h) for x in range(w) if not walls[y, x]
        ]
        if not free:
            return False
        start = free[0]
        visited: Set[Tuple[int, int]] = set()
        queue: deque = deque([start])
        visited.add(start)
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < w and 0 <= ny < h and not walls[ny, nx]:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))
        return len(visited) == len(free)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_wall(self, x: int, y: int) -> bool:
        """Return True if cell (x, y) is a wall.

        Args:
            x: Column index.
            y: Row index.

        Returns:
            True if wall or out-of-bounds.
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return bool(self.walls[y, x])

    def get_legal_actions(self, x: int, y: int) -> List[int]:
        """Return action indices that do not move the agent into a wall.

        Args:
            x: Current column.
            y: Current row.

        Returns:
            List of legal action indices (always includes Stop=4).
        """
        legal = []
        for action, (dx, dy) in _ACTION_DELTAS.items():
            nx, ny = x + dx, y + dy
            if not self.is_wall(nx, ny):
                legal.append(action)
        return legal

    def apply_action(self, x: int, y: int, action: int) -> Tuple[int, int]:
        """Compute new position after applying *action* from (x, y).

        Illegal moves (into walls or out of bounds) are clamped to (x, y).

        Args:
            x:      Current column.
            y:      Current row.
            action: Action index (0–4).

        Returns:
            New (x, y) tuple.
        """
        dx, dy = _ACTION_DELTAS.get(action, (0, 0))
        nx = max(0, min(self.width - 1, x + dx))
        ny = max(0, min(self.height - 1, y + dy))
        if self.is_wall(nx, ny):
            return x, y
        return nx, ny

    def is_home_territory(self, x: int, team: int) -> bool:
        """Return True if column *x* belongs to *team*'s home territory.

        Team 0 owns ``x < width // 2``.
        Team 1 owns ``x >= width // 2``.

        Args:
            x:    Column index.
            team: Team ID (0 or 1).

        Returns:
            True if x is in the given team's home half.
        """
        mid = self.width // 2
        if team == 0:
            return x < mid
        return x >= mid

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the walls array.

        Returns:
            Boolean ndarray of shape (height, width).
        """
        return self.walls.copy()
