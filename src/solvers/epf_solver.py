"""
Mixed Strategy Enforceable Payoff Frontier (EPF) Solver for Forest Collection Game

This implements proper EPF computation using mixed strategies with:
- Upper concave envelope operations (∨)
- Left-truncation operations (⊳)
- Convex combinations for mixed strategies
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from src.forest_game import ForestCollectionMDP, GameState, Action


@dataclass
class EPFPoint:
    """Represents a point on the EPF: (follower_payoff, leader_payoff)"""
    follower_payoff: float
    leader_payoff: float

    def __str__(self):
        return f"({self.follower_payoff:.2f}, {self.leader_payoff:.2f})"

    def __eq__(self, other):
        if not isinstance(other, EPFPoint):
            return False
        return (abs(self.follower_payoff - other.follower_payoff) < 1e-6 and
                abs(self.leader_payoff - other.leader_payoff) < 1e-6)

    def __hash__(self):
        return hash((round(self.follower_payoff, 6), round(self.leader_payoff, 6)))


class EPF:
    """
    Represents an Enforceable Payoff Frontier as a piecewise linear concave function
    Stored as a list of knots (breakpoints) sorted by follower_payoff
    """

    def __init__(self, points: List[EPFPoint]):
        self.points = self._remove_duplicates(points)
        self.points.sort(key=lambda p: p.follower_payoff)
        self._ensure_concavity()

    def _remove_duplicates(self, points: List[EPFPoint]) -> List[EPFPoint]:
        """Remove duplicate points"""
        seen = set()
        unique_points = []
        for point in points:
            if point not in seen:
                unique_points.append(point)
                seen.add(point)
        return unique_points

    def _ensure_concavity(self):
        """Ensure the EPF is concave by removing dominated points"""
        if len(self.points) <= 2:
            return

        # Remove points that are not on the upper concave envelope
        filtered_points = []
        for i, point in enumerate(self.points):
            is_dominated = False

            # Check if this point is dominated by any line segment between other points
            for j in range(len(self.points)):
                for k in range(j + 1, len(self.points)):
                    if j == i or k == i:
                        continue

                    p1, p2 = self.points[j], self.points[k]

                    # Skip if point is outside the x-range of the line segment
                    if not (min(p1.follower_payoff, p2.follower_payoff) <=
                           point.follower_payoff <=
                           max(p1.follower_payoff, p2.follower_payoff)):
                        continue

                    # Calculate y-value on line segment at point's x-coordinate
                    if abs(p2.follower_payoff - p1.follower_payoff) < 1e-9:
                        continue  # Vertical line, skip

                    t = ((point.follower_payoff - p1.follower_payoff) /
                         (p2.follower_payoff - p1.follower_payoff))
                    line_y = p1.leader_payoff + t * (p2.leader_payoff - p1.leader_payoff)

                    # If line segment gives higher leader payoff, this point is dominated
                    if line_y > point.leader_payoff + 1e-6:
                        is_dominated = True
                        break

                if is_dominated:
                    break

            if not is_dominated:
                filtered_points.append(point)

        self.points = filtered_points

    def evaluate(self, follower_payoff: float) -> float:
        """
        Evaluate the EPF at a given follower payoff value
        Returns the maximum leader payoff achievable for that follower payoff
        """
        if not self.points:
            return float('-inf')

        # Find the appropriate segment for interpolation
        if follower_payoff <= self.points[0].follower_payoff:
            return float('-inf')  # Outside domain

        if follower_payoff >= self.points[-1].follower_payoff:
            return float('-inf')  # Outside domain

        # Find the segment containing this follower payoff
        for i in range(len(self.points) - 1):
            p1, p2 = self.points[i], self.points[i + 1]

            if p1.follower_payoff <= follower_payoff <= p2.follower_payoff:
                # Linear interpolation
                if abs(p2.follower_payoff - p1.follower_payoff) < 1e-9:
                    return max(p1.leader_payoff, p2.leader_payoff)

                t = ((follower_payoff - p1.follower_payoff) /
                     (p2.follower_payoff - p1.follower_payoff))
                return p1.leader_payoff + t * (p2.leader_payoff - p1.leader_payoff)

        return float('-inf')

    def get_domain(self) -> Tuple[float, float]:
        """Get the domain of the EPF (min and max follower payoffs)"""
        if not self.points:
            return (0.0, 0.0)
        return (self.points[0].follower_payoff, self.points[-1].follower_payoff)

    def get_max_leader_payoff(self) -> float:
        """Get the maximum leader payoff achievable"""
        if not self.points:
            return 0.0
        return max(p.leader_payoff for p in self.points)

    def get_optimal_point(self) -> Optional[EPFPoint]:
        """Get the point that maximizes leader payoff"""
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.leader_payoff)

    def __str__(self):
        return f"EPF[{', '.join(str(p) for p in self.points)}]"


class MixedStrategyEPFSolver:
    """Computes Enforceable Payoff Frontiers using mixed strategies"""

    def __init__(self, mdp: ForestCollectionMDP, max_depth: int = 15):
        self.mdp = mdp
        self.max_depth = max_depth
        self.epf_cache: Dict[GameState, EPF] = {}
        self.grim_values: Dict[GameState, float] = {}
        self.altruistic_values: Dict[GameState, float] = {}

    def solve(self, initial_state: GameState) -> Tuple[EPF, Dict[GameState, EPF]]:
        """Compute EPF for the game starting from initial_state"""
        print("Computing grim and altruistic values...")
        self._compute_boundary_values(initial_state)

        print("Computing EPFs via backward induction with mixed strategies...")
        root_epf = self._compute_epf(initial_state, depth=0)

        print(f"Root EPF computed: {root_epf}")
        return root_epf, self.epf_cache.copy()

    def _compute_boundary_values(self, state: GameState):
        """Compute V(s) and V̄(s) for all reachable states"""
        visited = set()

        def compute_recursive(current_state: GameState, depth: int):
            if depth > self.max_depth or current_state in visited:
                return

            visited.add(current_state)

            if current_state.is_terminal():
                self.grim_values[current_state] = current_state.follower_total_fruit
                self.altruistic_values[current_state] = current_state.follower_total_fruit
                return

            # Recursively compute for children first
            children_states = []
            for action in Action:
                if self.mdp.is_valid_action(current_state, action):
                    next_state = self.mdp.transition(current_state, action)
                    compute_recursive(next_state, depth + 1)
                    children_states.append(next_state)

            if not children_states:
                self.grim_values[current_state] = current_state.follower_total_fruit
                self.altruistic_values[current_state] = current_state.follower_total_fruit
                return

            child_grim_values = [self.grim_values.get(child, 0) for child in children_states]
            child_altruistic_values = [self.altruistic_values.get(child, 0) for child in children_states]

            if current_state.is_leader_turn():
                # Leader chooses to minimize/maximize follower payoff
                self.grim_values[current_state] = min(child_grim_values)
                self.altruistic_values[current_state] = max(child_altruistic_values)
            else:
                # Follower chooses their best, but grim is leader's threat
                self.grim_values[current_state] = min(child_grim_values)
                self.altruistic_values[current_state] = max(child_altruistic_values)

        compute_recursive(state, 0)

    def _compute_epf(self, state: GameState, depth: int) -> EPF:
        """Compute EPF for a given state using backward induction"""
        if state in self.epf_cache:
            return self.epf_cache[state]

        if depth > self.max_depth or state.is_terminal():
            # Terminal state - EPF is just the actual payoff
            epf = EPF([EPFPoint(state.follower_total_fruit, state.leader_total_wood)])
            self.epf_cache[state] = epf
            return epf

        # Get valid actions and their resulting EPFs
        valid_actions = []
        child_states = []
        child_epfs = []

        for action in Action:
            if self.mdp.is_valid_action(state, action):
                next_state = self.mdp.transition(state, action)
                child_epf = self._compute_epf(next_state, depth + 1)

                valid_actions.append(action)
                child_states.append(next_state)
                child_epfs.append(child_epf)

        if not child_epfs:
            epf = EPF([EPFPoint(state.follower_total_fruit, state.leader_total_wood)])
            self.epf_cache[state] = epf
            return epf

        # Compute EPF based on whose turn it is
        if state.is_leader_turn():
            epf = self._compute_leader_epf_mixed(state, child_epfs)
        else:
            epf = self._compute_follower_epf_mixed(state, child_epfs, child_states)

        self.epf_cache[state] = epf
        return epf

    def _upper_concave_envelope(self, epfs: List[EPF]) -> EPF:
        """
        Compute the upper concave envelope of multiple EPFs
        This allows the leader to achieve any convex combination of child EPFs
        """
        if not epfs:
            return EPF([])

        if len(epfs) == 1:
            return epfs[0]

        # Collect all points from all EPFs
        all_points = []
        for epf in epfs:
            all_points.extend(epf.points)

        if not all_points:
            return EPF([])

        # Sort points by follower payoff
        all_points.sort(key=lambda p: p.follower_payoff)

        # Find the upper concave envelope
        envelope_points = []

        # Add leftmost point
        if all_points:
            envelope_points.append(all_points[0])

        for i in range(1, len(all_points)):
            current_point = all_points[i]

            # Remove points that are no longer on the envelope
            while (len(envelope_points) >= 2 and
                   self._is_below_line(envelope_points[-2], current_point, envelope_points[-1])):
                envelope_points.pop()

            # Add current point if it's not dominated
            if (not envelope_points or
                current_point.follower_payoff > envelope_points[-1].follower_payoff or
                current_point.leader_payoff > envelope_points[-1].leader_payoff):
                envelope_points.append(current_point)

        return EPF(envelope_points)

    def _is_below_line(self, p1: EPFPoint, p2: EPFPoint, test_point: EPFPoint) -> bool:
        """Check if test_point is below the line from p1 to p2"""
        if abs(p2.follower_payoff - p1.follower_payoff) < 1e-9:
            return False  # Vertical line

        # Calculate expected y-value on line at test_point's x-coordinate
        t = ((test_point.follower_payoff - p1.follower_payoff) /
             (p2.follower_payoff - p1.follower_payoff))
        expected_y = p1.leader_payoff + t * (p2.leader_payoff - p1.leader_payoff)

        return test_point.leader_payoff < expected_y - 1e-6

    def _left_truncate(self, epf: EPF, threshold: float) -> EPF:
        """
        Apply left-truncation: remove all points with follower_payoff < threshold
        This ensures incentive compatibility
        """
        truncated_points = []

        for i, point in enumerate(epf.points):
            if point.follower_payoff >= threshold - 1e-6:
                truncated_points.append(point)
            elif i < len(epf.points) - 1:
                # Check if we need to add an interpolated point at the threshold
                next_point = epf.points[i + 1]
                if next_point.follower_payoff > threshold:
                    # Interpolate to find the point exactly at the threshold
                    if abs(next_point.follower_payoff - point.follower_payoff) > 1e-9:
                        t = ((threshold - point.follower_payoff) /
                             (next_point.follower_payoff - point.follower_payoff))
                        interpolated_y = (point.leader_payoff +
                                        t * (next_point.leader_payoff - point.leader_payoff))
                        truncated_points.append(EPFPoint(threshold, interpolated_y))

        return EPF(truncated_points) if truncated_points else EPF([])

    def _compute_leader_epf_mixed(self, state: GameState, child_epfs: List[EPF]) -> EPF:
        """
        Compute EPF at leader's turn using mixed strategies
        Leader can achieve any point in the upper concave envelope of children EPFs
        """
        return self._upper_concave_envelope(child_epfs)

    def _compute_follower_epf_mixed(self, state: GameState, child_epfs: List[EPF],
                                   child_states: List[GameState]) -> EPF:
        """
        Compute EPF at follower's turn with incentive compatibility constraints
        """
        # Compute minimum required incentives for each child
        truncated_epfs = []

        for i, (child_epf, child_state) in enumerate(zip(child_epfs, child_states)):
            # Compute τ(s') = max of grim values from other children
            other_grim_values = []
            for j, other_child_state in enumerate(child_states):
                if i != j:
                    other_grim_values.append(self.grim_values.get(other_child_state, 0))

            if other_grim_values:
                threshold = max(other_grim_values)
            else:
                threshold = self.grim_values.get(child_state, 0)

            # Apply left-truncation to ensure incentive compatibility
            truncated_epf = self._left_truncate(child_epf, threshold)
            if truncated_epf.points:
                truncated_epfs.append(truncated_epf)

        # Take upper concave envelope of incentive-compatible EPFs
        if truncated_epfs:
            return self._upper_concave_envelope(truncated_epfs)
        else:
            # Fallback to current state payoffs
            return EPF([EPFPoint(state.follower_total_fruit, state.leader_total_wood)])


def solve_forest_epf_mixed(mdp: ForestCollectionMDP, max_depth: int = 15) -> Tuple[EPF, Dict]:
    """
    Convenience function to solve forest collection game and compute EPF with mixed strategies
    """
    solver = MixedStrategyEPFSolver(mdp, max_depth=max_depth)
    initial_state = mdp.get_initial_state()

    root_epf, all_epfs = solver.solve(initial_state)

    analysis = {
        "num_points": len(root_epf.points),
        "domain": root_epf.get_domain(),
        "max_leader_payoff": root_epf.get_max_leader_payoff(),
        "optimal_point": root_epf.get_optimal_point(),
        "points": root_epf.points
    }

    print(f"\nMixed Strategy EPF Analysis:")
    print(f"Number of EPF points: {analysis['num_points']}")
    print(f"Follower payoff domain: {analysis['domain']}")
    print(f"Max leader payoff: {analysis['max_leader_payoff']:.2f}")
    print(f"Optimal point for leader: {analysis['optimal_point']}")

    return root_epf, analysis


def solve_forest_epf(mdp: ForestCollectionMDP, max_depth: int = 15) -> Tuple[List[Tuple[float, float]], Dict]:
    """
    Alias for solve_forest_epf_mixed that returns points as list of tuples for backward compatibility
    """
    epf, analysis = solve_forest_epf_mixed(mdp, max_depth)
    
    # Convert EPF points to list of tuples for backward compatibility
    points_as_tuples = [(p.follower_payoff, p.leader_payoff) for p in epf.points]
    
    return points_as_tuples, analysis


# Example usage
if __name__ == "__main__":
    # Example with a simple 2x2 forest
    import numpy as np

    forest_map = np.array([
        [[5, 2], [2, 8]],
        [[8, 1], [3, 5]]
    ])