"""
Enforceable Payoff Frontier (EPF) Solver for Forest Collection Game

This implements the SEFCE algorithm using EPFs as described in the Function Approximation paper.
Key features:
1. Computes piecewise linear concave EPFs for each game state
2. Handles mixed strategies (at most 2 children per node)
3. Implements left truncation for incentive compatibility
4. Supports both pure and mixed commitment strategies
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from collections import defaultdict

# Import the existing forest game structure
from src.forest_game import ForestCollectionMDP, GameState, Action


@dataclass
class EPFPoint:
    """Point on the Enforceable Payoff Frontier: (μ2, U1(μ2))"""
    follower_payoff: float  # μ2 - follower's payoff
    leader_payoff: float  # U1(μ2) - leader's payoff for this follower payoff

    def __repr__(self):
        return f"({self.follower_payoff:.2f}, {self.leader_payoff:.2f})"

    def __lt__(self, other):
        return self.follower_payoff < other.follower_payoff


class EPF:
    """Enforceable Payoff Frontier - piecewise linear concave function"""

    def __init__(self, points: List[EPFPoint] = None):
        self.points = sorted(points or [], key=lambda p: p.follower_payoff)
        self.left_truncate_threshold = float('-inf')

    def add_point(self, follower_payoff: float, leader_payoff: float):
        """Add a point to the EPF"""
        self.points.append(EPFPoint(follower_payoff, leader_payoff))
        self.points.sort()

    def set_left_truncate(self, threshold: float):
        """Set the left truncation threshold τ(s')"""
        self.left_truncate_threshold = threshold

    def truncate(self) -> 'EPF':
        """
        Truncate EPF at the left truncation threshold.

        Removes all points to the left of threshold except the rightmost one,
        then interpolates a new point exactly at the threshold.
        """
        if not self.points or self.left_truncate_threshold == float('-inf'):
            return EPF(self.points.copy())

        threshold = self.left_truncate_threshold

        # Sort points by follower payoff
        sorted_points = sorted(self.points, key=lambda p: p.follower_payoff)

        # Find points to the left and right of threshold
        left_points = [p for p in sorted_points if p.follower_payoff < threshold]
        right_points = [p for p in sorted_points if p.follower_payoff >= threshold]

        truncated_points = []

        # Keep the rightmost point from the left side (if any)
        if left_points:
            rightmost_left = max(left_points, key=lambda p: p.follower_payoff)

            # Find the leftmost point from the right side
            if right_points:
                leftmost_right = min(right_points, key=lambda p: p.follower_payoff)

                # Interpolate a point exactly at the threshold
                if rightmost_left.follower_payoff != leftmost_right.follower_payoff:
                    # Linear interpolation between the two points
                    t = (threshold - rightmost_left.follower_payoff) / (
                            leftmost_right.follower_payoff - rightmost_left.follower_payoff
                    )
                    interpolated_leader_payoff = (
                            rightmost_left.leader_payoff +
                            t * (leftmost_right.leader_payoff - rightmost_left.leader_payoff)
                    )

                    # Add the interpolated point at the threshold
                    threshold_point = EPFPoint(threshold, interpolated_leader_payoff)
                    truncated_points.append(threshold_point)
                else:
                    # Points have same follower payoff, use the better leader payoff
                    better_leader_payoff = max(rightmost_left.leader_payoff, leftmost_right.leader_payoff)
                    threshold_point = EPFPoint(threshold, better_leader_payoff)
                    truncated_points.append(threshold_point)
            else:
                # No points to the right, add a point at threshold with same leader payoff
                threshold_point = EPFPoint(threshold, rightmost_left.leader_payoff)
                truncated_points.append(threshold_point)

        # Add all points at or to the right of threshold
        truncated_points.extend(right_points)

        # Create new EPF with truncated points
        result_epf = EPF(truncated_points)
        result_epf.set_left_truncate(threshold)


        return result_epf

    def evaluate(self, mu2: float) -> float:
        """Evaluate EPF at follower payoff μ2"""
        if mu2 < self.left_truncate_threshold:
            return float('-inf')

        if not self.points:
            return float('-inf')

        # Handle boundary cases
        if mu2 <= self.points[0].follower_payoff:
            return self.points[0].leader_payoff
        if mu2 >= self.points[-1].follower_payoff:
            return self.points[-1].leader_payoff

        # Linear interpolation between adjacent points
        for i in range(len(self.points) - 1):
            p1, p2 = self.points[i], self.points[i + 1]
            if p1.follower_payoff <= mu2 <= p2.follower_payoff:
                if p2.follower_payoff == p1.follower_payoff:
                    return max(p1.leader_payoff, p2.leader_payoff)

                # Linear interpolation
                t = (mu2 - p1.follower_payoff) / (p2.follower_payoff - p1.follower_payoff)
                return p1.leader_payoff + t * (p2.leader_payoff - p1.leader_payoff)

        return float('-inf')

    def get_maximum_point(self) -> Tuple[float, float]:
        """Get the point that maximizes leader payoff"""
        if not self.points:
            return (0.0, 0.0)

        # Find valid points (above truncation threshold)
        valid_points = [p for p in self.points if p.follower_payoff >= self.left_truncate_threshold]
        if not valid_points:
            return (0.0, 0.0)

        # Return point with maximum leader payoff
        max_point = max(valid_points, key=lambda p: p.leader_payoff)
        return (max_point.follower_payoff, max_point.leader_payoff)

    def upper_concave_envelope(self) -> 'EPF':
        """Compute upper concave envelope of this EPF's points"""
        if not self.points:
            return EPF()

        # Remove duplicate points and keep the one with highest leader payoff for each follower payoff
        points_dict = {}
        for p in self.points:
            if p.follower_payoff not in points_dict or p.leader_payoff > points_dict[p.follower_payoff]:
                points_dict[p.follower_payoff] = p.leader_payoff

        # Convert back to list of points and sort by follower payoff
        unique_points = [EPFPoint(mu2, u1) for mu2, u1 in points_dict.items()]
        unique_points.sort(key=lambda p: p.follower_payoff)

        if len(unique_points) <= 1:
            result_epf = EPF(unique_points)
            result_epf.left_truncate_threshold = self.left_truncate_threshold
            return result_epf

        # Use convex hull algorithm to find upper concave envelope
        # For concave functions, we want the "upper" envelope which is like lower convex hull but flipped
        envelope_points = []

        # Add first point
        envelope_points.append(unique_points[0])

        for i in range(1, len(unique_points)):
            current_point = unique_points[i]

            # Remove points that violate concavity
            while len(envelope_points) >= 2:
                p1 = envelope_points[-2]
                p2 = envelope_points[-1]
                p3 = current_point

                # Check if p2 is below the line from p1 to p3 (violates concavity)
                # For concavity, the middle point should be above the line connecting the endpoints
                if self._violates_concavity(p1, p2, p3):
                    envelope_points.pop()
                else:
                    break

            envelope_points.append(current_point)

        result_epf = EPF(envelope_points)
        result_epf.left_truncate_threshold = self.left_truncate_threshold
        return result_epf

    def _violates_concavity(self, p1: EPFPoint, p2: EPFPoint, p3: EPFPoint) -> bool:
        """Check if p2 violates concavity between p1 and p3"""
        if p3.follower_payoff == p1.follower_payoff:
            return False

        # Calculate where p2 should be on the line from p1 to p3
        t = (p2.follower_payoff - p1.follower_payoff) / (p3.follower_payoff - p1.follower_payoff)
        expected_leader_payoff = p1.leader_payoff + t * (p3.leader_payoff - p1.leader_payoff)

        # For concavity, p2 should be above or equal to the line
        # If p2 is below the line, it violates concavity and should be removed
        return p2.leader_payoff < expected_leader_payoff - 1e-9

    def _is_below_line(self, p1: EPFPoint, p2: EPFPoint, p3: EPFPoint) -> bool:
        """Check if p3 is below the line from p1 to p2 (for concavity)"""
        if p2.follower_payoff == p1.follower_payoff:
            return False

        # Calculate expected y-value on line p1-p2 at x-coordinate of p3
        t = (p3.follower_payoff - p1.follower_payoff) / (p2.follower_payoff - p1.follower_payoff)
        expected_y = p1.leader_payoff + t * (p2.leader_payoff - p1.leader_payoff)

        # For concavity, p3 should be above or on the line
        return p3.leader_payoff < expected_y - 1e-9

    def _is_concave_vertex(self, p1: EPFPoint, p2: EPFPoint, p3: EPFPoint) -> bool:
        """Check if p2 is a necessary vertex for maintaining concavity"""
        if p1.follower_payoff == p3.follower_payoff:
            return True

        # Calculate what p2's y-value would be if it were on the line from p1 to p3
        t = (p2.follower_payoff - p1.follower_payoff) / (p3.follower_payoff - p1.follower_payoff)
        line_y = p1.leader_payoff + t * (p3.leader_payoff - p1.leader_payoff)

        # p2 is a vertex if it's above the line p1-p3 (maintaining concavity)
        return p2.leader_payoff > line_y + 1e-9


@dataclass
class GameNode:
    """Node in the extensive-form game tree"""
    state: GameState
    player: int  # 0 = leader, 1 = follower
    children: Dict[Action, 'GameNode'] = field(default_factory=dict)
    parent: Optional['GameNode'] = None
    epf: Optional[EPF] = None
    mixed_strategy: Dict[Action, float] = field(default_factory=dict)
    is_terminal: bool = False

    def add_child(self, action: Action, child: 'GameNode'):
        """Add a child node"""
        self.children[action] = child
        child.parent = self


class ForestEPFSolver:
    """EPF Solver for Forest Collection Game implementing SEFCE algorithm"""

    def __init__(self, mdp: ForestCollectionMDP, max_depth: int = 15):
        self.mdp = mdp
        self.max_depth = max_depth
        self.game_tree = None
        self.state_to_node = {}  # Cache for states

    def solve(self, initial_state: GameState) -> Tuple[float, float, Dict[str, Dict[Action, float]]]:
        """
        Solve for optimal SEFCE using EPFs

        Returns:
            (leader_payoff, follower_payoff, mixed_strategy)
        """
        print("Building game tree...")
        self.game_tree = self._build_tree(initial_state, 0)

        print("Computing EPFs (Phase 1)...")
        self._compute_epfs(self.game_tree)

        print("Extracting strategy (Phase 2)...")
        strategy = self._extract_strategy()

        # Get optimal payoffs
        mu2_opt, u1_opt = self.game_tree.epf.get_maximum_point()

        print(f"Optimal solution: Leader={u1_opt:.2f}, Follower={mu2_opt:.2f}")
        return u1_opt, mu2_opt, strategy

    def _build_tree(self, state: GameState, depth: int) -> GameNode:
        """Build the extensive-form game tree"""
        state_key = state.to_key()

        # Check if game is terminal
        is_terminal = (state.is_terminal() or depth >= self.max_depth)

        # Check cache (but not for terminal nodes to avoid EPF sharing)
        if not is_terminal and state_key in self.state_to_node:
            return self.state_to_node[state_key]

        # Determine current player based on the game state
        current_player = 0 if state.is_leader_turn() else 1

        # Create node
        node = GameNode(
            state=state,
            player=current_player,
            is_terminal=is_terminal
        )

        self.state_to_node[state_key] = node

        # Handle terminal nodes
        if is_terminal:
            # Create EPF for terminal node based on accumulated rewards
            leader_reward = self.mdp.get_leader_reward(state)
            follower_reward = self.mdp.get_follower_reward(state)

            epf = EPF()
            epf.add_point(follower_reward, leader_reward)
            node.epf = epf
            return node

        # Handle non-terminal nodes - get valid actions for current player
        if state.is_leader_turn():
            if state.leader_steps_left > 0:
                valid_actions = self.mdp.get_valid_actions(state.leader_pos)
            else:
                valid_actions = []
        else:
            if state.follower_steps_left > 0:
                valid_actions = self.mdp.get_valid_actions(state.follower_pos)
            else:
                valid_actions = []

        if not valid_actions:
            # No valid actions - treat as terminal
            node.is_terminal = True
            leader_reward = self.mdp.get_leader_reward(state)
            follower_reward = self.mdp.get_follower_reward(state)

            epf = EPF()
            epf.add_point(follower_reward, leader_reward)
            node.epf = epf
            return node

        # Build children for each valid action
        for action in valid_actions:
            if self.mdp.is_valid_action(state, action):
                next_state = self.mdp.transition(state, action)
                child = self._build_tree(next_state, depth + 1)
                node.add_child(action, child)

        return node

    def _compute_epfs(self, node: GameNode):
        """Compute EPFs using backward induction (Phase 1)"""
        # Post-order traversal - compute children first
        for child in node.children.values():
            self._compute_epfs(child)

        if node.is_terminal:
            return  # EPF already set in _build_tree

        if node.player == 0:  # Leader node
            # Leader can mix freely - take upper concave envelope
            node.epf = self._compute_leader_epf(node)
        else:  # Follower node
            # Apply incentive compatibility constraints
            node.epf = self._compute_follower_epf(node)

    def _compute_leader_epf(self, node: GameNode) -> EPF:
        """Compute EPF at leader node - upper concave envelope of children EPFs"""
        if not node.children:
            return EPF()

        # Combine all EPF points from all children
        all_points = []
        children_epfs = list(node.children.values())

        for child in children_epfs:
            if child.epf and child.epf.points:
                all_points.extend(child.epf.points)

        if not all_points:
            return EPF()

        # Create combined EPF and compute upper concave envelope
        combined_epf = EPF(all_points)
        return combined_epf.upper_concave_envelope()

    def _compute_follower_epf(self, node: GameNode) -> EPF:
        """Compute EPF at follower node with left truncation"""
        if not node.children:
            return EPF()


        children_epfs = list(node.children.values())


        # Find the maximum of the minimum follower payoffs across all children
        # This is the truncation threshold for incentive compatibility
        truncation_threshold = float('-inf')

        for child in children_epfs:
            if child.epf and child.epf.points:
                min_follower_payoff = min(p.follower_payoff for p in child.epf.points)
                truncation_threshold = max(truncation_threshold, min_follower_payoff)


        for child in children_epfs:
            if child.epf and child.epf.points:
                child.epf.set_left_truncate(truncation_threshold)
                child.epf = child.epf.truncate()


        # Combine all EPF points from all children
        all_points = []
        for child in children_epfs:
            if child.epf and child.epf.points:
                valid_points = [p for p in child.epf.points]
                all_points.extend(valid_points)


        if not all_points:
            return EPF()

        # Create EPF with all valid points and compute upper concave envelope
        combined_epf = EPF(all_points)
        envelope_epf = combined_epf.upper_concave_envelope()

        return envelope_epf

    def _extract_strategy(self) -> Dict[str, Dict[Action, float]]:
        """Extract mixed strategy from EPF solution (Phase 2)"""
        if not self.game_tree or not self.game_tree.epf:
            return {}

        # Get optimal point
        mu2_opt, u1_opt = self.game_tree.epf.get_maximum_point()

        # Extract strategy via one-step lookahead
        strategy = {}
        self._extract_node_strategy(self.game_tree, mu2_opt, strategy, [])

        return strategy

    def _extract_node_strategy(self, node: GameNode, target_mu2: float,
                               strategy: Dict, path: List[Action]):
        """Extract strategy for a node given target follower payoff"""
        if node.is_terminal or not node.children:
            return

        state_key = f"depth_{len(path)}_player_{node.player}"

        if node.player == 0:  # Leader node
            # Find which children can achieve the target
            viable_children = []

            for action, child in node.children.items():
                if child.epf and child.epf.evaluate(target_mu2) != float('-inf'):
                    viable_children.append((action, child))

            if not viable_children:
                return

            if len(viable_children) == 1:
                # Pure strategy
                action, child = viable_children[0]
                strategy[state_key] = {action: 1.0}
                self._extract_node_strategy(child, target_mu2, strategy, path + [action])

            else:
                # Mixed strategy - find optimal mixing
                # For two children, solve for mixing probabilities
                action1, child1 = viable_children[0]
                action2, child2 = viable_children[1] if len(viable_children) > 1 else viable_children[0]

                # Simple case: equal mixing (could be optimized further)
                strategy[state_key] = {action1: 0.5, action2: 0.5}

                # Recursively extract for both children
                self._extract_node_strategy(child1, target_mu2, strategy, path + [action1])
                if child2 != child1:
                    self._extract_node_strategy(child2, target_mu2, strategy, path + [action2])

        else:  # Follower node
            # Follower chooses best response - find action that achieves target
            best_action = None
            best_value = float('-inf')

            for action, child in node.children.items():
                if child.epf:
                    value = child.epf.evaluate(target_mu2)
                    if value > best_value:
                        best_value = value
                        best_action = action

            if best_action:
                # Follower plays pure strategy (best response)
                strategy[state_key] = {best_action: 1.0}
                self._extract_node_strategy(node.children[best_action], target_mu2,
                                            strategy, path + [best_action])

    def visualize_tree(self, node: GameNode = None, max_depth: int = 3, show_epf: bool = True):
        """
        Visualize the game tree structure

        Args:
            node: Root node to start visualization from (default: game_tree root)
            max_depth: Maximum depth to display (default: 3)
            show_epf: Whether to show EPF information (default: True)
        """
        if node is None:
            node = self.game_tree

        if not node:
            print("No tree to visualize")
            return

        print("=== GAME TREE VISUALIZATION ===")
        self._print_tree_node(node, depth=0, max_depth=max_depth, show_epf=show_epf, prefix="")
        print("=" * 50)

    def _print_tree_node(self, node: GameNode, depth: int, max_depth: int, show_epf: bool, prefix: str):
        """Helper function to recursively print tree nodes"""
        if depth > max_depth:
            return

        # Create node description
        player_name = "Leader" if node.player == 0 else "Follower"
        terminal_str = " [TERMINAL]" if node.is_terminal else ""

        # State information
        if node.state:
            state_info = f"L_pos:{node.state.leader_pos} F_pos:{node.state.follower_pos} " \
                        f"L_steps:{node.state.leader_steps_left} F_steps:{node.state.follower_steps_left}"
        else:
            state_info = "No state"

        # EPF information
        epf_info = ""
        if show_epf and node.epf and node.epf.points:
            if len(node.epf.points) == 1:
                p = node.epf.points[0]
                epf_info = f" EPF:({p.follower_payoff:.2f},{p.leader_payoff:.2f})"
            else:
                epf_info = f" EPF:{len(node.epf.points)} points"
        elif show_epf:
            epf_info = " EPF:None"

        print(f"{prefix}├─ {player_name}{terminal_str} | {state_info}{epf_info}")

        # Print children
        if not node.is_terminal and node.children and depth < max_depth:
            child_prefix = prefix + "│  "
            children_list = list(node.children.items())

            for i, (action, child) in enumerate(children_list):
                is_last = (i == len(children_list) - 1)
                action_prefix = "└─" if is_last else "├─"
                next_prefix = prefix + ("   " if is_last else "│  ")

                print(f"{child_prefix}{action_prefix} Action: {action}")
                self._print_tree_node(child, depth + 1, max_depth, show_epf, next_prefix)

    def print_tree_stats(self, node: GameNode = None):
        """Print statistics about the game tree"""
        if node is None:
            node = self.game_tree

        if not node:
            print("No tree to analyze")
            return

        stats = self._collect_tree_stats(node)

        print("=== TREE STATISTICS ===")
        print(f"Total nodes: {stats['total_nodes']}")
        print(f"Terminal nodes: {stats['terminal_nodes']}")
        print(f"Leader nodes: {stats['leader_nodes']}")
        print(f"Follower nodes: {stats['follower_nodes']}")
        print(f"Maximum depth: {stats['max_depth']}")
        print(f"Average branching factor: {stats['total_children'] / max(1, stats['internal_nodes']):.2f}")
        print(f"Nodes with EPF: {stats['nodes_with_epf']}")
        print("=" * 30)

    def _collect_tree_stats(self, node: GameNode, depth: int = 0):
        """Recursively collect statistics about the tree"""
        stats = {
            'total_nodes': 1,
            'terminal_nodes': 1 if node.is_terminal else 0,
            'leader_nodes': 1 if node.player == 0 else 0,
            'follower_nodes': 1 if node.player == 1 else 0,
            'max_depth': depth,
            'total_children': len(node.children),
            'internal_nodes': 0 if node.is_terminal else 1,
            'nodes_with_epf': 1 if node.epf and node.epf.points else 0
        }

        for child in node.children.values():
            child_stats = self._collect_tree_stats(child, depth + 1)
            stats['total_nodes'] += child_stats['total_nodes']
            stats['terminal_nodes'] += child_stats['terminal_nodes']
            stats['leader_nodes'] += child_stats['leader_nodes']
            stats['follower_nodes'] += child_stats['follower_nodes']
            stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'])
            stats['total_children'] += child_stats['total_children']
            stats['internal_nodes'] += child_stats['internal_nodes']
            stats['nodes_with_epf'] += child_stats['nodes_with_epf']

        return stats

    def visualize_epf(self, node: GameNode = None, title: str = "EPF Visualization"):
        """Visualize the EPF at a given node"""
        if node is None:
            node = self.game_tree

        if not node or not node.epf:
            print("No EPF to visualize")
            return

        epf = node.epf
        if not epf.points:
            print("Empty EPF")
            return

        # Generate points for plotting
        mu2_values = [p.follower_payoff for p in epf.points]
        u1_values = [p.leader_payoff for p in epf.points]

        plt.figure(figsize=(10, 6))
        plt.plot(mu2_values, u1_values, 'b-o', linewidth=2, markersize=6)

        plt.xlabel('Follower Payoff (μ₂)')
        plt.ylabel('Leader Payoff (U₁)')
        plt.title(title)
        plt.grid(True, alpha=0.3)

        # Highlight optimal point
        mu2_opt, u1_opt = epf.get_maximum_point()
        plt.plot(mu2_opt, u1_opt, 'ro', markersize=10, label=f'Optimal: ({mu2_opt:.2f}, {u1_opt:.2f})')
        plt.legend()

        plt.tight_layout()
        plt.show()


# Example usage and testing
def test_forest_epf_solver():
    """Test the EPF solver on a small forest game"""
    # Create a small forest for testing
    forest_map = np.array([
        [[5, 2], [3, 8]],  # Wood, Fruit
        [[8, 1], [2, 6]]
    ])

    # Initialize MDP
    mdp = ForestCollectionMDP(
        grid_size=(2, 2),
        forest_map=forest_map,
        leader_start=(0, 0),
        follower_start=(1, 1),
        max_steps_leader=1,
        max_steps_follower=1,
        leader_starts_first=False
    )

    # Create initial state
    initial_state = mdp.get_initial_state()

    # Solve with EPF
    solver = ForestEPFSolver(mdp, max_depth=6)
    leader_payoff, follower_payoff, strategy = solver.solve(initial_state)

    print("\n=== EPF Solution Results ===")
    print(f"Leader payoff: {leader_payoff:.2f}")
    print(f"Follower payoff: {follower_payoff:.2f}")
    print("\nMixed strategy:")
    for state_key, actions in strategy.items():
        if actions:
            print(f"  {state_key}: {actions}")

    # Visualize root EPF
    if solver.game_tree:
        solver.visualize_epf(solver.game_tree, "Root Node EPF")

    return solver, strategy


if __name__ == "__main__":
    solver, strategy = test_forest_epf_solver()
