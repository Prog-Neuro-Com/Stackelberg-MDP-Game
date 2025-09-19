"""
Mixed Strategy Stackelberg Solver for Your Forest Collection Game
This integrates directly with your existing ForestCollectionMDP implementation
"""
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# Import your existing game classes
from src.forest_game import ForestCollectionMDP, GameState, Action, create_test_forest_game

from enum import Enum

class Action(Enum):
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)

@dataclass
class UtilityPoint:
    """Point in 2D utility space: (follower_utility, leader_utility)"""
    follower_utility: float
    leader_utility: float

    def __add__(self, other):
        return UtilityPoint(
            self.follower_utility + other.follower_utility,
            self.leader_utility + other.leader_utility
        )

    def __mul__(self, scalar: float):
        return UtilityPoint(
            self.follower_utility * scalar,
            self.leader_utility * scalar
        )

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __eq__(self, other):
        return (abs(self.follower_utility - other.follower_utility) < 1e-6 and
                abs(self.leader_utility - other.leader_utility) < 1e-6)

@dataclass
class LineSegment:
    """Line segment between two utility points"""
    p1: UtilityPoint
    p2: UtilityPoint
    source_info: str = ""

    def interpolate(self, alpha: float) -> UtilityPoint:
        """Get point at position alpha along the segment"""
        return alpha * self.p1 + (1 - alpha) * self.p2

    def contains_point(self, point: UtilityPoint, tolerance: float = 1e-6) -> bool:
        """Check if point lies on this line segment"""
        # Vector from p1 to p2
        v12_f = self.p2.follower_utility - self.p1.follower_utility
        v12_l = self.p2.leader_utility - self.p1.leader_utility

        # Vector from p1 to point
        v1p_f = point.follower_utility - self.p1.follower_utility
        v1p_l = point.leader_utility - self.p1.leader_utility

        # Check collinearity using cross product
        cross = v12_f * v1p_l - v12_l * v1p_f
        if abs(cross) > tolerance:
            return False

        # Check if point is within segment bounds
        # Use dot product to check if point is between p1 and p2
        dot_product = v1p_f * v12_f + v1p_l * v12_l
        if dot_product < -tolerance:
            return False

        length_squared = v12_f * v12_f + v12_l * v12_l
        if dot_product > length_squared + tolerance:
            return False

        return True

@dataclass
class TreeNode:
    """Node in the extensive-form game tree"""
    state: GameState
    node_id: str
    player: int  # 0 = terminal, 1 = leader, 2 = follower
    children: Dict[Action, 'TreeNode'] = field(default_factory=dict)

    # Mixed strategy algorithm data structures
    S1: List[LineSegment] = field(default_factory=list)  # Line segments
    S2: List[UtilityPoint] = field(default_factory=list)  # Endpoints

    def is_terminal(self) -> bool:
        return self.player == 0 or self.state.is_terminal()

class ForestMixedStackelbergSolver:
    """Mixed Strategy Stackelberg Solver for Forest Collection Game"""

    def __init__(self, forest_mdp: ForestCollectionMDP, max_depth: int = 12):
        self.mdp = forest_mdp
        self.max_depth = max_depth
        self.nodes: Dict[str, TreeNode] = {}
        self.node_counter = 0
        self.debug = True

    def solve_mixed_strategy(self, initial_state: Optional[GameState] = None) -> Dict:
        """
        Solve for mixed strategy Stackelberg equilibrium

        Returns:
            Dictionary with solution information
        """
        if initial_state is None:
            initial_state = self.mdp.get_initial_state()

        print(f"Solving Mixed Strategy Stackelberg Equilibrium")
        print(f"Initial state: {initial_state}")
        print(f"Max depth: {self.max_depth}")

        # Phase 1: Build extensive-form game tree
        print("\nPhase 1: Building game tree...")
        root = self._build_tree(initial_state, 0)
        print(f"Built tree with {len(self.nodes)} nodes")

        # Phase 2: Bottom-up computation of S1 and S2 sets
        print("\nPhase 2: Computing feasible outcome sets...")
        self._upward_pass(root)

        # Phase 3: Find optimal outcome
        print("\nPhase 3: Finding optimal leader outcome...")
        if not root.S2:
            raise ValueError("No feasible outcomes found")

        optimal_point = max(root.S2, key=lambda p: p.leader_utility)
        print(f"Optimal outcome: Leader={optimal_point.leader_utility:.2f}, Follower={optimal_point.follower_utility:.2f}")

        # Phase 4: Compute commitment strategy
        print("\nPhase 4: Computing commitment strategy...")
        commitment_strategy = self._downward_pass(root, optimal_point)

        return {
            'optimal_outcome': optimal_point,
            'commitment_strategy': commitment_strategy,
            'root_epf': root.S2,
            'tree_size': len(self.nodes),
            'root_node': root
        }

    def _build_tree(self, state: GameState, depth: int) -> TreeNode:
        """Build the extensive-form game tree recursively"""
        node_id = f"n{self.node_counter}"
        self.node_counter += 1

        # Determine node type
        if state.is_terminal() or depth >= self.max_depth:
            player = 0  # Terminal
        elif state.is_leader_turn():
            player = 1  # Leader
        else:
            player = 2  # Follower

        node = TreeNode(state=state, node_id=node_id, player=player)
        self.nodes[node_id] = node


        # Base case: terminal node
        if node.is_terminal():
            terminal_point = UtilityPoint(
                follower_utility=float(state.follower_total_fruit),
                leader_utility=float(state.leader_total_wood)
            )
            node.S1 = [LineSegment(terminal_point, terminal_point, f"leaf_{node_id}")]
            node.S2 = [terminal_point]
            return node

        # Recursive case: build children
        current_pos = state.get_current_player_pos()
        valid_actions = self.mdp.get_valid_actions(current_pos)

        for action in valid_actions:
            if self.mdp.is_valid_action(state, action):
                try:
                    next_state = self.mdp.transition(state, action)
                    child_node = self._build_tree(next_state, depth + 1)
                    node.children[action] = child_node
                except Exception as e:
                    if self.debug:
                        print(f"  Warning: Failed to create child with action {action}: {e}")
                    continue

        return node

    def _upward_pass(self, node: TreeNode):
        """Bottom-up computation of S1 and S2 sets"""
        if node.is_terminal():
            return

        # First, recursively process all children
        for child in node.children.values():
            self._upward_pass(child)

        if node.player == 1:  # Leader node
            self._process_leader_node(node)
        else:  # Follower node
            self._process_follower_node(node)

    def _process_leader_node(self, node: TreeNode):
        """Process a leader node - can commit to mixed strategies"""
        node.S1 = []
        node.S2 = []

        if not node.children:
            return

        # Collect all endpoints from children
        for child in node.children.values():
            node.S2.extend(child.S2)

        # Add existing line segments from children
        for child in node.children.values():
            node.S1.extend(child.S1)

        # Key insight: Create line segments by mixing between children
        children_list = list(node.children.values())

        # Mix between pairs of children (allows mixed strategies)
        for i, child1 in enumerate(children_list):
            for j, child2 in enumerate(children_list):
                if i < j:  # Avoid duplicate pairs
                    for p1 in child1.S2:
                        for p2 in child2.S2:
                            if p1 != p2:  # Only create segment if points are different
                                segment = LineSegment(
                                    p1, p2,
                                    f"mix({child1.node_id},{child2.node_id})"
                                )
                                node.S1.append(segment)

    def _process_follower_node(self, node: TreeNode):
        """Process a follower node - will best respond"""
        node.S1 = []
        node.S2 = []

        if not node.children:
            return

        # Follower chooses action that maximizes their utility
        # Compute the upper envelope of all children's possibilities

        all_segments = []
        for action, child in node.children.items():
            for segment in child.S1:
                all_segments.append((segment, action, child.node_id))

        if not all_segments:
            return

        # Compute upper envelope (simplified)
        upper_envelope = self._compute_upper_envelope(all_segments)

        for segment, _, _ in upper_envelope:
            node.S1.append(segment)
            if segment.p1 not in node.S2:
                node.S2.append(segment.p1)
            if segment.p2 not in node.S2:
                node.S2.append(segment.p2)

    def _compute_upper_envelope(self, segments_with_info):
        """Compute upper envelope for follower's best response"""
        if not segments_with_info:
            return []

        # Simplified envelope computation
        # Keep segments that are not strictly dominated
        non_dominated = []

        for seg, action, child_id in segments_with_info:
            is_dominated = False

            # Check if this segment is dominated by any other
            for other_seg, _, _ in segments_with_info:
                if seg != other_seg:
                    # Check if other_seg dominates seg for the follower
                    if (other_seg.p1.follower_utility >= seg.p1.follower_utility and
                        other_seg.p2.follower_utility >= seg.p2.follower_utility and
                        (other_seg.p1.follower_utility > seg.p1.follower_utility or
                         other_seg.p2.follower_utility > seg.p2.follower_utility)):
                        is_dominated = True
                        break

            if not is_dominated:
                non_dominated.append((seg, action, child_id))

        return non_dominated

    def _downward_pass(self, node: TreeNode, target_point: UtilityPoint) -> Dict[str, Dict[Action, float]]:
        """Top-down pass to determine commitment strategy"""
        strategy = {}
        self._compute_strategy_recursive(node, target_point, strategy)
        return strategy

    def _compute_strategy_recursive(self, node: TreeNode, target: UtilityPoint,
                                   strategy_dict: Dict[str, Dict[Action, float]]):
        """Recursively compute strategy to achieve target outcome"""
        if node.is_terminal():
            return

        if node.player == 1:  # Leader node
            # Find line segment containing target point
            containing_segment = None
            alpha = 0.5

            for segment in node.S1:
                if segment.contains_point(target):
                    containing_segment = segment
                    # Calculate interpolation parameter
                    denom_f = segment.p1.follower_utility - segment.p2.follower_utility
                    denom_l = segment.p1.leader_utility - segment.p2.leader_utility

                    if abs(denom_f) > 1e-6:
                        alpha = (target.follower_utility - segment.p2.follower_utility) / denom_f
                    elif abs(denom_l) > 1e-6:
                        alpha = (target.leader_utility - segment.p2.leader_utility) / denom_l
                    else:
                        alpha = 0.5

                    alpha = max(0.0, min(1.0, alpha))
                    break

            if containing_segment is None:
                # Find closest segment
                best_distance = float('inf')
                for segment in node.S1:
                    for point in [segment.p1, segment.p2]:
                        dist = ((point.leader_utility - target.leader_utility)**2 +
                               (point.follower_utility - target.follower_utility)**2)**0.5
                        if dist < best_distance:
                            best_distance = dist
                            containing_segment = segment
                            alpha = 0.0 if point == segment.p2 else 1.0

            if containing_segment is None:
                return

            # Determine strategy based on alpha
            strategy_dict[node.node_id] = {}

            if abs(alpha - 1.0) < 1e-6:  # Pure strategy to p1
                child, action = self._find_child_with_point(node, containing_segment.p1)
                if child and action:
                    strategy_dict[node.node_id][action] = 1.0
                    self._compute_strategy_recursive(child, containing_segment.p1, strategy_dict)

            elif abs(alpha) < 1e-6:  # Pure strategy to p2
                child, action = self._find_child_with_point(node, containing_segment.p2)
                if child and action:
                    strategy_dict[node.node_id][action] = 1.0
                    self._compute_strategy_recursive(child, containing_segment.p2, strategy_dict)

            else:  # Mixed strategy
                child1, action1 = self._find_child_with_point(node, containing_segment.p1)
                child2, action2 = self._find_child_with_point(node, containing_segment.p2)

                if child1 and action1 and child2 and action2:
                    strategy_dict[node.node_id][action1] = alpha
                    strategy_dict[node.node_id][action2] = 1.0 - alpha

                    self._compute_strategy_recursive(child1, containing_segment.p1, strategy_dict)
                    self._compute_strategy_recursive(child2, containing_segment.p2, strategy_dict)

        else:  # Follower node - best response
            # Find child that leads to target point
            best_child = None
            best_action = None
            best_distance = float('inf')

            for action, child in node.children.items():
                for point in child.S2:
                    distance = ((point.leader_utility - target.leader_utility)**2 +
                               (point.follower_utility - target.follower_utility)**2)**0.5
                    if distance < best_distance:
                        best_distance = distance
                        best_child = child
                        best_action = action

            if best_child and best_action:
                strategy_dict[node.node_id] = {best_action: 1.0}
                self._compute_strategy_recursive(best_child, target, strategy_dict)

    def _find_child_with_point(self, node: TreeNode, point: UtilityPoint) -> Tuple[Optional[TreeNode], Optional[Action]]:
        """Find child that contains the given point"""
        for action, child in node.children.items():
            if point in child.S2:
                return child, action

            # Check line segments
            for segment in child.S1:
                if segment.contains_point(point):
                    return child, action

        # Fallback: find closest child
        best_child = None
        best_action = None
        best_distance = float('inf')

        for action, child in node.children.items():
            for child_point in child.S2:
                distance = ((child_point.leader_utility - point.leader_utility)**2 +
                           (child_point.follower_utility - point.follower_utility)**2)**0.5
                if distance < best_distance:
                    best_distance = distance
                    best_child = child
                    best_action = action

        return best_child, best_action

    def visualize_results(self, solution: Dict):
        """Visualize the mixed strategy solution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: EPF (Enforceable Payoff Frontier)
        root_epf = solution['root_epf']
        if root_epf:
            follower_utils = [p.follower_utility for p in root_epf]
            leader_utils = [p.leader_utility for p in root_epf]

            axes[0,0].scatter(follower_utils, leader_utils, c='blue', s=50, alpha=0.7)

            optimal = solution['optimal_outcome']
            axes[0,0].scatter(optimal.follower_utility, optimal.leader_utility,
                            c='red', s=200, marker='*', label='Optimal')

            axes[0,0].set_xlabel('Follower Utility (Fruit)')
            axes[0,0].set_ylabel('Leader Utility (Wood)')
            axes[0,0].set_title('Enforceable Payoff Frontier')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)

        # Plot 2: Strategy summary
        strategy = solution['commitment_strategy']
        if strategy:
            mixed_nodes = []
            mixed_counts = []

            for node_id, actions in strategy.items():
                if len(actions) > 1:  # Mixed strategy
                    mixed_nodes.append(node_id)
                    mixed_counts.append(len(actions))

            if mixed_nodes:
                axes[0,1].bar(range(len(mixed_nodes[:10])), mixed_counts[:10])
                axes[0,1].set_xlabel('Node')
                axes[0,1].set_ylabel('Number of Actions in Mix')
                axes[0,1].set_title('Mixed Strategy Nodes')
                axes[0,1].set_xticks(range(len(mixed_nodes[:10])))
                axes[0,1].set_xticklabels([n[:6] for n in mixed_nodes[:10]], rotation=45)

        # Plot 3: Tree statistics
        tree_stats = {
            'Total Nodes': solution['tree_size'],
            'Terminal Nodes': len([n for n in self.nodes.values() if n.is_terminal()]),
            'Leader Nodes': len([n for n in self.nodes.values() if n.player == 1]),
            'Follower Nodes': len([n for n in self.nodes.values() if n.player == 2])
        }

        axes[1,0].bar(tree_stats.keys(), tree_stats.values())
        axes[1,0].set_title('Game Tree Statistics')
        axes[1,0].tick_params(axis='x', rotation=45)

        # Plot 4: Action distribution
        all_actions = {}
        for node_id, actions in strategy.items():
            for action, prob in actions.items():
                action_name = action.name if hasattr(action, 'name') else str(action)
                all_actions[action_name] = all_actions.get(action_name, 0) + prob

        if all_actions:
            axes[1,1].pie(all_actions.values(), labels=all_actions.keys(), autopct='%1.1f%%')
            axes[1,1].set_title('Action Distribution in Strategy')

        plt.tight_layout()
        return fig

    def print_solution_summary(self, solution: Dict):
        """Print a detailed summary of the solution"""
        print("\n" + "="*60)
        print("MIXED STRATEGY STACKELBERG SOLUTION SUMMARY")
        print("="*60)

        optimal = solution['optimal_outcome']
        print(f"Optimal Outcome:")
        print(f"  Leader Utility (Wood): {optimal.leader_utility:.3f}")
        print(f"  Follower Utility (Fruit): {optimal.follower_utility:.3f}")

        print(f"\nTree Statistics:")
        print(f"  Total Nodes: {solution['tree_size']}")
        print(f"  EPF Points at Root: {len(solution['root_epf'])}")

        strategy = solution['commitment_strategy']
        mixed_strategies = {k: v for k, v in strategy.items() if len(v) > 1}
        pure_strategies = {k: v for k, v in strategy.items() if len(v) == 1}

        print(f"\nStrategy Summary:")
        print(f"  Nodes with Pure Strategies: {len(pure_strategies)}")
        print(f"  Nodes with Mixed Strategies: {len(mixed_strategies)}")

        if mixed_strategies:
            print(f"\nSample Mixed Strategies:")
            for i, (node_id, actions) in enumerate(list(mixed_strategies.items())[:5]):
                print(f"  {node_id}:")
                for action, prob in actions.items():
                    action_name = action.name if hasattr(action, 'name') else str(action)
                    print(f"    {action_name}: {prob:.3f}")

        print("="*60)

# Main execution function
def solve_forest_mixed_strategy(game: ForestCollectionMDP):
    """
    Solve your forest collection game using mixed strategy Stackelberg equilibrium

    Args:
        :param game:
    """

    print("Creating Forest Collection Game...")

    # Create solver and solve
    solver = ForestMixedStackelbergSolver(game)
    solution = solver.solve_mixed_strategy()

    # Print results
    solver.print_solution_summary(solution)

    # Visualize results
    fig = solver.visualize_results(solution)
    plt.show()

    return solution, solver


# Test function
if __name__ == "__main__":
    print("Testing Mixed Strategy Stackelberg Solver for Forest Collection Game")
    print("="*70)

    # Run the solver
    solution, solver = solve_forest_mixed_strategy(None)

    if solution:
        print(f"\nSolution completed successfully!")
        print(f"Check the visualization plots for detailed results.")
    else:
        print("Solution failed - check imports and game setup.")