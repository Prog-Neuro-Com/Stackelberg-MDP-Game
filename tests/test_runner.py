"""
Test Runner for Mixed Strategy Algorithm on 2x2 Forest Game
This applies our Stackelberg algorithm to the simple test case
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# Import the test case
from tests.mixed_test import SimpleFores2x2MDP, GameState, Action


@dataclass
class UtilityPoint:
    follower_utility: float
    leader_utility: float

    def __add__(self, other):
        return UtilityPoint(self.follower_utility + other.follower_utility,
                            self.leader_utility + other.leader_utility)

    def __mul__(self, scalar: float):
        return UtilityPoint(self.follower_utility * scalar, self.leader_utility * scalar)

    def __rmul__(self, scalar: float):
        return self.__mul__(scalar)

    def __eq__(self, other):
        return (abs(self.follower_utility - other.follower_utility) < 1e-6 and
                abs(self.leader_utility - other.leader_utility) < 1e-6)

    def __str__(self):
        return f"({self.follower_utility:.1f}, {self.leader_utility:.1f})"


@dataclass
class LineSegment:
    p1: UtilityPoint
    p2: UtilityPoint
    source: str = ""

    def contains_point(self, point: UtilityPoint, tolerance: float = 1e-6) -> bool:
        # Vector from p1 to p2
        v12_f = self.p2.follower_utility - self.p1.follower_utility
        v12_l = self.p2.leader_utility - self.p1.leader_utility

        # Vector from p1 to point
        v1p_f = point.follower_utility - self.p1.follower_utility
        v1p_l = point.leader_utility - self.p1.leader_utility

        # Check collinearity
        cross = v12_f * v1p_l - v12_l * v1p_f
        if abs(cross) > tolerance:
            return False

        # Check bounds
        dot_product = v1p_f * v12_f + v1p_l * v12_l
        if dot_product < -tolerance:
            return False

        length_squared = v12_f * v12_f + v12_l * v12_l
        if dot_product > length_squared + tolerance:
            return False

        return True

    def __str__(self):
        return f"[{self.p1} -- {self.p2}]"


@dataclass
class TreeNode:
    state: GameState
    node_id: str
    player: int  # 0=terminal, 1=leader, 2=follower
    children: Dict[Action, 'TreeNode'] = field(default_factory=dict)
    S1: List[LineSegment] = field(default_factory=list)
    S2: List[UtilityPoint] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return self.player == 0 or self.state.is_terminal()


class SimpleStackelbergSolver:
    """Simplified Stackelberg solver for manual verification"""

    def __init__(self, game: SimpleFores2x2MDP):
        self.game = game
        self.nodes = {}
        self.node_counter = 0

    def solve(self) -> Dict:
        print("=" * 50)
        print("APPLYING MIXED STRATEGY ALGORITHM")
        print("=" * 50)

        initial_state = self.game.get_initial_state()

        # Phase 1: Build tree
        print("Phase 1: Building game tree...")
        root = self._build_tree(initial_state)
        print(f"Built tree with {len(self.nodes)} nodes")

        # Phase 2: Bottom-up pass
        print("\nPhase 2: Computing S1 and S2 sets...")
        self._upward_pass(root)

        # Phase 3: Find optimal
        print("\nPhase 3: Finding optimal outcome...")
        if not root.S2:
            raise ValueError("No outcomes found")

        optimal_point = max(root.S2, key=lambda p: p.leader_utility)
        print(f"Optimal outcome: {optimal_point}")

        # Phase 4: Extract strategy
        print("\nPhase 4: Computing strategy...")
        strategy = self._extract_strategy(root, optimal_point)

        return {
            'optimal_outcome': optimal_point,
            'strategy': strategy,
            'root_epf': root.S2,
            'root': root
        }

    def _build_tree(self, state: GameState) -> TreeNode:
        node_id = f"n{self.node_counter}"
        self.node_counter += 1

        # Determine player
        if state.is_terminal():
            player = 0
        elif state.is_leader_turn():
            player = 1
        else:
            player = 2

        node = TreeNode(state=state, node_id=node_id, player=player)
        self.nodes[node_id] = node

        # Terminal case
        if node.is_terminal():
            point = UtilityPoint(
                follower_utility=float(state.follower_total_fruit),
                leader_utility=float(state.leader_total_wood)
            )
            node.S1 = [LineSegment(point, point, f"leaf_{node_id}")]
            node.S2 = [point]
            print(f"  Terminal {node_id}: {point}")
            return node

        # Build children
        pos = state.get_current_player_pos()
        valid_actions = self.game.get_valid_actions(pos)

        for action in valid_actions:
            if self.game.is_valid_action(state, action):
                next_state = self.game.transition(state, action)
                child = self._build_tree(next_state)
                node.children[action] = child

        return node

    def _upward_pass(self, node: TreeNode):
        if node.is_terminal():
            return

        # Process children first
        for child in node.children.values():
            self._upward_pass(child)

        if node.player == 1:  # Leader
            self._process_leader_node(node)
        else:  # Follower
            self._process_follower_node(node)

    def _process_leader_node(self, node: TreeNode):
        """Leader can commit to mixed strategies"""
        print(f"\nProcessing Leader Node {node.node_id}:")
        print(f"  State: Leader at {node.state.leader_pos}, Follower at {node.state.follower_pos}")

        node.S1 = []
        node.S2 = []

        # Collect all points from children
        for action, child in node.children.items():
            print(f"  Child {action.name}: {len(child.S2)} points")
            for point in child.S2:
                print(f"    {point}")
            node.S2.extend(child.S2)

        # Add existing segments from children
        for child in node.children.values():
            node.S1.extend(child.S1)

        # Create mixing segments between different children
        children_list = list(node.children.values())
        for i, child1 in enumerate(children_list):
            for j, child2 in enumerate(children_list):
                if i < j:
                    for p1 in child1.S2:
                        for p2 in child2.S2:
                            if p1 != p2:
                                segment = LineSegment(p1, p2, f"mix({child1.node_id},{child2.node_id})")
                                node.S1.append(segment)
                                print(f"  Created mixing segment: {segment}")

        print(f"  Final: {len(node.S2)} points, {len(node.S1)} segments")

    def _process_follower_node(self, node: TreeNode):
        """Follower best responds"""
        print(f"\nProcessing Follower Node {node.node_id}:")
        print(f"  State: Leader at {node.state.leader_pos}, Follower at {node.state.follower_pos}")

        node.S1 = []
        node.S2 = []

        if not node.children:
            return

        # For each child, show what follower gets
        child_outcomes = []
        for action, child in node.children.items():
            print(f"  If follower plays {action.name}:")
            for point in child.S2:
                print(f"    Outcome: {point}")
                child_outcomes.append((point, action, child))

        # Follower chooses to maximize their utility
        # For simplicity, just keep all non-dominated outcomes
        best_follower_utility = max(point.follower_utility for point, _, _ in child_outcomes)

        print(f"  Best follower utility: {best_follower_utility}")

        for point, action, child in child_outcomes:
            if point.follower_utility >= best_follower_utility - 1e-6:  # Keep ties
                node.S2.append(point)
                node.S1.extend(child.S1)
                print(f"  Keeping outcome from {action.name}: {point}")

    def _extract_strategy(self, node: TreeNode, target: UtilityPoint) -> Dict:
        """Extract the strategy that achieves target outcome"""
        strategy = {}
        self._extract_recursive(node, target, strategy)
        return strategy

    def _extract_recursive(self, node: TreeNode, target: UtilityPoint, strategy_dict: Dict):
        if node.is_terminal():
            return

        if node.player == 1:  # Leader
            print(f"\nExtracting strategy for Leader Node {node.node_id}")
            print(f"  Target: {target}")

            # Find segment containing target
            containing_segment = None
            for segment in node.S1:
                if segment.contains_point(target):
                    containing_segment = segment
                    print(f"  Found containing segment: {segment}")
                    break

            if containing_segment is None:
                # Find closest point
                closest_point = min(node.S2,
                                    key=lambda p: (p.leader_utility - target.leader_utility) ** 2 +
                                                  (p.follower_utility - target.follower_utility) ** 2)
                print(f"  No containing segment, using closest point: {closest_point}")

                # Find child with this point
                for action, child in node.children.items():
                    if closest_point in child.S2:
                        strategy_dict[node.node_id] = {action: 1.0}
                        self._extract_recursive(child, closest_point, strategy_dict)
                        return
            else:
                # Calculate mixing parameter
                p1, p2 = containing_segment.p1, containing_segment.p2

                if abs(p1.follower_utility - p2.follower_utility) > 1e-6:
                    alpha = ((target.follower_utility - p2.follower_utility) /
                             (p1.follower_utility - p2.follower_utility))
                else:
                    alpha = ((target.leader_utility - p2.leader_utility) /
                             (p1.leader_utility - p2.leader_utility))

                alpha = max(0.0, min(1.0, alpha))
                print(f"  Mixing parameter alpha = {alpha:.3f}")

                if abs(alpha - 1.0) < 1e-6:  # Pure strategy to p1
                    child, action = self._find_child_with_point(node, p1)
                    strategy_dict[node.node_id] = {action: 1.0}
                    print(f"  Pure strategy: {action.name}")
                    self._extract_recursive(child, p1, strategy_dict)

                elif abs(alpha) < 1e-6:  # Pure strategy to p2
                    child, action = self._find_child_with_point(node, p2)
                    strategy_dict[node.node_id] = {action: 1.0}
                    print(f"  Pure strategy: {action.name}")
                    self._extract_recursive(child, p2, strategy_dict)

                else:  # Mixed strategy
                    child1, action1 = self._find_child_with_point(node, p1)
                    child2, action2 = self._find_child_with_point(node, p2)

                    strategy_dict[node.node_id] = {action1: alpha, action2: 1.0 - alpha}
                    print(f"  Mixed strategy: {alpha:.3f} * {action1.name} + {1 - alpha:.3f} * {action2.name}")

                    self._extract_recursive(child1, p1, strategy_dict)
                    self._extract_recursive(child2, p2, strategy_dict)

        else:  # Follower
            print(f"\nExtracting strategy for Follower Node {node.node_id}")
            print(f"  Target: {target}")

            # Find child with target point
            for action, child in node.children.items():
                if target in child.S2:
                    strategy_dict[node.node_id] = {action: 1.0}
                    print(f"  Follower chooses: {action.name}")
                    self._extract_recursive(child, target, strategy_dict)
                    return

    def _find_child_with_point(self, node: TreeNode, point: UtilityPoint) -> Tuple[
        Optional[TreeNode], Optional[Action]]:
        for action, child in node.children.items():
            if point in child.S2:
                return child, action
        return None, None


def test_mixed_strategy_algorithm():
    """Run the test case through our algorithm"""
    print("TESTING MIXED STRATEGY ALGORITHM ON 2x2 FOREST")
    print("=" * 60)

    # Create the game
    game = SimpleFores2x2MDP()

    # Show the setup
    print("Game Setup:")
    print("  Forest Layout:")
    print("    (0,1): W=7, F=3     (1,1): W=0, F=0  <- Follower starts")
    print("    (0,0): W=8, F=2     (1,0): W=1, F=9")
    print("           ^")
    print("        Leader starts")
    print("  Each player has 1 step")
    print()

    # First show manual analysis
    print("MANUAL ANALYSIS:")
    print("-" * 30)
    initial_state = game.get_initial_state()

    print("All possible game outcomes:")
    outcomes = []

    for leader_action in [Action.STAY, Action.UP, Action.RIGHT]:
        state1 = game.transition(initial_state, leader_action)

        for follower_action in game.get_valid_actions(state1.follower_pos):
            if game.is_valid_action(state1, follower_action):
                final_state = game.transition(state1, follower_action)
                outcome = {
                    'leader_action': leader_action.name,
                    'follower_action': follower_action.name,
                    'leader_utility': final_state.leader_total_wood,
                    'follower_utility': final_state.follower_total_fruit,
                    'final_leader_pos': final_state.leader_pos,
                    'final_follower_pos': final_state.follower_pos
                }
                outcomes.append(outcome)
                print(f"  {leader_action.name:5} → {follower_action.name:5}: "
                      f"Leader={final_state.leader_total_wood}, Follower={final_state.follower_total_fruit} "
                      f"(L@{final_state.leader_pos}, F@{final_state.follower_pos})")

    # Find Stackelberg outcomes (follower best responses)
    from collections import defaultdict
    by_leader = defaultdict(list)
    for outcome in outcomes:
        by_leader[outcome['leader_action']].append(outcome)

    print("\nStackelberg analysis (follower best responses):")
    stackelberg_outcomes = []
    for leader_action, group in by_leader.items():
        best_for_follower = max(group, key=lambda x: x['follower_utility'])
        stackelberg_outcomes.append(best_for_follower)
        print(f"  If leader commits to {leader_action}: "
              f"follower plays {best_for_follower['follower_action']}, "
              f"outcome = Leader {best_for_follower['leader_utility']}, "
              f"Follower {best_for_follower['follower_utility']}")

    best_pure = max(stackelberg_outcomes, key=lambda x: x['leader_utility'])
    print(f"\nBest pure strategy for leader:")
    print(f"  Action: {best_pure['leader_action']}")
    print(f"  Outcome: Leader {best_pure['leader_utility']}, Follower {best_pure['follower_utility']}")

    # Now run our algorithm
    print("\n" + "=" * 60)
    print("RUNNING MIXED STRATEGY ALGORITHM")
    print("=" * 60)

    solver = SimpleStackelbergSolver(game)
    solution = solver.solve()

    print("\n" + "=" * 60)
    print("COMPARISON OF RESULTS")
    print("=" * 60)

    optimal = solution['optimal_outcome']
    print(f"Pure strategy result:  Leader = {best_pure['leader_utility']}, Follower = {best_pure['follower_utility']}")
    print(f"Mixed strategy result: Leader = {optimal.leader_utility}, Follower = {optimal.follower_utility}")

    improvement = optimal.leader_utility - best_pure['leader_utility']
    print(f"Improvement from mixed strategy: {improvement}")

    if abs(improvement) < 1e-6:
        print("→ No improvement! Mixed strategy not beneficial in this case.")
        print("  (This might happen if the game structure doesn't reward mixing)")
    else:
        print(f"→ Mixed strategy provides improvement of {improvement}!")

    print(f"\nCommitment strategy found:")
    for node_id, actions in solution['strategy'].items():
        if actions:  # Only show non-empty strategies
            action_str = ", ".join([f"{action.name}: {prob:.3f}" for action, prob in actions.items()])
            print(f"  Node {node_id}: {action_str}")
            if len(actions) > 1:
                print(f"    ^ This is a MIXED strategy!")

    # Visualize the EPF
    epf_points = solution['root_epf']
    if len(epf_points) > 1:
        plt.figure(figsize=(10, 6))

        # Plot all feasible outcomes
        follower_utils = [p.follower_utility for p in epf_points]
        leader_utils = [p.leader_utility for p in epf_points]
        plt.scatter(follower_utils, leader_utils, c='blue', s=100, alpha=0.7, label='Feasible outcomes')

        # Highlight optimal
        opt = solution['optimal_outcome']
        plt.scatter(opt.follower_utility, opt.leader_utility, c='red', s=200, marker='*', label='Optimal (mixed)')

        # Mark pure strategy result
        plt.scatter(best_pure['follower_utility'], best_pure['leader_utility'],
                    c='green', s=150, marker='s', label='Best pure strategy')

        plt.xlabel('Follower Utility (Fruit)')
        plt.ylabel('Leader Utility (Wood)')
        plt.title('Enforceable Payoff Frontier - 2x2 Forest Test Case')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Annotate points
        for i, point in enumerate(epf_points):
            plt.annotate(f'({point.follower_utility:.1f},{point.leader_utility:.1f})',
                         (point.follower_utility, point.leader_utility),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

        plt.show()

    return solution, game


if __name__ == "__main__":
    solution, game = test_mixed_strategy_algorithm()