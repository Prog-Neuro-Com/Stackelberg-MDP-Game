"""
EPF Solver Demonstration and Step-by-Step Analysis

This script demonstrates the key concepts of EPF computation:
1. Phase 1: Computing EPFs via backward induction
2. Left truncation for incentive compatibility
3. Upper concave envelope computation
4. Mixed strategy extraction (at most 2 children)
5. Phase 2: Strategy extraction via one-step lookahead
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import our EPF solver
from forest_epf_solver import ForestEPFSolver, EPF, EPFPoint, GameNode
from src.forest_game import ForestCollectionMDP, GameState, Action


def create_strategic_forest() -> Tuple[ForestCollectionMDP, GameState]:
    """Create a forest that demonstrates strategic conflicts"""
    # Design a forest with clear strategic tensions
    forest_map = np.array([
        [[10, 1], [2, 10], [5, 5]],  # High wood vs High fruit vs Cooperative
        [[3, 7], [8, 3], [1, 9]],  # Mixed resources with trade-offs
        [[6, 4], [4, 6], [7, 7]]  # Balanced options
    ])

    mdp = ForestCollectionMDP(
        grid_size=(3, 3),
        forest_map=forest_map,
        leader_start=(0, 0),
        follower_start=(0, 0),
        max_steps_leader=4,
        max_steps_follower=4,
        leader_starts_first=True
    )

    initial_state = mdp.get_initial_state()
    return mdp, initial_state


def demonstrate_epf_computation():
    """Step-by-step demonstration of EPF computation"""
    print("=" * 60)
    print("EPF SOLVER DEMONSTRATION")
    print("=" * 60)

    # Create strategic forest
    mdp, initial_state = create_strategic_forest()

    print("\n1. FOREST SETUP")
    print("Forest map (Wood, Fruit):")
    for i in range(3):
        row = []
        for j in range(3):
            wood, fruit = mdp.forest_map[i, j]
            row.append(f"({wood},{fruit})")
        print(f"  Row {i}: {' '.join(row)}")

    print(f"\nInitial state: Leader at {initial_state.leader_pos}, Follower at {initial_state.follower_pos}")
    print(f"Max steps: Leader={mdp.max_steps_leader}, Follower={mdp.max_steps_follower}")

    # Initialize solver
    solver = ForestEPFSolver(mdp, max_depth=8)

    print("\n2. BUILDING GAME TREE")
    solver.game_tree = solver._build_tree(initial_state, 0)

    # Count nodes
    total_nodes = count_nodes(solver.game_tree)
    terminal_nodes = count_terminal_nodes(solver.game_tree)

    print(f"Game tree built:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Terminal nodes: {terminal_nodes}")
    print(f"  Non-terminal nodes: {total_nodes - terminal_nodes}")

    print("\n3. PHASE 1: COMPUTING EPFs")
    print("Computing EPFs via backward induction...")

    # Demonstrate EPF computation on a small subtree
    demo_node = find_interesting_node(solver.game_tree)
    if demo_node:
        print(f"\nDemonstrating on node at depth with {len(demo_node.children)} children")
        print(f"Node player: {'Leader' if demo_node.player == 0 else 'Follower'}")

        # Show children EPFs before combination
        if not demo_node.is_terminal:
            print("\nChildren EPFs:")
            for action, child in demo_node.children.items():
                if child.epf and child.epf.points:
                    print(f"  Action {action}: {child.epf.points}")

    # Compute all EPFs
    solver._compute_epfs(solver.game_tree)

    print("\n4. ROOT EPF ANALYSIS")
    root_epf = solver.game_tree.epf
    if root_epf and root_epf.points:
        print(f"Root EPF has {len(root_epf.points)} points:")
        for i, point in enumerate(root_epf.points):
            print(f"  Point {i + 1}: Follower={point.follower_payoff:.2f}, Leader={point.leader_payoff:.2f}")

        print(f"Left truncation threshold: {root_epf.left_truncate_threshold:.2f}")

        # Find optimal point
        mu2_opt, u1_opt = root_epf.get_maximum_point()
        print(f"Optimal point: Follower={mu2_opt:.2f}, Leader={u1_opt:.2f}")

    print("\n5. PHASE 2: STRATEGY EXTRACTION")
    strategy = solver._extract_strategy()

    print("Extracted mixed strategy:")
    for state_key, actions in strategy.items():
        if actions:
            action_str = ", ".join(f"{action.name}:{prob:.2f}" for action, prob in actions.items())
            print(f"  {state_key}: {action_str}")

    print("\n6. DEMONSTRATING KEY CONCEPTS")

    # Concept 1: Mixed strategy limitation (at most 2 children)
    print("\nCONCEPT 1: Mixed Strategy Limitation")
    mixed_nodes = find_mixed_strategy_nodes(solver.game_tree, strategy)
    print(f"Found {len(mixed_nodes)} nodes with mixed strategies:")
    for node_info in mixed_nodes[:3]:  # Show first 3
        print(f"  {node_info}")

    # Concept 2: Left truncation examples
    print("\nCONCEPT 2: Left Truncation (Incentive Compatibility)")
    truncation_examples = find_truncation_examples(solver.game_tree)
    print(f"Found {len(truncation_examples)} nodes with truncation:")
    for example in truncation_examples[:3]:  # Show first 3
        print(f"  {example}")

    # Concept 3: Threat-based incentives
    print("\nCONCEPT 3: Threat-Based Incentives")
    threat_examples = analyze_threat_structure(solver.game_tree)
    print(f"Threat analysis: {threat_examples}")

    return solver, mdp


def count_nodes(node: GameNode) -> int:
    """Count total nodes in tree"""
    count = 1
    for child in node.children.values():
        count += count_nodes(child)
    return count


def count_terminal_nodes(node: GameNode) -> int:
    """Count terminal nodes in tree"""
    if node.is_terminal:
        return 1
    count = 0
    for child in node.children.values():
        count += count_terminal_nodes(child)
    return count


def find_interesting_node(node: GameNode, depth: int = 0) -> GameNode:
    """Find a non-terminal node with multiple children for demonstration"""
    if not node.is_terminal and len(node.children) > 1 and depth > 0:
        return node

    for child in node.children.values():
        result = find_interesting_node(child, depth + 1)
        if result:
            return result
    return None


def find_mixed_strategy_nodes(node: GameNode, strategy: Dict, path: str = "", depth: int = 0) -> List[str]:
    """Find nodes where mixed strategies are used"""
    mixed_nodes = []

    state_key = f"depth_{depth}_player_{node.player}"
    if state_key in strategy:
        actions = strategy[state_key]
        if len(actions) > 1 and any(prob > 0.01 and prob < 0.99 for prob in actions.values()):
            action_str = ", ".join(f"{action.name}:{prob:.2f}" for action, prob in actions.items())
            mixed_nodes.append(f"Node {state_key}: {action_str} ({'Leader' if node.player == 0 else 'Follower'})")

    # Recursively check children
    for action, child in node.children.items():
        mixed_nodes.extend(find_mixed_strategy_nodes(child, strategy, path + f"→{action.name}", depth + 1))

    return mixed_nodes


def find_truncation_examples(node: GameNode, examples: List = None, depth: int = 0) -> List[str]:
    """Find examples of left truncation in EPFs"""
    if examples is None:
        examples = []

    if node.epf and node.epf.left_truncate_threshold > float('-inf'):
        player_name = 'Leader' if node.player == 0 else 'Follower'
        examples.append(f"Depth {depth} ({player_name}): Threshold={node.epf.left_truncate_threshold:.2f}, "
                        f"Points={len(node.epf.points)}")

    for child in node.children.values():
        find_truncation_examples(child, examples, depth + 1)

    return examples


def analyze_threat_structure(node: GameNode) -> Dict[str, int]:
    """Analyze the threat structure in the game tree"""
    analysis = {
        'follower_nodes': 0,
        'nodes_with_threats': 0,
        'max_threat_value': 0.0,
        'avg_children_per_node': 0.0
    }

    total_nodes = 0
    total_children = 0

    def analyze_node(n: GameNode):
        nonlocal total_nodes, total_children
        total_nodes += 1
        total_children += len(n.children)

        if n.player == 1:  # Follower node
            analysis['follower_nodes'] += 1

            if n.epf and n.epf.left_truncate_threshold > float('-inf'):
                analysis['nodes_with_threats'] += 1
                analysis['max_threat_value'] = max(analysis['max_threat_value'],
                                                   n.epf.left_truncate_threshold)

        for child in n.children.values():
            analyze_node(child)

    analyze_node(node)

    if total_nodes > 0:
        analysis['avg_children_per_node'] = total_children / total_nodes

    return analysis


def visualize_epf_evolution(solver: ForestEPFSolver):
    """Visualize how EPFs evolve up the tree"""
    print("\n7. EPF EVOLUTION VISUALIZATION")

    # Find nodes at different depths
    nodes_by_depth = collect_nodes_by_depth(solver.game_tree)

    # Create subplots for different depths
    max_depth = min(3, max(nodes_by_depth.keys()) if nodes_by_depth else 0)

    if max_depth > 0:
        fig, axes = plt.subplots(1, max_depth + 1, figsize=(15, 4))
        if max_depth == 0:
            axes = [axes]

        for depth in range(max_depth + 1):
            if depth in nodes_by_depth and nodes_by_depth[depth]:
                # Take first node at this depth with non-empty EPF
                node = None
                for n in nodes_by_depth[depth]:
                    if n.epf and n.epf.points:
                        node = n
                        break

                if node:
                    epf = node.epf
                    mu2_values = [p.follower_payoff for p in epf.points]
                    u1_values = [p.leader_payoff for p in epf.points]

                    axes[depth].plot(mu2_values, u1_values, 'b-o', linewidth=2, markersize=4)
                    axes[depth].axvline(x=epf.left_truncate_threshold, color='r', linestyle='--', alpha=0.7)
                    axes[depth].set_title(f'Depth {depth}\n({'Leader' if node.player == 0 else 'Follower'})')
                    axes[depth].set_xlabel('Follower Payoff')
                    axes[depth].set_ylabel('Leader Payoff')
                    axes[depth].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle('EPF Evolution by Tree Depth', y=1.02)
        plt.show()


def collect_nodes_by_depth(node: GameNode, depth: int = 0, collection: Dict = None) -> Dict[int, List[GameNode]]:
    """Collect nodes organized by depth"""
    if collection is None:
        collection = {}

    if depth not in collection:
        collection[depth] = []
    collection[depth].append(node)

    for child in node.children.values():
        collect_nodes_by_depth(child, depth + 1, collection)

    return collection


def demonstrate_mixed_strategy_mechanics():
    """Demonstrate the core insight about mixed strategies"""
    print("\n8. MIXED STRATEGY MECHANICS DEEP DIVE")

    print("\nKEY INSIGHT: In two-player games, optimal mixed strategies")
    print("mix between AT MOST 2 actions at any decision node.")
    print("\nThis is because:")
    print("1. EPFs are piecewise linear concave functions")
    print("2. Any point on EPF = convex combination of ≤ 2 vertices")
    print("3. More than 2 children would be suboptimal")

    # Create a simple example to demonstrate
    print("\nExample: Leader node with 3 children")

    # Create example EPF points for 3 children
    child1_epf = EPF([EPFPoint(0, 10), EPFPoint(5, 8)])  # High leader payoff, low follower
    child2_epf = EPF([EPFPoint(3, 6), EPFPoint(8, 4)])  # Medium payoffs
    child3_epf = EPF([EPFPoint(6, 2), EPFPoint(10, 0)])  # High follower, low leader

    print("Child 1 EPF: High leader payoff, low follower payoff")
    print("Child 2 EPF: Medium payoffs for both")
    print("Child 3 EPF: High follower payoff, low leader payoff")

    # Compute envelope
    envelope = child1_epf.upper_concave_envelope(child2_epf).upper_concave_envelope(child3_epf)

    print(f"\nResulting envelope has {len(envelope.points)} points")
    print("Only points on the upper boundary matter for mixing!")

    return envelope


def demonstrate_left_truncation():
    """Demonstrate left truncation mechanism"""
    print("\n9. LEFT TRUNCATION MECHANISM")

    print("\nLeft truncation implements incentive compatibility:")
    print("If leader promises follower payoff < τ(s'), follower will deviate")
    print("τ(s') = max over alternative actions of worst payoff leader can inflict")

    # Example with follower node
    print("\nExample: Follower choosing between 3 actions")

    # Child EPFs representing outcomes of follower's choices
    action_a_epf = EPF([EPFPoint(2, 8), EPFPoint(4, 6)])  # Low follower payoff
    action_b_epf = EPF([EPFPoint(6, 4), EPFPoint(8, 2)])  # High follower payoff
    action_c_epf = EPF([EPFPoint(1, 9), EPFPoint(3, 7)])  # Very low follower payoff

    print("Action A outcomes: Follower gets 2-4, Leader gets 6-8")
    print("Action B outcomes: Follower gets 6-8, Leader gets 2-4")
    print("Action C outcomes: Follower gets 1-3, Leader gets 7-9")

    # Compute truncation thresholds
    # For Action A: τ = max(min(B), min(C)) = max(6, 1) = 6
    # For Action B: τ = max(min(A), min(C)) = max(2, 1) = 2
    # For Action C: τ = max(min(A), min(B)) = max(2, 6) = 6

    print("\nTruncation thresholds:")
    print("For Action A: τ = 6 (follower needs ≥6 to not choose B)")
    print("For Action B: τ = 2 (follower needs ≥2 to not choose A)")
    print("For Action C: τ = 6 (follower needs ≥6 to not choose B)")

    print("\nAfter truncation:")
    print("Action A: No valid points (all < 6)")
    print("Action B: All points valid (≥2)")
    print("Action C: No valid points (all < 6)")

    print("Result: Only Action B remains feasible!")


def run_complete_demo():
    """Run the complete EPF solver demonstration"""
    print("🌲 FOREST GAME EPF SOLVER - COMPLETE DEMONSTRATION 🌲")

    # Main demonstration
    solver, mdp = demonstrate_epf_computation()

    # Visualize EPF evolution
    visualize_epf_evolution(solver)

    # Demonstrate core mechanics
    envelope = demonstrate_mixed_strategy_mechanics()

    # Demonstrate left truncation
    demonstrate_left_truncation()

    print("\n" + "=" * 60)
    print("SUMMARY OF KEY EPF CONCEPTS DEMONSTRATED:")
    print("=" * 60)

    print("\n✅ PHASE 1: EPF Computation")
    print("  - Backward induction from leaves to root")
    print("  - Leader nodes: Upper concave envelope of children")
    print("  - Follower nodes: Left truncation + envelope")
    print("  - Mixed strategies between ≤2 children only")

    print("\n✅ PHASE 2: Strategy Extraction")
    print("  - One-step lookahead from optimal root point")
    print("  - Mixed strategies when multiple children viable")
    print("  - Pure follower responses (incentive compatible)")

    print("\n✅ INCENTIVE COMPATIBILITY")
    print("  - Left truncation threshold τ(s') prevents deviation")
    print("  - Follower gets ≥ best punishment from alternatives")
    print("  - Leader commits to credible threats")

    print("\n✅ STACKELBERG POWER")
    print("  - Leader commits first, gains first-mover advantage")
    print("  - Can threaten follower with punishment strategies")
    print("  - Achieves better payoff than Nash equilibrium")

    print(f"\n🎯 Final solution: Leader={solver.game_tree.epf.get_maximum_point()[1]:.2f}, "
          f"Follower={solver.game_tree.epf.get_maximum_point()[0]:.2f}")

    return solver


if __name__ == "__main__":
    solver = run_complete_demo()