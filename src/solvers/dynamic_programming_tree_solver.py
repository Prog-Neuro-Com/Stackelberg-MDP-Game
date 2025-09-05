"""
Letchford-Conitzer Algorithm Implementation for Forest Collection Game

This implements the exact algorithm from "Computing Optimal Strategies to Commit to in
Extensive-Form Games" (Theorem 1) for perfect-information games with pure strategy commitment.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from src.forest_game import ForestCollectionMDP, GameState, Action


@dataclass
class GameTreeNode:
    """Represents a node in the extensive-form game tree"""
    state: GameState
    current_player: int  # 0 = leader, 1 = follower
    children: Dict[Action, 'GameTreeNode']
    parent: Optional['GameTreeNode'] = None
    reachable_leaves: Set[int] = None  # S_v in the paper
    optimal_leaf: Optional[int] = None
    commitment_strategy: Dict[Action, float] = None

    def __post_init__(self):
        if self.children is None:
            self.children = {}
        if self.reachable_leaves is None:
            self.reachable_leaves = set()
        if self.commitment_strategy is None:
            self.commitment_strategy = {}


class LetchfordConitzerSolver:
    def __init__(self, mdp: ForestCollectionMDP, max_depth: int = 20):
        self.mdp = mdp
        self.max_depth = max_depth
        self.game_tree = None
        self.leaf_nodes = []  # All terminal nodes
        self.leaf_utilities = []  # (leader_utility, follower_utility) for each leaf

    def solve(self, initial_state: GameState, leader_starts: bool = True) -> Tuple[float, float, List[Action]]:
        """
        Main solving function implementing the LC algorithm

        Args:
            initial_state: Starting game state
            leader_starts: If True, leader moves first; if False, follower moves first

        Returns:
            (leader_payoff, follower_payoff, optimal_action_sequence)
        """
        print(f"Building game tree ({'Leader' if leader_starts else 'Follower'} starts)...")
        root = self._build_game_tree(initial_state, leader_starts)

        print(f"Game tree built with {len(self.leaf_nodes)} terminal nodes")
        print("Computing reachable sets (bottom-up pass)...")
        self._compute_reachable_sets(root)

        print("Finding optimal outcome...")
        optimal_leaf_idx = self._find_optimal_leaf(root)

        print("Computing commitment strategy (top-down pass)...")
        action_sequence = self._extract_strategy(root, optimal_leaf_idx)

        # Get final payoffs
        if optimal_leaf_idx < len(self.leaf_utilities):
            leader_payoff, follower_payoff = self.leaf_utilities[optimal_leaf_idx]
        else:
            leader_payoff = follower_payoff = 0.0

        print(f"Solution found: Leader={leader_payoff:.2f}, Follower={follower_payoff:.2f}")
        return leader_payoff, follower_payoff, action_sequence

    def _build_game_tree(self, initial_state: GameState, leader_starts: bool) -> GameTreeNode:
        """Build the extensive-form game tree"""
        self.leaf_nodes = []
        self.leaf_utilities = []

        root = GameTreeNode(
            state=initial_state,
            current_player=0 if leader_starts else 1,
            children={}
        )

        self._expand_node(root, depth=0)
        self.game_tree = root
        return root

    def _expand_node(self, node: GameTreeNode, depth: int):
        """Recursively expand game tree node"""
        # Terminal conditions
        if depth >= self.max_depth or self._is_terminal(node.state):
            # Terminal node - add to leaf list
            leaf_idx = len(self.leaf_nodes)
            self.leaf_nodes.append(node)

            leader_payoff = self.mdp.get_leader_reward(node.state)
            follower_payoff = self.mdp.get_follower_reward(node.state)
            self.leaf_utilities.append((leader_payoff, follower_payoff))

            # Terminal node is reachable by itself
            node.reachable_leaves = {leaf_idx}
            return

        # Check if current player has any steps left
        current_player_has_steps = (
            (node.current_player == 0 and node.state.leader_steps_left > 0) or
            (node.current_player == 1 and node.state.follower_steps_left > 0)
        )

        if not current_player_has_steps:
            # Current player has no steps left, skip to the other player
            new_state = GameState(
                leader_pos=node.state.leader_pos,
                follower_pos=node.state.follower_pos,
                leader_steps_left=node.state.leader_steps_left,
                follower_steps_left=node.state.follower_steps_left,
                leader_total_wood=node.state.leader_total_wood,
                follower_total_fruit=node.state.follower_total_fruit,
                turn=not node.state.turn  # Switch turn
            )

            child = GameTreeNode(
                state=new_state,
                current_player=1 - node.current_player,  # Switch players
                children={},
                parent=node
            )

            # Use a special "PASS" action to represent skipping the turn
            node.children[Action.STAY] = child
            self._expand_node(child, depth + 1)
            return

        # Get valid actions for current player
        if node.current_player == 0:  # Leader's turn
            valid_actions = self.mdp.get_valid_actions(node.state.leader_pos)
        else:  # Follower's turn
            valid_actions = self.mdp.get_valid_actions(node.state.follower_pos)

        # Expand each possible action
        for action in valid_actions:
            new_state = self._apply_action(node.state, action, node.current_player)

            child = GameTreeNode(
                state=new_state,
                current_player=1 - node.current_player,  # Switch players
                children={},
                parent=node
            )

            node.children[action] = child
            self._expand_node(child, depth + 1)

    def _is_terminal(self, state: GameState) -> bool:
        """Check if the game state is terminal"""
        return state.is_terminal()

    def _apply_action(self, state: GameState, action: Action, current_player: int) -> GameState:
        """Apply action for the current player and create new state"""
        if current_player == 0:  # Leader's action
            if state.leader_steps_left <= 0:
                # Leader has no steps left, just switch turn without taking action
                return GameState(
                    leader_pos=state.leader_pos,
                    follower_pos=state.follower_pos,
                    leader_steps_left=state.leader_steps_left,
                    follower_steps_left=state.follower_steps_left,
                    leader_total_wood=state.leader_total_wood,
                    follower_total_fruit=state.follower_total_fruit,
                    turn=False  # Switch to follower
                )
            
            new_leader_pos = self.mdp.apply_action(state.leader_pos, action)
            leader_wood, _ = self.mdp.get_cell_rewards(new_leader_pos)

            return GameState(
                leader_pos=new_leader_pos,
                follower_pos=state.follower_pos,
                leader_steps_left=state.leader_steps_left - 1,
                follower_steps_left=state.follower_steps_left,
                leader_total_wood=state.leader_total_wood + leader_wood,
                follower_total_fruit=state.follower_total_fruit,
                turn=False  # Switch to follower's turn
            )
        else:  # Follower's action
            if state.follower_steps_left <= 0:
                # Follower has no steps left, just switch turn without taking action
                return GameState(
                    leader_pos=state.leader_pos,
                    follower_pos=state.follower_pos,
                    leader_steps_left=state.leader_steps_left,
                    follower_steps_left=state.follower_steps_left,
                    leader_total_wood=state.leader_total_wood,
                    follower_total_fruit=state.follower_total_fruit,
                    turn=True  # Switch to leader
                )
            
            new_follower_pos = self.mdp.apply_action(state.follower_pos, action)
            _, follower_fruit = self.mdp.get_cell_rewards(new_follower_pos)

            return GameState(
                leader_pos=state.leader_pos,
                follower_pos=new_follower_pos,
                leader_steps_left=state.leader_steps_left,
                follower_steps_left=state.follower_steps_left - 1,
                leader_total_wood=state.leader_total_wood,
                follower_total_fruit=state.follower_total_fruit + follower_fruit,
                turn=True  # Switch to leader's turn
            )

    def _compute_reachable_sets(self, node: GameTreeNode):
        """
        Bottom-up computation of reachable sets (Algorithm from LC paper)

        For each node v, compute S_v = set of leaf nodes reachable by leader's commitment
        """
        # Post-order traversal - compute children first
        for child in node.children.values():
            self._compute_reachable_sets(child)

        if not node.children:
            # Leaf node - already handled in _expand_node
            return

        if node.current_player == 0:  # Leader's node
            # S_v = ⋃_{w child of v} S_w
            # Leader can reach any leaf reachable from any child
            node.reachable_leaves = set()
            for child in node.children.values():
                node.reachable_leaves.update(child.reachable_leaves)

        else:  # Follower's node
            # S_v = ⋃_{w child of v} {l ∈ S_w : max_{w'≠w} u_i(m(w')) ≤ u_i(l)}
            # Follower will only choose paths where payoff ≥ best punishment threat

            # First, compute m(w) for each child w
            child_punishments = {}  # child -> min utility in that subtree
            for action, child in node.children.items():
                if child.reachable_leaves:
                    min_utility = min(self.leaf_utilities[leaf_idx][1]
                                    for leaf_idx in child.reachable_leaves)
                    child_punishments[action] = min_utility
                else:
                    child_punishments[action] = 0.0

            # Find the best punishment threat: max_{w'≠w} u_i(m(w'))
            node.reachable_leaves = set()

            for action, child in node.children.items():
                # Find best punishment on other children
                other_punishments = [child_punishments[other_action]
                                   for other_action in child_punishments
                                   if other_action != action]

                if other_punishments:
                    best_threat = max(other_punishments)
                else:
                    best_threat = float('-inf')  # No alternative, so no threat

                # Include leaves from this child that satisfy the condition
                for leaf_idx in child.reachable_leaves:
                    follower_utility = self.leaf_utilities[leaf_idx][1]
                    if follower_utility >= best_threat:
                        node.reachable_leaves.add(leaf_idx)

    def _find_optimal_leaf(self, root: GameTreeNode) -> int:
        """Find the leaf that maximizes leader's utility"""
        if not root.reachable_leaves:
            return 0

        best_leaf = max(root.reachable_leaves,
                       key=lambda leaf_idx: self.leaf_utilities[leaf_idx][0])
        root.optimal_leaf = best_leaf
        return best_leaf

    def _extract_strategy(self, node: GameTreeNode, target_leaf: int) -> List[Action]:
        """
        Top-down strategy extraction (Algorithm from LC paper)

        Compute strategy(v, l) that specifies commitment to achieve leaf l from node v
        """
        if target_leaf not in node.reachable_leaves:
            return []

        path = []
        current_node = node

        while current_node.children and not self._is_terminal(current_node.state):
            if current_node.current_player == 0:  # Leader's turn
                # Find child w that is ancestor of target leaf
                chosen_action = None
                for action, child in current_node.children.items():
                    if target_leaf in child.reachable_leaves:
                        chosen_action = action
                        current_node = child
                        break

                if chosen_action is not None:
                    path.append(chosen_action)
                    # Store commitment strategy
                    if current_node.parent:
                        current_node.parent.commitment_strategy[chosen_action] = 1.0
                else:
                    break

            else:  # Follower's turn
                # Find child w that is ancestor of target leaf
                chosen_action = None
                chosen_child = None

                for action, child in current_node.children.items():
                    if target_leaf in child.reachable_leaves:
                        chosen_action = action
                        chosen_child = child
                        break

                if chosen_action is not None:
                    path.append(chosen_action)
                    current_node = chosen_child

                    # For other children, the leader must commit to punishment
                    for action, child in current_node.parent.children.items():
                        if action != chosen_action and child.reachable_leaves:
                            # Find punishment leaf: m(w') = argmin_{l ∈ S_w'} u_i(l)
                            punishment_leaf = min(child.reachable_leaves,
                                                key=lambda l: self.leaf_utilities[l][1])
                            # Recursively set punishment strategy (simplified)
                            self._set_punishment_strategy(child, punishment_leaf)
                else:
                    break

        return path

    def _set_punishment_strategy(self, node: GameTreeNode, punishment_leaf: int):
        """Set commitment strategy to achieve punishment leaf (simplified)"""
        if node.current_player == 0 and node.children:
            # Leader commits to action leading toward punishment
            for action, child in node.children.items():
                if punishment_leaf in child.reachable_leaves:
                    node.commitment_strategy[action] = 1.0
                    self._set_punishment_strategy(child, punishment_leaf)
                    break

    def get_commitment_strategy(self) -> Dict[str, Dict[Action, float]]:
        """Extract the complete commitment strategy from the solved tree"""
        if not self.game_tree:
            return {}

        strategy = {}
        self._collect_strategy(self.game_tree, strategy, [])
        return strategy

    def _collect_strategy(self, node: GameTreeNode, strategy: Dict, path: List):
        """Recursively collect commitment strategy from tree"""
        if node.current_player == 0 and node.commitment_strategy:
            state_key = f"depth_{len(path)}_player_0"
            strategy[state_key] = node.commitment_strategy.copy()

        for action, child in node.children.items():
            self._collect_strategy(child, strategy, path + [action])


def solve_forest_stackelberg(mdp: ForestCollectionMDP,
                           max_depth: int = 15) -> Tuple[float, float, List[Action]]:
    """
    Convenience function to solve forest collection game with LC algorithm

    Args:
        mdp: The forest collection MDP instance
        max_depth: Maximum tree depth for search
    Returns:
        (leader_payoff, follower_payoff, optimal_action_sequence)
    """
    solver = LetchfordConitzerSolver(mdp, max_depth=max_depth)
    initial_state = mdp.get_initial_state()
    return solver.solve(initial_state, leader_starts=initial_state.turn)


def compare_starting_players(mdp: ForestCollectionMDP,
                           max_depth: int = 15) -> Dict[str, Tuple[float, float, List[Action]]]:
    """
    Compare outcomes when leader vs follower starts first

    Returns:
        Dictionary with results for both starting conditions
    """
    results = {}

    print("Solving with Leader starting first...")
    leader_payoff_1, follower_payoff_1, actions_1 = solve_forest_stackelberg(
        mdp, max_depth
    )
    results['leader_starts'] = (leader_payoff_1, follower_payoff_1, actions_1)

    print("\nSolving with Follower starting first...")
    leader_payoff_2, follower_payoff_2, actions_2 = solve_forest_stackelberg(
        mdp, max_depth
    )
    results['follower_starts'] = (leader_payoff_2, follower_payoff_2, actions_2)

    print(f"\nComparison Results:")
    print(f"Leader starts:   Leader={leader_payoff_1:.2f}, Follower={follower_payoff_1:.2f}")
    print(f"Follower starts: Leader={leader_payoff_2:.2f}, Follower={follower_payoff_2:.2f}")
    print(f"Leader advantage: {leader_payoff_1 - leader_payoff_2:.2f}")
    print(f"Follower advantage: {follower_payoff_2 - follower_payoff_1:.2f}")

    return results


# Example usage
if __name__ == "__main__":
    # Example usage with both starting conditions
    # mdp = ForestCollectionMDP(...)

    # Test both starting players
    # results = compare_starting_players(mdp)

    # Or test specific starting condition
    # leader_payoff, follower_payoff, actions = solve_forest_stackelberg(mdp, leader_starts=True)
    # print(f"Leader starts first: Leader={leader_payoff}, Follower={follower_payoff}")
    # print(f"Action sequence: {actions}")

    # leader_payoff, follower_payoff, actions = solve_forest_stackelberg(mdp, leader_starts=False)
    # print(f"Follower starts first: Leader={leader_payoff}, Follower={follower_payoff}")
    # print(f"Action sequence: {actions}")
    pass