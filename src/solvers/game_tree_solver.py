"""
Game Tree Solver for Sequential Forest Collection Game with Stackelberg Threats

This module implements minimax with alpha-beta pruning for the turn-based forest
collection game, incorporating credible threat mechanisms where the leader can
commit to punishment strategies that lead to mutual losses.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict
import copy
from matplotlib import pyplot as plt

# Assume the updated forest game is imported
from src.forest_game import ForestCollectionMDP, GameState, Action


@dataclass
class GameNode:
    state: GameState
    depth: int
    is_leader_turn: bool
    parent: Optional['GameNode'] = None
    children: List['GameNode'] = None
    best_action: Optional[Action] = None
    value: Optional[float] = None
    alpha: float = float('-inf')
    beta: float = float('inf')

    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class ThreatStrategy:
    """Defines a threat strategy for the leader"""
    name: str
    threat_condition: callable  # Function that determines when to threaten
    punishment_action: callable  # Function that determines punishment action
    cooperation_reward: float  # Bonus for cooperation
    punishment_cost: float  # Cost of punishment to leader


class SequentialGameTreeSolver:
    """
    Game tree solver for sequential forest collection with Stackelberg threats

    Uses minimax with alpha-beta pruning and incorporates credible threats
    where the leader can commit to punishment strategies.
    """

    def __init__(self,
                 mdp: ForestCollectionMDP,
                 max_depth: int = 20,
                 enable_pruning: bool = True,
                 enable_caching: bool = True):

        self.mdp = mdp
        self.max_depth = max_depth
        self.enable_pruning = enable_pruning
        self.enable_caching = enable_caching

        # Caching and statistics
        self.transposition_table = {}  # State -> (value, best_action, depth)
        self.nodes_evaluated = 0
        self.cache_hits = 0
        self.pruned_branches = 0

        # Threat mechanisms
        self.threat_strategies = self._initialize_threat_strategies()
        self.current_threat = None

        # Game tree root
        self.game_tree_root = None

    def _initialize_threat_strategies(self) -> Dict[str, ThreatStrategy]:
        """Initialize different threat strategies the leader can use"""

        def low_cooperation_condition(state: GameState, history: List[GameState]) -> bool:
            """Check if follower is being uncooperative"""
            if len(history) < 4:
                return False

            # Check if follower has been avoiding high-fruit areas that also have decent wood
            cooperative_moves = 0
            for i in range(len(history) - 3, len(history)):
                prev_state = history[i]
                if not prev_state.turn:  # Follower's turn
                    pos = prev_state.follower_pos
                    wood, fruit = self.mdp.get_cell_rewards(pos)
                    # Cooperative if going to cells that benefit both players
                    if wood >= 3 and fruit >= 5:
                        cooperative_moves += 1

            return cooperative_moves < 1  # Less than 1 cooperative move in last 3

        def fruit_avoidance_punishment(state: GameState) -> Action:
            """Choose action that avoids high-fruit areas"""
            current_pos = state.leader_pos
            valid_actions = self.mdp.get_valid_actions(current_pos)

            best_action = Action.STAY
            lowest_fruit = float('inf')
            highest_wood = -1

            for action in valid_actions:
                new_pos = self.mdp.apply_action(current_pos, action)
                wood, fruit = self.mdp.get_cell_rewards(new_pos)

                # Prioritize low fruit areas with decent wood
                if fruit < lowest_fruit or (fruit == lowest_fruit and wood > highest_wood):
                    lowest_fruit = fruit
                    highest_wood = wood
                    best_action = action

            return best_action

        def blocking_punishment(state: GameState) -> Action:
            """Choose action that blocks follower's access to high-fruit areas"""
            current_pos = state.leader_pos
            follower_pos = state.follower_pos
            valid_actions = self.mdp.get_valid_actions(current_pos)

            # Find high-fruit areas
            high_fruit_areas = []
            for x in range(self.mdp.width):
                for y in range(self.mdp.height):
                    if self.mdp.forest_map[x, y, 1] > 7:  # High fruit threshold
                        high_fruit_areas.append((x, y))

            if not high_fruit_areas:
                return fruit_avoidance_punishment(state)

            # Choose action that gets closer to blocking path to high-fruit areas
            best_action = Action.STAY
            min_distance_to_block = float('inf')

            for action in valid_actions:
                new_pos = self.mdp.apply_action(current_pos, action)

                # Calculate minimum distance to any high-fruit area
                min_dist_to_fruit = min(
                    abs(new_pos[0] - fruit_area[0]) + abs(new_pos[1] - fruit_area[1])
                    for fruit_area in high_fruit_areas
                )

                if min_dist_to_fruit < min_distance_to_block:
                    min_distance_to_block = min_dist_to_fruit
                    best_action = action

            return best_action

        strategies = {
            'no_threat': ThreatStrategy(
                name='no_threat',
                threat_condition=lambda s, h: False,
                punishment_action=lambda s: self._greedy_leader_action(s),
                cooperation_reward=0.0,
                punishment_cost=0.0
            ),

            'fruit_avoidance': ThreatStrategy(
                name='fruit_avoidance',
                threat_condition=low_cooperation_condition,
                punishment_action=fruit_avoidance_punishment,
                cooperation_reward=2.0,
                punishment_cost=3.0
            ),

            'blocking': ThreatStrategy(
                name='blocking',
                threat_condition=low_cooperation_condition,
                punishment_action=blocking_punishment,
                cooperation_reward=1.5,
                punishment_cost=2.0
            ),
        }

        return strategies

    def _greedy_leader_action(self, state: GameState) -> Action:
        """Get greedy action for leader (maximize wood collection)"""
        current_pos = state.leader_pos
        valid_actions = self.mdp.get_valid_actions(current_pos)

        best_action = Action.STAY
        best_wood = -1

        for action in valid_actions:
            new_pos = self.mdp.apply_action(current_pos, action)
            wood, _ = self.mdp.get_cell_rewards(new_pos)

            if wood > best_wood:
                best_wood = wood
                best_action = action

        return best_action

    def solve_with_threats(self,
                           threat_strategy: str = 'fruit_avoidance',
                           leader_starts: bool = True) -> Tuple[float, float, List[Action]]:
        """
        Solve the game with a specific threat strategy

        Args:
            threat_strategy: Name of threat strategy to use
            leader_starts: Whether leader moves first

        Returns:
            (leader_payoff, follower_payoff, optimal_action_sequence)
        """

        if threat_strategy not in self.threat_strategies:
            raise ValueError(f"Unknown threat strategy: {threat_strategy}")

        self.current_threat = self.threat_strategies[threat_strategy]

        # Reset statistics
        self.nodes_evaluated = 0
        self.cache_hits = 0
        self.pruned_branches = 0
        self.transposition_table.clear()

        # Create initial state with correct turn
        initial_state = self.mdp.get_initial_state()
        initial_state = GameState(
            leader_pos=initial_state.leader_pos,
            follower_pos=initial_state.follower_pos,
            leader_steps_left=initial_state.leader_steps_left,
            follower_steps_left=initial_state.follower_steps_left,
            leader_total_wood=initial_state.leader_total_wood,
            follower_total_fruit=initial_state.follower_total_fruit,
            turn=leader_starts  # True = leader's turn, False = follower's turn
        )

        # Build and solve game tree
        root_node = GameNode(state=initial_state, depth=0, is_leader_turn=leader_starts)
        self.game_tree_root = root_node

        start_time = time.time()
        value = self._minimax_with_threats(root_node, [], alpha=float('-inf'), beta=float('inf'))
        solve_time = time.time() - start_time

        # Extract solution path
        optimal_actions = self._extract_optimal_path(root_node)

        # Simulate to get final payoffs
        final_leader_payoff, final_follower_payoff = self._simulate_game(optimal_actions, initial_state)

        print(f"\nSolution Statistics ({threat_strategy}):")
        print(f"  Solve time: {solve_time:.3f}s")
        print(f"  Nodes evaluated: {self.nodes_evaluated}")
        print(f"  Cache hits: {self.cache_hits}")
        print(f"  Pruned branches: {self.pruned_branches}")
        print(f"  Game tree depth: {self._compute_tree_depth(root_node)}")
        print(f"  Final payoffs: Leader={final_leader_payoff:.2f}, Follower={final_follower_payoff:.2f}")

        return final_leader_payoff, final_follower_payoff, optimal_actions

    def _minimax_with_threats(self,
                              node: GameNode,
                              history: List[GameState],
                              alpha: float = float('-inf'),
                              beta: float = float('inf')) -> float:
        """
        Minimax algorithm with alpha-beta pruning and threat handling
        """

        self.nodes_evaluated += 1

        # Check cache
        if self.enable_caching:
            cache_key = self._get_cache_key(node.state, len(history))
            if cache_key in self.transposition_table:
                cached_value, cached_action, cached_depth = self.transposition_table[cache_key]
                if cached_depth >= node.depth:  # Use cached result if it's from deeper search
                    node.best_action = cached_action
                    node.value = cached_value
                    self.cache_hits += 1
                    return cached_value

        # Terminal conditions
        if node.depth >= self.max_depth or node.state.is_terminal():
            value = self._evaluate_terminal_state(node.state, history)
            node.value = value
            return value

        # Determine if this is a threat situation
        is_threatening = (self.current_threat and
                          self.current_threat.threat_condition(node.state, history))

        if node.is_leader_turn:
            # Leader's turn - maximize leader's utility
            max_eval = float('-inf')
            best_action = None

            valid_actions = self.mdp.get_valid_actions(node.state.leader_pos)

            # If threatening, leader must follow threat strategy
            if is_threatening:
                threat_action = self.current_threat.punishment_action(node.state)
                if threat_action in valid_actions:
                    valid_actions = [threat_action]  # Force threat action

            for action in valid_actions:
                child_state = self._transition_with_turn_switch(node.state, action)
                child_node = GameNode(
                    state=child_state,
                    depth=node.depth + 1,
                    is_leader_turn=False,
                    parent=node
                )
                node.children.append(child_node)

                new_history = history + [node.state]
                eval_score = self._minimax_with_threats(child_node, new_history, alpha, beta)

                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action

                if self.enable_pruning:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        self.pruned_branches += 1
                        break

            node.value = max_eval
            node.best_action = best_action

        else:
            # Follower's turn - maximize follower's utility, but aware of leader's threats
            max_eval = float('-inf')
            best_action = None

            valid_actions = self.mdp.get_valid_actions(node.state.follower_pos)

            for action in valid_actions:
                child_state = self._transition_with_turn_switch(node.state, action)
                child_node = GameNode(
                    state=child_state,
                    depth=node.depth + 1,
                    is_leader_turn=True,
                    parent=node
                )
                node.children.append(child_node)

                new_history = history + [node.state]

                # Follower evaluates from their perspective, but considers threat consequences
                follower_value = self._evaluate_follower_action_with_threats(
                    child_node, new_history, alpha, beta
                )

                if follower_value > max_eval:
                    max_eval = follower_value
                    best_action = action

                if self.enable_pruning:
                    beta = min(beta, follower_value)
                    if beta <= alpha:
                        self.pruned_branches += 1
                        break

            node.value = max_eval
            node.best_action = best_action

        # Cache result
        if self.enable_caching:
            cache_key = self._get_cache_key(node.state, len(history))
            self.transposition_table[cache_key] = (node.value, node.best_action, node.depth)

        return node.value

    def _evaluate_follower_action_with_threats(self,
                                               child_node: GameNode,
                                               history: List[GameState],
                                               alpha: float,
                                               beta: float) -> float:
        """
        Evaluate follower's action considering potential leader threats
        """

        # Get the raw minimax value
        raw_value = self._minimax_with_threats(child_node, history, alpha, beta)

        # If there's no active threat, return raw value
        if not self.current_threat or self.current_threat.name == 'no_threat':
            return raw_value

        # Check if this action would trigger leader's threat in future
        future_threat_probability = self._estimate_threat_probability(child_node.state, history)

        # Adjust follower's valuation based on threat risk
        threat_penalty = future_threat_probability * self.current_threat.punishment_cost
        cooperation_bonus = (1 - future_threat_probability) * self.current_threat.cooperation_reward

        adjusted_value = raw_value - threat_penalty + cooperation_bonus

        return adjusted_value

    def _estimate_threat_probability(self, state: GameState, history: List[GameState]) -> float:
        """
        Estimate probability that leader will execute threat given current trajectory
        """

        if not self.current_threat:
            return 0.0

        # Simple heuristic: if follower keeps avoiding cooperation, threat probability increases
        recent_history = history[-3:] if len(history) >= 3 else history

        uncooperative_moves = 0
        for hist_state in recent_history:
            if not hist_state.turn:  # Follower's move
                pos = hist_state.follower_pos
                wood, fruit = self.mdp.get_cell_rewards(pos)

                # Uncooperative if going to high-fruit, low-wood areas
                if fruit > 6 and wood < 3:
                    uncooperative_moves += 1

        if len(recent_history) == 0:
            return 0.1  # Low baseline threat probability

        threat_prob = min(0.8, uncooperative_moves / len(recent_history))
        return threat_prob

    def _transition_with_turn_switch(self, state: GameState, action: Action) -> GameState:

        if state.turn:  # Leader's turn
            new_leader_pos = self.mdp.apply_action(state.leader_pos, action)
            leader_wood, _ = self.mdp.get_cell_rewards(new_leader_pos)

            new_state = GameState(
                leader_pos=new_leader_pos,
                follower_pos=state.follower_pos,
                leader_steps_left=state.leader_steps_left - 1,
                follower_steps_left=state.follower_steps_left,
                leader_total_wood=state.leader_total_wood + leader_wood,
                follower_total_fruit=state.follower_total_fruit,
                turn=False  # Switch to follower's turn
            )
        else:  # Follower's turn
            new_follower_pos = self.mdp.apply_action(state.follower_pos, action)
            _, follower_fruit = self.mdp.get_cell_rewards(new_follower_pos)

            new_state = GameState(
                leader_pos=state.leader_pos,
                follower_pos=new_follower_pos,
                leader_steps_left=state.leader_steps_left,
                follower_steps_left=state.follower_steps_left - 1,
                leader_total_wood=state.leader_total_wood,
                follower_total_fruit=state.follower_total_fruit + follower_fruit,
                turn=True  # Switch to leader's turn
            )

        return new_state

    def _evaluate_terminal_state(self, state: GameState, history: List[GameState]) -> float:
        """
        Evaluate terminal state considering both players' utilities and threat effects
        """

        leader_payoff = self.mdp.get_leader_reward(state)
        follower_payoff = self.mdp.get_follower_reward(state)

        # For Stackelberg equilibrium, we primarily optimize leader's payoff
        # but consider the overall game outcome

        if not self.current_threat or self.current_threat.name == 'no_threat':
            return leader_payoff

        # Check if threats were executed (mutual punishment)
        threat_was_executed = any(
            self.current_threat.threat_condition(hist_state, history[:i])
            for i, hist_state in enumerate(history)
        )

        if threat_was_executed:
            # Both players suffer when threats are executed
            punishment_cost = self.current_threat.punishment_cost
            return leader_payoff - punishment_cost
        else:
            # Reward successful cooperation
            cooperation_bonus = self.current_threat.cooperation_reward
            return leader_payoff + cooperation_bonus * (follower_payoff / 100)  # Scale bonus

    def _get_cache_key(self, state: GameState, history_length: int) -> Tuple:
        """Generate cache key for transposition table"""
        return (state.to_key(), history_length, self.current_threat.name if self.current_threat else 'none')

    def _extract_optimal_path(self, root: GameNode) -> List[Action]:
        """Extract optimal action sequence from solved game tree"""
        path = []
        current = root

        while current and current.best_action is not None and current.children:
            path.append(current.best_action)
            # Find child corresponding to best action
            for child in current.children:
                if child.parent == current:  # This should be the chosen child
                    current = child
                    break
            else:
                break

        return path

    def _simulate_game(self, actions: List[Action], initial_state: GameState) -> Tuple[float, float]:
        """Simulate game with given action sequence"""
        current_state = initial_state

        for i, action in enumerate(actions):
            if current_state.is_terminal():
                break
            current_state = self._transition_with_turn_switch(current_state, action)

        return self.mdp.get_leader_reward(current_state), self.mdp.get_follower_reward(current_state)

    def _compute_tree_depth(self, node: GameNode) -> int:
        """Compute maximum depth of game tree"""
        if not node.children:
            return node.depth

        return max(self._compute_tree_depth(child) for child in node.children)

    def compare_threat_strategies(self,
                                  leader_starts: bool = True) -> Dict[str, Tuple[float, float]]:
        """
        Compare different threat strategies and return results
        """
        results = {}

        print("Comparing Threat Strategies")
        print("=" * 50)

        for strategy_name in self.threat_strategies.keys():
            print(f"\nTesting strategy: {strategy_name}")
            try:
                leader_payoff, follower_payoff, actions = self.solve_with_threats(
                    threat_strategy=strategy_name,
                    leader_starts=leader_starts
                )

                results[strategy_name] = (leader_payoff, follower_payoff)
                print(f"  Result: Leader={leader_payoff:.2f}, Follower={follower_payoff:.2f}")

            except Exception as e:
                print(f"  Error: {e}")
                results[strategy_name] = (0.0, 0.0)

        # Find best strategy for leader
        best_strategy = max(results.items(), key=lambda x: x[1][0])
        print(f"\nBest strategy for leader: {best_strategy[0]} "
              f"(Leader={best_strategy[1][0]:.2f}, Follower={best_strategy[1][1]:.2f})")

        return results

    def visualize_solution(self, actions: List[Action], initial_state: GameState):
        """Visualize the solution path on the forest grid"""

        # Simulate to get full trajectory
        states = [initial_state]
        current_state = initial_state

        for action in actions:
            if current_state.is_terminal():
                break
            current_state = self._transition_with_turn_switch(current_state, action)
            states.append(current_state)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Forest with paths
        wood_map = self.mdp.forest_map[:, :, 0].T
        fruit_map = self.mdp.forest_map[:, :, 1].T

        # Show combined resource map
        combined_map = wood_map + fruit_map
        im = ax1.imshow(combined_map, cmap='YlOrBr', origin='lower', alpha=0.7)

        # Extract leader and follower paths
        leader_path = [(state.leader_pos[0], state.leader_pos[1]) for state in states]
        follower_path = [(state.follower_pos[0], state.follower_pos[1]) for state in states]

        # Plot paths
        leader_x, leader_y = zip(*leader_path)
        follower_x, follower_y = zip(*follower_path)

        ax1.plot(leader_x, leader_y, 'b-o', linewidth=3, markersize=8,
                 label='Leader Path', alpha=0.8)
        ax1.plot(follower_x, follower_y, 'r-s', linewidth=3, markersize=8,
                 label='Follower Path', alpha=0.8)

        # Mark starting positions
        ax1.plot(leader_x[0], leader_y[0], 'bo', markersize=12, label='Leader Start')
        ax1.plot(follower_x[0], follower_y[0], 'ro', markersize=12, label='Follower Start')

        ax1.set_title(f'Game Solution with {self.current_threat.name if self.current_threat else "No"} Threat')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Payoffs over time
        leader_payoffs = [state.leader_total_wood for state in states]
        follower_payoffs = [state.follower_total_fruit for state in states]

        steps = list(range(len(states)))
        ax2.plot(steps, leader_payoffs, 'b-o', linewidth=2, label='Leader (Wood)')
        ax2.plot(steps, follower_payoffs, 'r-s', linewidth=2, label='Follower (Fruit)')

        ax2.set_xlabel('Game Step')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Payoff Accumulation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


# Convenience functions
def solve_sequential_stackelberg(forest_mdp: ForestCollectionMDP,
                                 threat_strategy: str = 'fruit_avoidance',
                                 max_depth: int = 15) -> Tuple[float, float, List[Action]]:
    """
    Convenience function to solve sequential Stackelberg game
    """
    solver = SequentialGameTreeSolver(forest_mdp, max_depth=max_depth)
    return solver.solve_with_threats(threat_strategy=threat_strategy)


def analyze_all_threats(forest_mdp: ForestCollectionMDP) -> Dict[str, Tuple[float, float]]:
    """
    Analyze all available threat strategies
    """
    solver = SequentialGameTreeSolver(forest_mdp, max_depth=12)
    return solver.compare_threat_strategies()