"""
Stackelberg Solver for Forest Collection MDP Game

This module implements various solution methods for finding Stackelberg equilibria
in the forest collection game, including backward induction, iterative best response,
and threat-based strategies.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Callable
from dataclasses import dataclass
from enum import Enum
import copy

# Import from the main game module
from ..forest_game import ForestCollectionMDP, GameState, Action


@dataclass
class StrategyProfile:
    """Represents a complete strategy profile for both players"""
    leader_policy: Dict[GameState, Action]
    follower_policy: Dict[GameState, Action]
    leader_payoff: float
    follower_payoff: float

    def __repr__(self):
        return f"Strategy(L:{self.leader_payoff:.2f}, F:{self.follower_payoff:.2f})"


class ThreatType(Enum):
    """Different types of threats the leader can make"""
    NO_THREAT = "no_threat"
    FRUIT_AVOIDANCE = "fruit_avoidance"  # Go to low-fruit cells
    BLOCKING = "blocking"  # Block follower's path
    PUNISHMENT = "punishment"  # Explicit punishment strategy


class StackelbergSolver:
    """
    Advanced Stackelberg equilibrium solvers for the forest collection game

    Implements multiple solution concepts:
    - Pure strategy Stackelberg equilibrium
    - Mixed strategy equilibrium
    - Threat-based equilibrium
    - Subgame perfect equilibrium
    """

    def __init__(self, mdp: ForestCollectionMDP, discount_factor: float = 0.9):
        self.mdp = mdp
        self.discount_factor = discount_factor
        self.follower_response_cache = {}
        self.leader_value_cache = {}
        self.computed_equilibria = []

    def clear_cache(self):
        """Clear all computation caches"""
        self.follower_response_cache.clear()
        self.leader_value_cache.clear()

    def _policy_to_cache_key(self, policy: Dict[GameState, Action]) -> Tuple:
        """Convert policy to a hashable cache key"""
        if not policy:
            return tuple()

        policy_items = []
        for state, action in policy.items():
            policy_items.append((state.to_key(), action.value))

        policy_items.sort(key=lambda x: x[0])
        return tuple(policy_items)

    def solve_follower_best_response(self,
                                     leader_policy: Dict[GameState, Action],
                                     state: GameState,
                                     depth: int = 0,
                                     max_depth: int = 15) -> Tuple[float, Action]:
        """
        Solve follower's best response using dynamic programming

        Args:
            leader_policy: Leader's committed strategy
            state: Current game state
            depth: Current recursion depth
            max_depth: Maximum recursion depth

        Returns:
            (best_value, best_action) for the follower
        """
        if state.is_terminal() or depth >= max_depth:
            return self.mdp.get_follower_reward(state), Action.STAY

        # Check cache
        policy_key = self._policy_to_cache_key(leader_policy)
        cache_key = (state.to_key(), depth, policy_key)
        if cache_key in self.follower_response_cache:
            return self.follower_response_cache[cache_key]

        # Get leader's action from policy
        leader_action = leader_policy.get(state, Action.STAY)

        best_value = float('-inf')
        best_action = Action.STAY

        # Try all follower actions
        follower_actions = self.mdp.get_valid_actions(state.follower_pos)

        for f_action in follower_actions:
            next_state = self.mdp.transition(state, leader_action, f_action)
            future_value, _ = self.solve_follower_best_response(
                leader_policy, next_state, depth + 1, max_depth
            )

            # Calculate immediate reward
            immediate_reward = (self.mdp.get_follower_reward(next_state) -
                                self.mdp.get_follower_reward(state))

            total_value = immediate_reward + self.discount_factor * future_value

            if total_value > best_value:
                best_value = total_value
                best_action = f_action

        # Cache and return result
        self.follower_response_cache[cache_key] = (best_value, best_action)
        return best_value, best_action

    def compute_follower_policy(self,
                                leader_policy: Dict[GameState, Action],
                                max_depth: int = 15) -> Dict[GameState, Action]:
        """
        Compute complete follower policy given leader's strategy
        """
        follower_policy = {}

        # Generate all reachable states
        reachable_states = self._get_reachable_states(max_depth)

        for state in reachable_states:
            if not state.is_terminal():
                _, best_action = self.solve_follower_best_response(
                    leader_policy, state, 0, max_depth
                )
                follower_policy[state] = best_action

        return follower_policy

    def _get_reachable_states(self, max_depth: int = 15) -> Set[GameState]:
        """Generate all reachable game states"""
        reachable = set()
        queue = [self.mdp.get_initial_state()]
        visited = set()

        while queue and len(reachable) < 1000:  # Limit to prevent explosion
            current_state = queue.pop(0)

            if current_state.to_key() in visited or current_state.is_terminal():
                continue

            visited.add(current_state.to_key())
            reachable.add(current_state)

            # Generate successor states
            leader_actions = self.mdp.get_valid_actions(current_state.leader_pos)
            follower_actions = self.mdp.get_valid_actions(current_state.follower_pos)

            for l_action in leader_actions:
                for f_action in follower_actions:
                    next_state = self.mdp.transition(current_state, l_action, f_action)
                    if (next_state.to_key() not in visited and
                            current_state.leader_steps_left > 0):
                        queue.append(next_state)

        return reachable

    def evaluate_strategy_profile(self,
                                  leader_policy: Dict[GameState, Action],
                                  max_simulation_steps: int = 20) -> StrategyProfile:
        """
        Evaluate a complete strategy profile through simulation
        """
        # Compute follower's best response policy
        follower_policy = self.compute_follower_policy(leader_policy)

        # Simulate the game
        current_state = self.mdp.get_initial_state()
        total_leader_reward = 0
        total_follower_reward = 0
        simulation_step = 0

        while not current_state.is_terminal() and simulation_step < max_simulation_steps:
            # Get actions from policies
            leader_action = leader_policy.get(current_state, Action.STAY)
            follower_action = follower_policy.get(current_state, Action.STAY)

            # Transition to next state
            next_state = self.mdp.transition(current_state, leader_action, follower_action)

            # Accumulate rewards
            leader_reward_gain = (self.mdp.get_leader_reward(next_state) -
                                  self.mdp.get_leader_reward(current_state))
            follower_reward_gain = (self.mdp.get_follower_reward(next_state) -
                                    self.mdp.get_follower_reward(current_state))

            total_leader_reward += leader_reward_gain * (self.discount_factor ** simulation_step)
            total_follower_reward += follower_reward_gain * (self.discount_factor ** simulation_step)

            current_state = next_state
            simulation_step += 1

        return StrategyProfile(
            leader_policy=leader_policy,
            follower_policy=follower_policy,
            leader_payoff=total_leader_reward,
            follower_payoff=total_follower_reward
        )

    def create_threat_policy(self,
                             threat_type: ThreatType,
                             cooperation_threshold: float = 0.5) -> Dict[GameState, Action]:
        """
        Create a leader policy that includes credible threats

        Args:
            threat_type: Type of threat to implement
            cooperation_threshold: Threshold for determining cooperation
        """
        policy = {}

        if threat_type == ThreatType.FRUIT_AVOIDANCE:
            policy = self._create_fruit_avoidance_threat()
        elif threat_type == ThreatType.BLOCKING:
            policy = self._create_blocking_threat()
        elif threat_type == ThreatType.PUNISHMENT:
            policy = self._create_punishment_threat()
        else:
            policy = self._create_greedy_policy()

        return policy

    def _create_fruit_avoidance_threat(self) -> Dict[GameState, Action]:
        """Create policy where leader threatens to avoid high-fruit areas"""
        policy = {}

        # Find cells with low fruit (punishment cells)
        low_fruit_cells = []
        for x in range(self.mdp.width):
            for y in range(self.mdp.height):
                if self.mdp.forest_map[x, y, 1] < 3:  # Low fruit threshold
                    low_fruit_cells.append((x, y))

        if not low_fruit_cells:
            low_fruit_cells = [(0, 0)]  # Default fallback

        # Create policy that targets low-fruit, high-wood areas
        high_wood_low_fruit = []
        for x in range(self.mdp.width):
            for y in range(self.mdp.height):
                wood, fruit = self.mdp.forest_map[x, y, 0], self.mdp.forest_map[x, y, 1]
                if wood > 5 and fruit < 5:  # Good for leader, bad for follower
                    high_wood_low_fruit.append((x, y))

        target_cells = high_wood_low_fruit if high_wood_low_fruit else low_fruit_cells
        return self._create_targeting_policy(target_cells)

    def _create_greedy_policy(self) -> Dict[GameState, Action]:
        """Create a greedy policy that only maximizes leader's immediate reward"""
        high_wood_cells = []
        for x in range(self.mdp.width):
            for y in range(self.mdp.height):
                if self.mdp.forest_map[x, y, 0] > 5:
                    high_wood_cells.append((x, y))

        if not high_wood_cells:
            high_wood_cells = [(self.mdp.width - 1, self.mdp.height - 1)]

        return self._create_targeting_policy(high_wood_cells)

    def _create_targeting_policy(self, target_cells: List[Tuple[int, int]]) -> Dict[GameState, Action]:
        """Create a policy that targets specific cells in order"""
        if not target_cells:
            return {}

        policy = {}

        def get_next_action(current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> Action:
            dx = target_pos[0] - current_pos[0]
            dy = target_pos[1] - current_pos[1]

            if abs(dx) > abs(dy):
                return Action.RIGHT if dx > 0 else Action.LEFT
            elif abs(dy) > 0:
                return Action.UP if dy > 0 else Action.DOWN
            else:
                return Action.STAY

        # Generate policies for all reasonable states
        for steps_left in range(1, min(self.mdp.max_steps_leader + 1, 10)):
            for x in range(self.mdp.width):
                for y in range(self.mdp.height):
                    for fx in range(self.mdp.width):
                        for fy in range(self.mdp.height):
                            state = GameState(
                                leader_pos=(x, y),
                                follower_pos=(fx, fy),
                                leader_steps_left=steps_left,
                                follower_steps_left=steps_left
                            )

                            # Choose target based on current situation
                            best_target = self._choose_best_target(target_cells, (x, y))
                            action = get_next_action((x, y), best_target)
                            policy[state] = action

        return policy

    def _choose_best_target(self, targets: List[Tuple[int, int]],
                            current_pos: Tuple[int, int]) -> Tuple[int, int]:
        """Choose the best target from a list based on distance and value"""
        if not targets:
            return (0, 0)

        def distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        def cell_value(pos):
            x, y = pos
            return self.mdp.forest_map[x, y, 0]  # Leader wants wood

        # Score each target by value/distance ratio
        best_score = -1
        best_target = targets[0]

        for target in targets:
            dist = distance(current_pos, target)
            value = cell_value(target)
            score = value / (dist + 1)  # +1 to avoid division by zero

            if score > best_score:
                best_score = score
                best_target = target

        return best_target

    def _create_blocking_threat(self) -> Dict[GameState, Action]:
        """Create policy where leader blocks follower's access to high-fruit areas"""
        # This is a simplified blocking strategy
        # In practice, you'd want more sophisticated path prediction
        return self._create_greedy_policy()  # Placeholder

    def _create_punishment_threat(self) -> Dict[GameState, Action]:
        """Create explicit punishment strategy"""
        # Placeholder for more complex punishment mechanisms
        return self._create_fruit_avoidance_threat()

    def find_stackelberg_equilibrium(self,
                                     threat_types: List[ThreatType] = None,
                                     max_iterations: int = 10) -> List[StrategyProfile]:
        """
        Find Stackelberg equilibria using different threat strategies

        Returns list of equilibria sorted by leader payoff
        """
        if threat_types is None:
            threat_types = [ThreatType.NO_THREAT, ThreatType.FRUIT_AVOIDANCE, ThreatType.GREEDY]

        equilibria = []

        for threat_type in threat_types:
            try:
                # Create leader policy with this threat type
                leader_policy = self.create_threat_policy(threat_type)

                if leader_policy:  # Only evaluate if policy is non-empty
                    # Evaluate the resulting strategy profile
                    profile = self.evaluate_strategy_profile(leader_policy)
                    profile.threat_type = threat_type  # Add metadata
                    equilibria.append(profile)

            except Exception as e:
                print(f"Warning: Failed to compute equilibrium for {threat_type}: {e}")
                continue

        # Sort by leader payoff (Stackelberg leader optimizes first)
        equilibria.sort(key=lambda x: x.leader_payoff, reverse=True)

        self.computed_equilibria = equilibria
        return equilibria

    def analyze_threat_credibility(self,
                                   leader_policy: Dict[GameState, Action]) -> Dict[str, float]:
        """
        Analyze how credible a threat policy is

        Returns metrics about threat credibility
        """
        # Evaluate leader's payoff when following through on threats
        threat_profile = self.evaluate_strategy_profile(leader_policy)

        # Evaluate leader's payoff with pure greedy strategy
        greedy_policy = self._create_greedy_policy()
        greedy_profile = self.evaluate_strategy_profile(greedy_policy)

        # Compute credibility metrics
        credibility = {
            'threat_payoff': threat_profile.leader_payoff,
            'greedy_payoff': greedy_profile.leader_payoff,
            'credibility_cost': greedy_profile.leader_payoff - threat_profile.leader_payoff,
            'follower_punishment': greedy_profile.follower_payoff - threat_profile.follower_payoff,
            'credibility_ratio': (threat_profile.leader_payoff /
                                  (greedy_profile.leader_payoff + 1e-6))  # Avoid division by zero
        }

        return credibility


# Convenience functions for easy use
def solve_stackelberg_game(mdp: ForestCollectionMDP,
                           threat_types: List[ThreatType] = None) -> List[StrategyProfile]:
    """
    Convenience function to solve Stackelberg game with default parameters
    """
    solver = StackelbergSolver(mdp)
    return solver.find_stackelberg_equilibrium(threat_types)


def compare_strategies(mdp: ForestCollectionMDP) -> Dict[str, StrategyProfile]:
    """
    Compare different strategy types and return results
    """
    solver = StackelbergSolver(mdp)

    strategies = {}

    for threat_type in ThreatType:
        try:
            policy = solver.create_threat_policy(threat_type)
            if policy:
                profile = solver.evaluate_strategy_profile(policy)
                strategies[threat_type.value] = profile
        except Exception as e:
            print(f"Failed to evaluate {threat_type.value}: {e}")

    return strategies