"""
Simple MDP Solver for Single-Player Reward Maximization

This module implements classic MDP solution algorithms including:
- Value Iteration
- Policy Iteration
- Q-Learning (for comparison)

Designed for single-player forest collection where the goal is to maximize
total reward (wood collection) over a finite or infinite horizon.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt

# Import from the main game module
from src.forest_game import ForestCollectionMDP, GameState, Action


@dataclass(frozen=True)
class SinglePlayerState:
    position: Tuple[int, int]
    steps_left: int
    total_reward: float = 0.0

    def to_key(self) -> Tuple:
        return (self.position, self.steps_left, self.total_reward)

    @classmethod
    def from_key(cls, key: Tuple):
        return cls(position = key[0],
                   steps_left = key[1],
                   total_reward = key[2])


    def is_terminal(self) -> bool:
        return self.steps_left <= 0


@dataclass
class MDPSolution:
    """Contains the solution to an MDP"""
    value_function: Dict[SinglePlayerState, float]
    policy: Dict[SinglePlayerState, Action]
    q_values: Dict[Tuple[SinglePlayerState, Action], float]
    convergence_history: List[float]
    algorithm_used: str
    iterations: int


class SinglePlayerMDP:
    """
    Single-player MDP version of the forest collection game

    Player navigates grid to maximize reward collection within step limit.
    """

    def __init__(self,
                 forest_mdp: ForestCollectionMDP,
                 resource_type: str = "wood",  # "wood", "fruit", or "both"
                 discount_factor: float = 0.95):

        self.forest_mdp = forest_mdp
        self.resource_type = resource_type
        self.discount_factor = discount_factor

        self.width = forest_mdp.width
        self.height = forest_mdp.height
        self.forest_map = forest_mdp.forest_map
        self.start_position = forest_mdp.leader_start
        self.max_steps = forest_mdp.max_steps_leader

    def get_initial_state(self) -> SinglePlayerState:
        return SinglePlayerState(
            position=self.start_position,
            steps_left=self.max_steps,
            total_reward=0.0
        )

    def get_reward(self, state: SinglePlayerState, action: Action) -> float:
        new_position = self._apply_action(state.position, action)
        x, y = new_position

        if self.resource_type == "wood":
            return float(self.forest_map[x, y, 0])
        elif self.resource_type == "fruit":
            return float(self.forest_map[x, y, 1])
        elif self.resource_type == "both":
            return float(self.forest_map[x, y, 0] + self.forest_map[x, y, 1])
        else:
            return 0.0

    def transition(self, state: SinglePlayerState, action: Action) -> SinglePlayerState:
        if state.is_terminal():
            return state

        new_position = self._apply_action(state.position, action)
        reward = self.get_reward(state, action)

        return SinglePlayerState(
            position=new_position,
            steps_left=state.steps_left - 1,
            total_reward=state.total_reward + reward
        )

    def _apply_action(self, position: Tuple[int, int], action: Action) -> Tuple[int, int]:
        x, y = position
        dx, dy = action.value

        new_x = max(0, min(self.width - 1, x + dx))
        new_y = max(0, min(self.height - 1, y + dy))

        return (new_x, new_y)

    def get_valid_actions(self, state: SinglePlayerState) -> List[Action]:
        """Get valid actions from current state"""
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    def get_all_states(self) -> Set[SinglePlayerState]:
        """Generate all possible states (for finite horizon problems)"""
        states = set()

        for x in range(self.width):
            for y in range(self.height):
                for steps in range(self.max_steps + 1):
                    # For simplicity, we don't enumerate all possible total_reward values
                    # Instead, we'll use position and steps_left as the key state features
                    state = SinglePlayerState(
                        position=(x, y),
                        steps_left=steps,
                        total_reward=0.0  # We'll track this separately
                    )
                    states.add(state)

        return states


class SimpleMDPSolver:
    """
    Classic MDP solver implementing value iteration and policy iteration
    """

    def __init__(self, mdp: SinglePlayerMDP, tolerance: float = 1e-6):
        self.mdp = mdp
        self.tolerance = tolerance
        self.solutions = {}

    def value_iteration(self, max_iterations: int = 1000) -> MDPSolution:
        """
        Solve MDP using value iteration algorithm

        Returns optimal value function and policy
        """
        print("Running Value Iteration...")

        # Initialize value function
        V = {}  # V[state] = value
        all_states = self._get_reachable_states()

        # Initialize all state values to 0
        for state in all_states:
            V[state] = 0.0

        convergence_history = []

        for iteration in range(max_iterations):
            V_old = V.copy()
            max_delta = 0.0

            # Update value for each state
            for state in all_states:
                if state.is_terminal():
                    V[state] = 0.0  # Terminal states have 0 value
                    continue

                # Compute value for each action
                action_values = []
                valid_actions = self.mdp.get_valid_actions(state)

                for action in valid_actions:
                    # Get immediate reward
                    reward = self.mdp.get_reward(state, action)

                    # Get next state
                    next_state = self.mdp.transition(state, action)

                    # Compute action value
                    action_value = reward + self.mdp.discount_factor * V.get(next_state, 0.0)
                    action_values.append(action_value)

                # Take maximum over actions
                if action_values:
                    V[state] = max(action_values)

                # Track convergence
                delta = abs(V[state] - V_old[state])
                max_delta = max(max_delta, delta)

            convergence_history.append(max_delta)

            # Check convergence
            if max_delta < self.tolerance:
                print(f"Value Iteration converged in {iteration + 1} iterations")
                break

        # Extract optimal policy
        policy = self._extract_policy_from_values(V)

        # Compute Q-values
        q_values = self._compute_q_values(V)

        solution = MDPSolution(
            value_function=V,
            policy=policy,
            q_values=q_values,
            convergence_history=convergence_history,
            algorithm_used="Value Iteration",
            iterations=iteration + 1
        )

        self.solutions["value_iteration"] = solution
        return solution

    def policy_iteration(self, max_iterations: int = 100) -> MDPSolution:
        """
        Solve MDP using policy iteration algorithm
        """
        print("Running Policy Iteration...")

        all_states = self._get_reachable_states()

        # Initialize random policy
        policy = {}
        for state in all_states:
            if not state.is_terminal():
                valid_actions = self.mdp.get_valid_actions(state)
                if valid_actions:
                    policy[state] = valid_actions[0]  # Arbitrary initial policy

        convergence_history = []

        for iteration in range(max_iterations):
            # Policy Evaluation: compute value function for current policy
            V = self._policy_evaluation(policy, max_eval_iterations=100)

            # Policy Improvement: update policy based on value function
            policy_stable = True
            old_policy = policy.copy()

            for state in all_states:
                if state.is_terminal():
                    continue

                # Find best action for this state
                best_action = None
                best_value = float('-inf')

                valid_actions = self.mdp.get_valid_actions(state)
                for action in valid_actions:
                    reward = self.mdp.get_reward(state, action)
                    next_state = self.mdp.transition(state, action)
                    action_value = reward + self.mdp.discount_factor * V.get(next_state, 0.0)

                    if action_value > best_value:
                        best_value = action_value
                        best_action = action

                if best_action is not None:
                    policy[state] = best_action

                    # Check if policy changed
                    if old_policy.get(state) != best_action:
                        policy_stable = False

            # Track convergence (policy changes)
            policy_changes = sum(1 for s in all_states
                                 if old_policy.get(s) != policy.get(s))
            convergence_history.append(policy_changes)

            if policy_stable:
                print(f"Policy Iteration converged in {iteration + 1} iterations")
                break

        # Final value function evaluation
        V = self._policy_evaluation(policy, max_eval_iterations=200)

        # Compute Q-values
        q_values = self._compute_q_values(V)

        solution = MDPSolution(
            value_function=V,
            policy=policy,
            q_values=q_values,
            convergence_history=convergence_history,
            algorithm_used="Policy Iteration",
            iterations=iteration + 1
        )

        self.solutions["policy_iteration"] = solution
        return solution

    def _get_reachable_states(self) -> Set[SinglePlayerState]:
        """Generate reachable states using BFS"""
        reachable = set()
        queue = [self.mdp.get_initial_state()]
        visited = set()

        while queue and len(reachable) < 10000:  # Limit to prevent explosion
            current_state = queue.pop(0)
            state_key = (current_state.position, current_state.steps_left)

            if state_key in visited or current_state.is_terminal():
                continue

            visited.add(state_key)
            reachable.add(current_state)

            # Generate successor states
            valid_actions = self.mdp.get_valid_actions(current_state)
            for action in valid_actions:
                next_state = self.mdp.transition(current_state, action)
                next_key = (next_state.position, next_state.steps_left)

                if next_key not in visited and next_state.steps_left >= 0:
                    queue.append(next_state)

        return reachable

    def _policy_evaluation(self, policy: Dict, max_eval_iterations: int = 100) -> Dict:
        """Evaluate value function for given policy"""
        all_states = self._get_reachable_states()
        V = {state: 0.0 for state in all_states}

        for _ in range(max_eval_iterations):
            V_old = V.copy()
            max_delta = 0.0

            for state in all_states:
                if state.is_terminal():
                    V[state] = 0.0
                    continue

                action = policy.get(state)
                if action is not None:
                    reward = self.mdp.get_reward(state, action)
                    next_state = self.mdp.transition(state, action)
                    V[state] = reward + self.mdp.discount_factor * V.get(next_state, 0.0)

                delta = abs(V[state] - V_old[state])
                max_delta = max(max_delta, delta)

            if max_delta < self.tolerance:
                break

        return V

    def _extract_policy_from_values(self, V: Dict) -> Dict:
        """Extract optimal policy from value function"""
        policy = {}
        all_states = self._get_reachable_states()

        for state in all_states:
            if state.is_terminal():
                continue

            best_action = None
            best_value = float('-inf')

            valid_actions = self.mdp.get_valid_actions(state)
            for action in valid_actions:
                reward = self.mdp.get_reward(state, action)
                next_state = self.mdp.transition(state, action)
                action_value = reward + self.mdp.discount_factor * V.get(next_state, 0.0)

                if action_value > best_value:
                    best_value = action_value
                    best_action = action

            if best_action is not None:
                policy[state] = best_action

        return policy

    def _compute_q_values(self, V: Dict) -> Dict:
        """Compute Q-values from value function"""
        q_values = {}
        all_states = self._get_reachable_states()

        for state in all_states:
            if state.is_terminal():
                continue

            valid_actions = self.mdp.get_valid_actions(state)
            for action in valid_actions:
                reward = self.mdp.get_reward(state, action)
                next_state = self.mdp.transition(state, action)
                q_value = reward + self.mdp.discount_factor * V.get(next_state, 0.0)
                q_values[(state, action)] = q_value

        return q_values

    def simulate_optimal_policy(self, solution: MDPSolution,
                                visualize: bool = True) -> Tuple[float, List[Tuple]]:
        """
        Simulate the optimal policy and return total reward and path
        """
        current_state = self.mdp.get_initial_state()
        total_reward = 0.0
        path = [current_state.position]

        step = 0
        while not current_state.is_terminal() and step < self.mdp.max_steps:
            action = solution.policy.get(current_state, Action.STAY)
            reward = self.mdp.get_reward(current_state, action)

            current_state = self.mdp.transition(current_state, action)
            total_reward += reward * (self.mdp.discount_factor ** step)

            path.append(current_state.position)
            step += 1

        if visualize:
            self._visualize_path(path, total_reward)

        return total_reward, path

    def _visualize_path(self, path: List[Tuple], total_reward: float):
        """Visualize the optimal path on the forest grid"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Forest map with path
        forest_values = self.mdp.forest_map[:, :, 0]  # Wood values
        if self.mdp.resource_type == "fruit":
            forest_values = self.mdp.forest_map[:, :, 1]
        elif self.mdp.resource_type == "both":
            forest_values = self.mdp.forest_map[:, :, 0] + self.mdp.forest_map[:, :, 1]

        im = ax1.imshow(forest_values.T, cmap='YlOrBr', origin='lower')

        # Plot path
        if len(path) > 1:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax1.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.8, label='Optimal Path')
            ax1.scatter(path_x[0], path_y[0], c='green', s=100, marker='s', label='Start')
            ax1.scatter(path_x[-1], path_y[-1], c='red', s=100, marker='X', label='End')

        # Add value annotations
        for i in range(self.mdp.width):
            for j in range(self.mdp.height):
                ax1.text(i, j, f'{forest_values[i, j]:.0f}',
                         ha='center', va='center', fontsize=8, color='white')

        ax1.set_title(f'Optimal Path (Total Reward: {total_reward:.2f})')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.legend()
        plt.colorbar(im, ax=ax1)

        # Plot 2: Reward collection over time
        step_rewards = []
        cumulative_rewards = []
        cumulative = 0.0

        current_state = self.mdp.get_initial_state()
        solution = self.solutions.get("value_iteration") or self.solutions.get("policy_iteration")

        if solution:
            step = 0
            while not current_state.is_terminal() and step < len(path) - 1:
                action = solution.policy.get(current_state, Action.STAY)
                reward = self.mdp.get_reward(current_state, action)

                step_rewards.append(reward)
                cumulative += reward
                cumulative_rewards.append(cumulative)

                current_state = self.mdp.transition(current_state, action)
                step += 1

            ax2.plot(step_rewards, 'bo-', label='Step Reward', markersize=6)
            ax2.plot(cumulative_rewards, 'r-', label='Cumulative Reward', linewidth=2)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Reward')
            ax2.set_title('Reward Collection Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_algorithms(self) -> None:
        """Compare different MDP solution algorithms"""
        print("\nMDP Algorithm Comparison")
        print("=" * 50)

        algorithms = ["value_iteration", "policy_iteration"]

        for alg_name in algorithms:
            if alg_name in self.solutions:
                solution = self.solutions[alg_name]
                total_reward, path = self.simulate_optimal_policy(solution, visualize=False)

                print(f"\n{solution.algorithm_used}:")
                print(f"  Iterations: {solution.iterations}")
                print(f"  Optimal Value: {total_reward:.3f}")
                print(f"  Path Length: {len(path)}")
                print(f"  Convergence: {solution.convergence_history[-1]:.6f}")


def solve_single_player_forest(forest_mdp: ForestCollectionMDP,
                               resource_type: str = "wood") -> MDPSolution:
    """
    Convenience function to solve single-player forest collection
    """
    # Convert to single-player MDP
    single_mdp = SinglePlayerMDP(forest_mdp, resource_type)

    # Solve using value iteration
    solver = SimpleMDPSolver(single_mdp)
    solution = solver.value_iteration()

    # Simulate and visualize
    solver.simulate_optimal_policy(solution)

    return solution


if __name__ == "__main__":
    # Example usage
    # Create test forest
    forest_map = np.array([
        [[5, 2], [3, 8], [7, 1]],
        [[2, 9], [8, 3], [4, 6]],
        [[6, 1], [1, 7], [9, 4]]
    ])

    # Create forest MDP
    forest_mdp = ForestCollectionMDP(
        grid_size=(3, 3),
        forest_map=forest_map,
        leader_start=(0, 0),
        max_steps_leader=5
    )

    print("Single-Player MDP Solver Example")
    print("================================")

    # Solve for wood collection
    single_mdp = SinglePlayerMDP(forest_mdp, resource_type="fruit")
    solver = SimpleMDPSolver(single_mdp)

    # Try both algorithms
    vi_solution = solver.value_iteration()
    pi_solution = solver.policy_iteration()

    # Compare results
    solver.compare_algorithms()

    # Simulate optimal policy
    print(f"\nSimulating optimal policy...")
    total_reward, path = solver.simulate_optimal_policy(pi_solution)
    print(f"Optimal total reward: {total_reward:.3f}")
    print(f"Optimal path: {path}")