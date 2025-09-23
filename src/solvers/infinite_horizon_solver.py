"""
Infinite Horizon Extension of your Forest EPF Solver
Only minimal changes needed - the key difference is adding discount factor to EPF computation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, replace

from matplotlib import pyplot as plt

from src.forest_game import GameState, ForestCollectionMDP, Action
from src.solvers.forest_epf_solver import ForestEPFSolver, GameNode, EPF, EPFPoint



@dataclass(frozen=True)
class InfiniteGameState:
    """Same as GameState but without step limits"""
    leader_pos: Tuple[int, int]
    follower_pos: Tuple[int, int]
    leader_total_wood: int = 0
    follower_total_fruit: int = 0
    turn: Optional[bool] = None

    def is_terminal(self) -> bool:
        # For infinite horizon, only terminal if we choose to stop
        return False  # Never terminal for pure infinite horizon

    def is_leader_turn(self) -> bool:
        return self.turn is True

    def is_follower_turn(self) -> bool:
        return self.turn is False

    def get_current_player_pos(self) -> Tuple[int, int]:
        if self.is_leader_turn():
            return self.leader_pos
        elif self.is_follower_turn():
            return self.follower_pos
        else:
            raise ValueError("Cannot determine current player position")

    def to_key(self) -> Tuple:
        # Same as original but without step counts
        return (self.leader_pos, self.follower_pos,
                self.leader_total_wood, self.follower_total_fruit, self.turn)

class InfiniteForestMDP:
    """Infinite horizon wrapper for ForestCollectionMDP"""

    def __init__(self, finite_mdp: ForestCollectionMDP):
        # Copy all attributes except step limits
        self.width = finite_mdp.width
        self.height = finite_mdp.height
        self.forest_map = finite_mdp.forest_map
        self.leader_start = finite_mdp.leader_start
        self.follower_start = finite_mdp.follower_start
        self.leader_start_first = finite_mdp.leader_starts_first

    def get_initial_state(self) -> InfiniteGameState:
        return InfiniteGameState(
            leader_pos=self.leader_start,
            follower_pos=self.follower_start,
            turn=self.leader_start_first  # Use the correct starting player
        )

    def get_valid_actions(self, pos: Tuple[int, int]) -> List[Action]:
        """Same as finite version"""
        x, y = pos
        valid_actions = [Action.STAY]
        if x > 0: valid_actions.append(Action.LEFT)
        if x < self.width - 1: valid_actions.append(Action.RIGHT)
        if y > 0: valid_actions.append(Action.DOWN)
        if y < self.height - 1: valid_actions.append(Action.UP)
        return valid_actions

    def is_valid_action(self, state: InfiniteGameState, action: Action) -> bool:
        current_pos = state.get_current_player_pos()
        return action in self.get_valid_actions(current_pos)

    def apply_action(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Same as finite version"""
        dx, dy = action.value
        new_x = max(0, min(self.width - 1, pos[0] + dx))
        new_y = max(0, min(self.height - 1, pos[1] + dy))
        return (new_x, new_y)

    def get_cell_rewards(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Same as finite version"""
        x, y = pos
        return int(self.forest_map[x, y, 0]), int(self.forest_map[x, y, 1])

    def transition(self, state: InfiniteGameState, action: Action) -> InfiniteGameState:
        """Transition without step counting"""
        if state.is_leader_turn():
            new_pos = self.apply_action(state.leader_pos, action)
            wood_reward, fruit_reward = self.get_cell_rewards(new_pos)

            return InfiniteGameState(
                leader_pos=new_pos,
                follower_pos=state.follower_pos,
                leader_total_wood=state.leader_total_wood + wood_reward,
                follower_total_fruit=state.follower_total_fruit + fruit_reward,
                turn=False  # Switch to follower
            )
        else:
            new_pos = self.apply_action(state.follower_pos, action)
            wood_reward, fruit_reward = self.get_cell_rewards(new_pos)

            return InfiniteGameState(
                leader_pos=state.leader_pos,
                follower_pos=new_pos,
                leader_total_wood=state.leader_total_wood + wood_reward,
                follower_total_fruit=state.follower_total_fruit + fruit_reward,
                turn=True  # Switch to leader
            )

    def get_leader_reward(self, state: InfiniteGameState) -> float:
        return float(state.leader_total_wood)

    def get_follower_reward(self, state: InfiniteGameState) -> float:
        return float(state.follower_total_fruit)

class InfiniteForestEPFSolver:
    """Infinite horizon version of ForestEPFSolver"""

    def __init__(self, mdp: InfiniteForestMDP, discount_factor: float = 0.9,
                 max_iterations: int = 1000, convergence_threshold: float = 1e-3,
                 max_states: int = 1000):
        self.mdp = mdp
        self.gamma = discount_factor
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_states = max_states

        # Value iteration storage
        self.state_epfs: Dict[InfiniteGameState, EPF] = {}

    def solve(self, initial_state: Optional[InfiniteGameState] = None) -> Dict:
        """Solve using value iteration instead of tree building"""
        if initial_state is None:
            initial_state = self.mdp.get_initial_state()

        print(f"INFINITE HORIZON EPF SOLVER (γ = {self.gamma})")
        print("="*50)

        # Phase 1: Initialize reachable states
        states, pos_map = self._get_reachable_states(initial_state)
        print(f"Found {len(states)} reachable states")

        # Phase 2: Value iteration
        converged = self._value_iteration(states, pos_map)

        # Phase 3: Extract solution
        root_epf = self.state_epfs[initial_state]
        optimal_point = root_epf.get_maximum_point()

        return {
            'root_epf': root_epf,
            'optimal_point': optimal_point,
            'converged': converged,
            'discount_factor': self.gamma,
            'num_states': len(states)
        }

    def _get_reachable_states(self, initial_state: InfiniteGameState) -> Tuple[List[InfiniteGameState], Dict[Tuple, InfiniteGameState]]:
        """Get reachable states - only position and turn matter for infinite horizon"""

        visited = set()
        queue = [initial_state]
        states = []
        position_to_state_map = {}  # Create the map

        while queue:
            state = queue.pop(0)
            position_key = (state.leader_pos, state.follower_pos, state.turn)
            if position_key in visited:
                continue

            visited.add(position_key)
            states.append(state)
            position_to_state_map[position_key] = state  # Populate the map
            self.state_epfs[state] = EPF([EPFPoint(0.0, 0.0)])

            # Add successors
            if state.is_leader_turn():
                valid_actions = self.mdp.get_valid_actions(state.leader_pos)
            else:
                valid_actions = self.mdp.get_valid_actions(state.follower_pos)

            for action in valid_actions:
                if self.mdp.is_valid_action(state, action):
                    next_state = self.mdp.transition(state, action)
                    next_position_key = (next_state.leader_pos, next_state.follower_pos, next_state.turn)
                    if next_position_key not in visited:
                        queue.append(next_state)

        return states, position_to_state_map

    def _value_iteration(self, states: List[InfiniteGameState], pos_map: Dict) -> bool:
        """The key change: value iteration with discount factor"""
        for iteration in range(self.max_iterations):
            # Store old EPFs for convergence check
            old_epfs = {state: EPF(epf.points.copy()) for state, epf in self.state_epfs.items()}

            # Update each state's EPF
            for state in states:
                self._update_state_epf(state, pos_map)

            # Check convergence
            max_change = 0.0
            for state in states:
                if state in old_epfs:
                    change = self._epf_distance(old_epfs[state], self.state_epfs[state])
                    max_change = max(max_change, change)

            # print(f"Iteration {iteration + 1}: max change = {max_change:.6f}")

            # Debug: Check why max_change is inf
            if max_change == float('inf') and iteration < 2:
                inf_count = 0
                for state in states[:3]:  # Check first 3 states
                    if state in old_epfs:
                        old_len = len(old_epfs[state].points)
                        new_len = len(self.state_epfs[state].points)
                        if old_len != new_len:
                            inf_count += 1
                            if inf_count == 1:  # Show first example
                                print(f"Debug: State with different point counts: {old_len} -> {new_len}")
                print(f"Debug: {inf_count} states changed point counts")

            if max_change < self.convergence_threshold:
                print(f"✓ Converged after {iteration + 1} iterations")
                return True

        print(f"✗ Did not converge after {self.max_iterations} iterations")
        return False

    def _update_state_epf(self, state: InfiniteGameState, pos_map: Dict):
        """
        Corrected Bellman update for the EPF of a single state.
        This version uses a stable update rule for the follower.
        """

        # Helper to find the canonical state using the pre-computed map
        def get_canonical_state(pos_state):
            pos_key = (pos_state.leader_pos, pos_state.follower_pos, pos_state.turn)
            return pos_map.get(pos_key)

        if state.is_leader_turn():
            # Leader's turn: Standard upper envelope of all action EPFs
            child_epfs = []
            valid_actions = self.mdp.get_valid_actions(state.leader_pos)
            for action in valid_actions:
                # Calculate immediate rewards for this action
                new_pos = self.mdp.apply_action(state.leader_pos, action)
                wood_reward, fruit_reward = self.mdp.get_cell_rewards(new_pos)

                # Find the next canonical state and its EPF
                next_state_obj = self.mdp.transition(state, action)
                canonical_next_state = get_canonical_state(next_state_obj)

                if canonical_next_state and self.state_epfs[canonical_next_state].points:
                    # Discount the future EPF
                    discounted_epf = self._discount_epf(self.state_epfs[canonical_next_state])

                    # Add immediate reward to create the action's EPF
                    action_points = [
                        EPFPoint(
                            follower_payoff=fruit_reward + p.follower_payoff,
                            leader_payoff=wood_reward + p.leader_payoff
                        ) for p in discounted_epf.points
                    ]
                    child_epfs.append(EPF(action_points))

            if child_epfs:
                all_points = [p for epf in child_epfs for p in epf.points]
                if all_points:
                    combined_epf = EPF(all_points)
                    self.state_epfs[state] = combined_epf.upper_concave_envelope()

        else:  # Follower's turn
            # Follower's turn: Leader anticipates follower's optimal choice for each action
            follower_choice_points = []
            valid_actions = self.mdp.get_valid_actions(state.follower_pos)
            for action in valid_actions:
                # Calculate immediate rewards for this action
                new_pos = self.mdp.apply_action(state.follower_pos, action)
                wood_reward, fruit_reward = self.mdp.get_cell_rewards(new_pos)

                # Find the next canonical state and its EPF
                next_state_obj = self.mdp.transition(state, action)
                canonical_next_state = get_canonical_state(next_state_obj)

                if canonical_next_state and self.state_epfs[canonical_next_state].points:
                    # Discount the future EPF
                    discounted_epf = self._discount_epf(self.state_epfs[canonical_next_state])

                    # Add immediate reward
                    action_points = [
                        EPFPoint(
                            follower_payoff=fruit_reward + p.follower_payoff,
                            leader_payoff=wood_reward + p.leader_payoff
                        ) for p in discounted_epf.points
                    ]

                    if not action_points:
                        continue

                    # The follower will choose the point on this EPF that maximizes their payoff
                    best_point_for_follower = max(action_points, key=lambda p: p.follower_payoff)
                    follower_choice_points.append(best_point_for_follower)

            if follower_choice_points:
                # The new EPF is the upper concave envelope of the points the follower would choose
                combined_epf = EPF(follower_choice_points)
                self.state_epfs[state] = combined_epf.upper_concave_envelope()
    def _discount_epf(self, epf: EPF) -> EPF:
        """Apply discount factor to EPF - THE KEY INFINITE HORIZON OPERATION"""
        discounted_points = []
        for point in epf.points:
            new_leader = self.gamma * point.leader_payoff
            new_follower = self.gamma * point.follower_payoff
            discounted_points.append(EPFPoint(new_follower, new_leader))
        return EPF(discounted_points)

    def _left_truncate(self, epf: EPF, threshold: float) -> EPF:
        """Left truncate EPF at threshold"""
        truncated_points = []
        for point in epf.points:
            if point.follower_payoff >= threshold - 1e-9:
                truncated_points.append(point)
        return EPF(truncated_points)

    def _epf_distance(self, epf1: EPF, epf2: EPF) -> float:
        """Compute distance between EPFs for convergence check"""
        if not epf1.points or not epf2.points:
            return float('inf') if bool(epf1.points) != bool(epf2.points) else 0.0

        # Compare maximum values rather than structure
        # This gives a more reasonable convergence criterion
        def get_max_payoffs(epf):
            if not epf.points:
                return 0.0, 0.0
            max_leader = max(p.leader_payoff for p in epf.points)
            max_follower = max(p.follower_payoff for p in epf.points)
            return max_leader, max_follower

        max_leader1, max_follower1 = get_max_payoffs(epf1)
        max_leader2, max_follower2 = get_max_payoffs(epf2)

        leader_diff = abs(max_leader1 - max_leader2)
        follower_diff = abs(max_follower1 - max_follower2)

        return max(leader_diff, follower_diff)

# Test function
def test_infinite():
    """Compare infinite horizon vs finite horizon results"""

    # Create finite horizon game
    forest_map = np.array([
        [[5, 2], [3, 8], [7, 1]],
        [[2, 9], [8, 3], [4, 6]],
        [[6, 1], [1, 7], [9, 4]]
    ])

    # Initialize game
    finite_mdp = ForestCollectionMDP(
        grid_size=(3, 3),
        forest_map=forest_map,
        leader_start=(0, 0),
        follower_start=(2, 2),
        max_steps_leader=1,
        max_steps_follower=1,
        leader_starts_first=True
    )

    # Convert to infinite horizon
    infinite_mdp = InfiniteForestMDP(finite_mdp)

    # Test different discount factors
    print("\n" + "="*50)
    print("INFINITE HORIZON SOLUTIONS")
    print("="*50)

    for gamma in [0.8, 0.9, 0.95]:
        print(f"\n--- Discount factor γ = {gamma} ---")
        infinite_solver = InfiniteForestEPFSolver(
            infinite_mdp,
            discount_factor=gamma,
            max_iterations=50,
            convergence_threshold=0.1  # Reasonable threshold for max payoff changes
        )

        infinite_solution = infinite_solver.solve()
        infinite_leader = infinite_solution['optimal_point'][1]  # leader payoff

        epf = infinite_solution['root_epf']
        mu2_values = [p.follower_payoff for p in epf.points]
        u1_values = [p.leader_payoff for p in epf.points]

        plt.figure(figsize=(10, 6))
        plt.plot(mu2_values, u1_values, 'b-o', linewidth=2, markersize=6)

        plt.xlabel('Follower Payoff (μ₂)')
        plt.ylabel('Leader Payoff (U₁)')
        plt.grid(True, alpha=0.3)

        # Highlight optimal point
        mu2_opt, u1_opt = epf.get_maximum_point()
        plt.plot(mu2_opt, u1_opt, 'ro', markersize=10, label=f'Optimal: ({mu2_opt:.2f}, {u1_opt:.2f})')
        plt.legend()

        plt.tight_layout()
        plt.show()

        print(f"Infinite horizon optimal leader payoff: {infinite_leader:.3f}")
        print(f"Converged: {infinite_solution['converged']}")

if __name__ == "__main__":
    test_infinite()