import numpy as np
from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import networkx as nx


class Action(Enum):
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)


@dataclass
class Cell:
    """Represents a cell in the forest grid"""
    wood: int
    fruit: int

    def __repr__(self):
        return f"Cell(W:{self.wood}, F:{self.fruit})"


@dataclass
class GameState:
    """Complete state of the forest collection game"""
    leader_pos: Tuple[int, int]
    follower_pos: Tuple[int, int]
    leader_steps_left: int
    follower_steps_left: int
    leader_total_wood: int = 0
    follower_total_fruit: int = 0

    def is_terminal(self) -> bool:
        return self.leader_steps_left <= 0 or self.follower_steps_left <= 0


class ForestCollectionMDP:
    """
    Stackelberg MDP for forest resource collection game

    Leader wants wood, follower wants fruit.
    Leader can threaten to go to low-fruit areas if follower doesn't cooperate.
    """

    def __init__(self,
                 grid_size: Tuple[int, int],
                 forest_map: Optional[np.ndarray] = None,
                 leader_start: Tuple[int, int] = (0, 0),
                 follower_start: Tuple[int, int] = (0, 0),
                 max_steps_leader: int = 10,
                 max_steps_follower: int = 10):

        self.width, self.height = grid_size
        self.leader_start = leader_start
        self.follower_start = follower_start
        self.max_steps_leader = max_steps_leader
        self.max_steps_follower = max_steps_follower

        # Initialize forest map
        if forest_map is not None:
            assert forest_map.shape == (self.width, self.height,
                                        2), "Forest map should be (width, height, 2) for wood and fruit"
            self.forest_map = forest_map
        else:
            self.forest_map = self._generate_random_forest()

    def _generate_random_forest(self) -> np.ndarray:
        """Generate a random forest with wood and fruit distributions"""
        np.random.seed(42)  # For reproducibility
        forest = np.zeros((self.width, self.height, 2), dtype=int)

        # Generate wood (leader's resource)
        forest[:, :, 0] = np.random.randint(0, 10, (self.width, self.height))

        # Generate fruit (follower's resource) - often inversely correlated with wood
        forest[:, :, 1] = np.random.randint(0, 10, (self.width, self.height))

        # Create some high-value cells for both resources
        for _ in range(3):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            forest[x, y, 0] = np.random.randint(15, 25)  # High wood

        for _ in range(3):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            forest[x, y, 1] = np.random.randint(15, 25)  # High fruit

        return forest

    def get_initial_state(self) -> GameState:
        """Get the initial game state"""
        return GameState(
            leader_pos=self.leader_start,
            follower_pos=self.follower_start,
            leader_steps_left=self.max_steps_leader,
            follower_steps_left=self.max_steps_follower
        )

    def get_valid_actions(self, pos: Tuple[int, int]) -> List[Action]:
        """Get valid actions from current position"""
        x, y = pos
        valid_actions = [Action.STAY]

        if x > 0: valid_actions.append(Action.LEFT)
        if x < self.width - 1: valid_actions.append(Action.RIGHT)
        if y > 0: valid_actions.append(Action.DOWN)
        if y < self.height - 1: valid_actions.append(Action.UP)

        return valid_actions

    def apply_action(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        """Apply action to position and return new position"""
        dx, dy = action.value
        new_x = max(0, min(self.width - 1, pos[0] + dx))
        new_y = max(0, min(self.height - 1, pos[1] + dy))
        return (new_x, new_y)

    def get_cell_rewards(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Get wood and fruit at given position"""
        x, y = pos
        return int(self.forest_map[x, y, 0]), int(self.forest_map[x, y, 1])

    def transition(self, state: GameState,
                   leader_action: Action,
                   follower_action: Action) -> GameState:
        """Execute one step of the game"""
        if state.is_terminal():
            return state

        # Apply actions
        new_leader_pos = self.apply_action(state.leader_pos, leader_action)
        new_follower_pos = self.apply_action(state.follower_pos, follower_action)

        # Collect resources
        leader_wood, leader_fruit = self.get_cell_rewards(new_leader_pos)
        follower_wood, follower_fruit = self.get_cell_rewards(new_follower_pos)

        # Update state
        new_state = GameState(
            leader_pos=new_leader_pos,
            follower_pos=new_follower_pos,
            leader_steps_left=state.leader_steps_left - 1,
            follower_steps_left=state.follower_steps_left - 1,
            leader_total_wood=state.leader_total_wood + leader_wood,
            follower_total_fruit=state.follower_total_fruit + follower_fruit
        )

        return new_state

    def get_leader_reward(self, state: GameState) -> float:
        """Leader's reward is total wood collected"""
        return float(state.leader_total_wood)

    def get_follower_reward(self, state: GameState) -> float:
        """Follower's reward is total fruit collected"""
        return float(state.follower_total_fruit)

    def visualize_forest(self, show_resources: str = "both"):
        """Visualize the forest map"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Wood distribution
        wood_map = self.forest_map[:, :, 0].T
        im1 = axes[0].imshow(wood_map, cmap='YlOrBr', origin='lower')
        axes[0].set_title('Wood Distribution (Leader Resource)')
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')

        # Add text annotations
        for i in range(self.width):
            for j in range(self.height):
                axes[0].text(i, j, f'{wood_map[j, i]:.0f}',
                             ha='center', va='center', fontsize=8)

        plt.colorbar(im1, ax=axes[0])

        # Fruit distribution
        fruit_map = self.forest_map[:, :, 1].T
        im2 = axes[1].imshow(fruit_map, cmap='Reds', origin='lower')
        axes[1].set_title('Fruit Distribution (Follower Resource)')
        axes[1].set_xlabel('X coordinate')
        axes[1].set_ylabel('Y coordinate')

        # Add text annotations
        for i in range(self.width):
            for j in range(self.height):
                axes[1].text(i, j, f'{fruit_map[j, i]:.0f}',
                             ha='center', va='center', fontsize=8)

        plt.colorbar(im2, ax=axes[1])

        # Mark starting positions
        axes[0].plot(self.leader_start[0], self.leader_start[1], 'bo',
                     markersize=10, label='Leader Start')
        axes[1].plot(self.follower_start[0], self.follower_start[1], 'go',
                     markersize=10, label='Follower Start')

        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.show()


class StackelbergForestSolver:
    """
    Solver for Stackelberg equilibrium in the forest collection game
    """

    def __init__(self, mdp: ForestCollectionMDP):
        self.mdp = mdp
        self.state_values = {}
        self.policies = {}

    def solve_follower_best_response(self,
                                     leader_policy: Dict,
                                     state: GameState,
                                     depth: int = 0,
                                     max_depth: int = 20) -> Tuple[float, Action]:
        """
        Solve follower's best response given leader's policy
        Uses backward induction
        """
        if state.is_terminal() or depth >= max_depth:
            return self.mdp.get_follower_reward(state), Action.STAY

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

            current_reward = self.mdp.get_follower_reward(next_state) - \
                             self.mdp.get_follower_reward(state)
            total_value = current_reward + 0.9 * future_value  # discount factor

            if total_value > best_value:
                best_value = total_value
                best_action = f_action

        return best_value, best_action

    def evaluate_strategy_profile(self,
                                  leader_policy: Dict,
                                  max_depth: int = 20) -> Tuple[float, float]:
        """
        Evaluate a strategy profile and return (leader_payoff, follower_payoff)
        """
        initial_state = self.mdp.get_initial_state()

        # Simulate the game
        current_state = initial_state
        total_leader_reward = 0
        total_follower_reward = 0

        for step in range(max_depth):
            if current_state.is_terminal():
                break

            # Get leader action
            leader_action = leader_policy.get(current_state, Action.STAY)

            # Get follower's best response
            _, follower_action = self.solve_follower_best_response(
                leader_policy, current_state, 0, max_depth - step
            )

            # Transition
            next_state = self.mdp.transition(current_state, leader_action, follower_action)

            # Accumulate rewards
            total_leader_reward += (self.mdp.get_leader_reward(next_state) -
                                    self.mdp.get_leader_reward(current_state))
            total_follower_reward += (self.mdp.get_follower_reward(next_state) -
                                      self.mdp.get_follower_reward(current_state))

            current_state = next_state

        return total_leader_reward, total_follower_reward


# Example usage and testing
if __name__ == "__main__":
    # Create a small forest for testing
    forest_map = np.array([
        [[5, 2], [3, 8], [7, 1]],
        [[2, 9], [8, 3], [4, 6]],
        [[6, 1], [1, 7], [9, 4]]
    ])

    # Initialize game
    game = ForestCollectionMDP(
        grid_size=(3, 3),
        forest_map=forest_map,
        leader_start=(0, 0),
        follower_start=(2, 2),
        max_steps_leader=5,
        max_steps_follower=5
    )

    print("Forest Collection Stackelberg MDP Game")
    print("=====================================")
    print(f"Grid size: {game.width} x {game.height}")
    print(f"Leader starts at: {game.leader_start}")
    print(f"Follower starts at: {game.follower_start}")
    print(f"Max steps - Leader: {game.max_steps_leader}, Follower: {game.max_steps_follower}")

    # Show initial state
    initial_state = game.get_initial_state()
    print(f"\nInitial state: {initial_state}")

    # Visualize forest
    game.visualize_forest()

    # Initialize solver
    solver = StackelbergForestSolver(game)

    # Test with a simple leader policy (go right then up)
    simple_leader_policy = {}
    test_state = initial_state

    print(f"\nTesting simple leader policy...")
    leader_payoff, follower_payoff = solver.evaluate_strategy_profile(simple_leader_policy)
    print(f"Leader payoff: {leader_payoff}")
    print(f"Follower payoff: {follower_payoff}")