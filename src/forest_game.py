import numpy as np
from typing import Tuple, List, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt


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


@dataclass(frozen=True)
class GameState:
    """Complete state of the forest collection game with turn-based play"""
    leader_pos: Tuple[int, int]
    follower_pos: Tuple[int, int]
    leader_steps_left: int
    follower_steps_left: int
    leader_total_wood: int = 0
    follower_total_fruit: int = 0
    turn: Optional[bool] = None  # True = leader's turn, False = follower's turn, None = simultaneous

    def is_terminal(self) -> bool:
        """Game ends when both players have no steps left"""
        return self.leader_steps_left <= 0 and self.follower_steps_left <= 0

    def is_leader_turn(self) -> bool:
        """Check if it's currently the leader's turn"""
        return self.turn is True

    def is_follower_turn(self) -> bool:
        """Check if it's currently the follower's turn"""
        return self.turn is False

    def get_current_player_pos(self) -> Tuple[int, int]:
        """Get position of the player whose turn it is"""
        if self.is_leader_turn():
            return self.leader_pos
        elif self.is_follower_turn():
            return self.follower_pos
        else:
            raise ValueError("Cannot determine current player position in simultaneous game")

    def get_current_player_steps_left(self) -> int:
        """Get steps left for the player whose turn it is"""
        if self.is_leader_turn():
            return self.leader_steps_left
        elif self.is_follower_turn():
            return self.follower_steps_left
        else:
            raise ValueError("Cannot determine current player steps in simultaneous game")

    def to_key(self) -> Tuple:
        """Convert state to a hashable key for use in dictionaries"""
        return (self.leader_pos, self.follower_pos,
                self.leader_steps_left, self.follower_steps_left,
                self.leader_total_wood, self.follower_total_fruit, self.turn)

    @classmethod
    def from_key(cls, key: Tuple):
        """Create GameState from a key tuple"""
        return cls(
            leader_pos=key[0],
            follower_pos=key[1],
            leader_steps_left=key[2],
            follower_steps_left=key[3],
            leader_total_wood=key[4],
            follower_total_fruit=key[5],
            turn=key[6]
        )


class ForestCollectionMDP:
    """
    Turn-based Stackelberg MDP for forest resource collection game

    Leader wants wood, follower wants fruit.
    Players alternate turns in a sequential game structure.
    Leader can commit to threat strategies that punish uncooperative followers.
    """

    def __init__(self,
                 grid_size: Tuple[int, int],
                 forest_map: Optional[np.ndarray] = None,
                 leader_start: Tuple[int, int] = (0, 0),
                 follower_start: Tuple[int, int] = (0, 0),
                 max_steps_leader: int = 10,
                 max_steps_follower: int = 10,
                 leader_starts_first: bool = True):

        self.width, self.height = grid_size
        self.leader_start = leader_start
        self.follower_start = follower_start
        self.max_steps_leader = max_steps_leader
        self.max_steps_follower = max_steps_follower
        self.leader_starts_first = leader_starts_first

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

        # Generate fruit (follower's resource) - sometimes inversely correlated with wood
        forest[:, :, 1] = np.random.randint(0, 10, (self.width, self.height))

        # Create some high-value cells for both resources
        for _ in range(3):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            forest[x, y, 0] = np.random.randint(15, 25)  # High wood

        for _ in range(3):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            forest[x, y, 1] = np.random.randint(15, 25)  # High fruit

        # Create some conflict areas (high wood, low fruit) and (low wood, high fruit)
        for _ in range(2):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            forest[x, y, 0] = np.random.randint(12, 20)  # High wood
            forest[x, y, 1] = np.random.randint(0, 3)  # Low fruit

        for _ in range(2):
            x, y = np.random.randint(0, self.width), np.random.randint(0, self.height)
            forest[x, y, 0] = np.random.randint(0, 3)  # Low wood
            forest[x, y, 1] = np.random.randint(12, 20)  # High fruit

        return forest

    def get_initial_state(self, leader_starts: Optional[bool] = None) -> GameState:
        """Get the initial game state with proper turn assignment"""
        if leader_starts is None:
            leader_starts = self.leader_starts_first

        return GameState(
            leader_pos=self.leader_start,
            follower_pos=self.follower_start,
            leader_steps_left=self.max_steps_leader,
            follower_steps_left=self.max_steps_follower,
            turn=leader_starts  # True = leader's turn, False = follower's turn
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

    def transition(self, state: GameState, action: Action) -> GameState:
        """Execute one step of the sequential game"""
        if state.is_terminal():
            return state

        if state.turn is None:
            raise ValueError("Cannot transition: turn is not defined (use sequential mode)")

        if state.is_leader_turn():
            # Leader's turn
            if state.leader_steps_left <= 0:
                # Leader has no steps left, skip to follower
                return GameState(
                    leader_pos=state.leader_pos,
                    follower_pos=state.follower_pos,
                    leader_steps_left=state.leader_steps_left,
                    follower_steps_left=state.follower_steps_left,
                    leader_total_wood=state.leader_total_wood,
                    follower_total_fruit=state.follower_total_fruit,
                    turn=False  # Switch to follower
                )

            new_leader_pos = self.apply_action(state.leader_pos, action)
            leader_wood, _ = self.get_cell_rewards(new_leader_pos)

            new_state = GameState(
                leader_pos=new_leader_pos,
                follower_pos=state.follower_pos,
                leader_steps_left=state.leader_steps_left - 1,
                follower_steps_left=state.follower_steps_left,
                leader_total_wood=state.leader_total_wood + leader_wood,
                follower_total_fruit=state.follower_total_fruit,
                turn=False  # Switch to follower's turn
            )

        else:
            # Follower's turn
            if state.follower_steps_left <= 0:
                # Follower has no steps left, skip to leader
                return GameState(
                    leader_pos=state.leader_pos,
                    follower_pos=state.follower_pos,
                    leader_steps_left=state.leader_steps_left,
                    follower_steps_left=state.follower_steps_left,
                    leader_total_wood=state.leader_total_wood,
                    follower_total_fruit=state.follower_total_fruit,
                    turn=True  # Switch to leader
                )

            new_follower_pos = self.apply_action(state.follower_pos, action)
            _, follower_fruit = self.get_cell_rewards(new_follower_pos)

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

    def transition_simultaneous(self, state: GameState,
                                leader_action: Action,
                                follower_action: Action) -> GameState:
        """Execute simultaneous moves (for compatibility with original version)"""
        if state.is_terminal():
            return state

        # Apply actions simultaneously
        new_leader_pos = self.apply_action(state.leader_pos, leader_action)
        new_follower_pos = self.apply_action(state.follower_pos, follower_action)

        # Collect resources
        leader_wood, _ = self.get_cell_rewards(new_leader_pos)
        _, follower_fruit = self.get_cell_rewards(new_follower_pos)

        # Update state
        new_state = GameState(
            leader_pos=new_leader_pos,
            follower_pos=new_follower_pos,
            leader_steps_left=state.leader_steps_left - 1,
            follower_steps_left=state.follower_steps_left - 1,
            leader_total_wood=state.leader_total_wood + leader_wood,
            follower_total_fruit=state.follower_total_fruit + follower_fruit,
            turn=state.turn  # Maintain turn state
        )

        return new_state

    def get_leader_reward(self, state: GameState) -> float:
        """Leader's reward is total wood collected"""
        return float(state.leader_total_wood)

    def get_follower_reward(self, state: GameState) -> float:
        """Follower's reward is total fruit collected"""
        return float(state.follower_total_fruit)

    def get_current_player_valid_actions(self, state: GameState) -> List[Action]:
        """Get valid actions for the current player whose turn it is"""
        if state.turn is None:
            raise ValueError("Cannot get current player actions: turn is not defined")

        current_pos = state.get_current_player_pos()
        return self.get_valid_actions(current_pos)

    def who_moves_next(self, state: GameState) -> Optional[str]:
        """Determine who moves next given current state"""
        if state.is_terminal():
            return None

        if state.turn is True:
            return "leader"
        elif state.turn is False:
            return "follower"
        else:
            return "simultaneous"

    def create_conflict_forest(self) -> np.ndarray:
        """Create a forest map with strategic conflict between resources"""
        forest = np.zeros((self.width, self.height, 2), dtype=int)

        # Create zones with different resource characteristics
        for x in range(self.width):
            for y in range(self.height):
                # Base resources
                forest[x, y, 0] = np.random.randint(1, 6)  # Base wood
                forest[x, y, 1] = np.random.randint(1, 6)  # Base fruit

        # High wood, low fruit zones (good for leader, bad for follower)
        wood_zones = [(0, 0), (self.width - 1, 0), (self.width // 2, self.height // 2)]
        for x, y in wood_zones:
            if 0 <= x < self.width and 0 <= y < self.height:
                forest[x, y, 0] = np.random.randint(15, 25)  # High wood
                forest[x, y, 1] = np.random.randint(0, 3)  # Low fruit

        # High fruit, low wood zones (good for follower, bad for leader)
        fruit_zones = [(0, self.height - 1), (self.width - 1, self.height - 1), (self.width // 4, self.height // 4)]
        for x, y in fruit_zones:
            if 0 <= x < self.width and 0 <= y < self.height:
                forest[x, y, 0] = np.random.randint(0, 3)  # Low wood
                forest[x, y, 1] = np.random.randint(15, 25)  # High fruit

        # Cooperative zones (good for both)
        coop_zones = [(self.width // 2, 0), (0, self.height // 2)]
        for x, y in coop_zones:
            if 0 <= x < self.width and 0 <= y < self.height:
                forest[x, y, 0] = np.random.randint(8, 15)  # Good wood
                forest[x, y, 1] = np.random.randint(8, 15)  # Good fruit

        return forest

    def visualize_forest(self, state: Optional[GameState] = None,
                         show_players: bool = True, title_suffix: str = ""):
        """Visualize the forest map with optional player positions"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Wood distribution
        wood_map = self.forest_map[:, :, 0].T
        im1 = axes[0].imshow(wood_map, cmap='YlOrBr', origin='lower')
        axes[0].set_title(f'Wood Distribution (Leader Resource) {title_suffix}')
        axes[0].set_xlabel('X coordinate')
        axes[0].set_ylabel('Y coordinate')

        # Add text annotations for wood values
        for i in range(self.width):
            for j in range(self.height):
                axes[0].text(i, j, f'{wood_map[j, i]:.0f}',
                             ha='center', va='center', fontsize=8,
                             color='white' if wood_map[j, i] > 10 else 'black')

        plt.colorbar(im1, ax=axes[0])

        # Fruit distribution
        fruit_map = self.forest_map[:, :, 1].T
        im2 = axes[1].imshow(fruit_map, cmap='Reds', origin='lower')
        axes[1].set_title(f'Fruit Distribution (Follower Resource) {title_suffix}')
        axes[1].set_xlabel('X coordinate')
        axes[1].set_ylabel('Y coordinate')

        # Add text annotations for fruit values
        for i in range(self.width):
            for j in range(self.height):
                axes[1].text(i, j, f'{fruit_map[j, i]:.0f}',
                             ha='center', va='center', fontsize=8,
                             color='white' if fruit_map[j, i] > 10 else 'black')

        plt.colorbar(im2, ax=axes[1])

        # Show player positions if provided
        if show_players:
            if state is not None:
                # Current positions
                axes[0].plot(state.leader_pos[0], state.leader_pos[1], 'bo',
                             markersize=12, label=f'Leader (Turn: {state.is_leader_turn()})')
                axes[1].plot(state.follower_pos[0], state.follower_pos[1], 'go',
                             markersize=12, label=f'Follower (Turn: {state.is_follower_turn()})')

                # Show whose turn it is with special marker
                if state.is_leader_turn():
                    axes[0].plot(state.leader_pos[0], state.leader_pos[1], 'y*',
                                 markersize=20, label='Current Player')
                elif state.is_follower_turn():
                    axes[1].plot(state.follower_pos[0], state.follower_pos[1], 'y*',
                                 markersize=20, label='Current Player')
            else:
                # Starting positions
                axes[0].plot(self.leader_start[0], self.leader_start[1], 'bo',
                             markersize=10, label='Leader Start')
                axes[1].plot(self.follower_start[0], self.follower_start[1], 'go',
                             markersize=10, label='Follower Start')

        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.show()

    def simulate_sequential_game(self,
                                 leader_policy: Dict[GameState, Action],
                                 follower_policy: Dict[GameState, Action],
                                 max_steps: int = 50,
                                 verbose: bool = False) -> Tuple[GameState, List[GameState]]:
        """
        Simulate a complete sequential game given both players' policies

        Returns:
            (final_state, game_history)
        """
        current_state = self.get_initial_state()
        history = [current_state]

        if verbose:
            print(f"Starting sequential game simulation:")
            print(f"Initial: Leader at {current_state.leader_pos}, Follower at {current_state.follower_pos}")
            print(f"Turn order: {'Leader first' if current_state.turn else 'Follower first'}")

        step = 0
        while not current_state.is_terminal() and step < max_steps:
            if verbose:
                player = "Leader" if current_state.is_leader_turn() else "Follower"
                print(f"\nStep {step + 1}: {player}'s turn")
                print(f"  Current state: {current_state}")

            # Get action from appropriate policy
            if current_state.is_leader_turn():
                action = leader_policy.get(current_state, Action.STAY)
                if verbose:
                    print(f"  Leader chooses: {action}")
            else:
                action = follower_policy.get(current_state, Action.STAY)
                if verbose:
                    print(f"  Follower chooses: {action}")

            # Execute transition
            current_state = self.transition(current_state, action)
            history.append(current_state)

            if verbose:
                print(f"  New state: {current_state}")
                print(f"  Rewards so far: Leader={current_state.leader_total_wood}, "
                      f"Follower={current_state.follower_total_fruit}")

            step += 1

        if verbose:
            print(f"\nGame ended after {step} steps")
            print(f"Final payoffs: Leader={current_state.leader_total_wood}, "
                  f"Follower={current_state.follower_total_fruit}")

        return current_state, history

    def analyze_cooperation_vs_conflict(self) -> Dict[str, float]:
        """
        Analyze the strategic tension between cooperation and conflict in the forest

        Returns metrics about the game's strategic structure
        """

        # Find cooperative cells (good for both players)
        cooperative_cells = []
        conflict_cells = []
        leader_advantage_cells = []
        follower_advantage_cells = []

        for x in range(self.width):
            for y in range(self.height):
                wood = self.forest_map[x, y, 0]
                fruit = self.forest_map[x, y, 1]

                if wood >= 8 and fruit >= 8:
                    cooperative_cells.append((x, y))
                elif wood >= 12 and fruit <= 3:
                    leader_advantage_cells.append((x, y))
                elif wood <= 3 and fruit >= 12:
                    follower_advantage_cells.append((x, y))
                elif abs(wood - fruit) > 8:
                    conflict_cells.append((x, y))

        total_cells = self.width * self.height

        analysis = {
            'cooperation_ratio': len(cooperative_cells) / total_cells,
            'conflict_ratio': len(conflict_cells) / total_cells,
            'leader_advantage_ratio': len(leader_advantage_cells) / total_cells,
            'follower_advantage_ratio': len(follower_advantage_cells) / total_cells,
            'total_wood': float(np.sum(self.forest_map[:, :, 0])),
            'total_fruit': float(np.sum(self.forest_map[:, :, 1])),
            'resource_correlation': float(np.corrcoef(
                self.forest_map[:, :, 0].flatten(),
                self.forest_map[:, :, 1].flatten()
            )[0, 1])
        }

        return analysis

    def create_strategic_forest(self, conflict_level: str = "medium") -> None:
        """
        Create a strategically designed forest to test specific game theory concepts

        Args:
            conflict_level: "low", "medium", or "high" conflict between resources
        """

        if conflict_level == "low":
            # Mostly cooperative - resources are positively correlated
            self.forest_map = self._create_cooperative_forest()
        elif conflict_level == "medium":
            # Mixed - some cooperation, some conflict
            self.forest_map = self.create_conflict_forest()
        elif conflict_level == "high":
            # High conflict - resources are negatively correlated
            self.forest_map = self._create_high_conflict_forest()
        else:
            raise ValueError("conflict_level must be 'low', 'medium', or 'high'")

    def _create_cooperative_forest(self) -> np.ndarray:
        """Create forest where wood and fruit are positively correlated"""
        forest = np.zeros((self.width, self.height, 2), dtype=int)

        for x in range(self.width):
            for y in range(self.height):
                base_value = np.random.randint(3, 12)
                noise_wood = np.random.randint(-2, 3)
                noise_fruit = np.random.randint(-2, 3)

                forest[x, y, 0] = max(0, base_value + noise_wood)
                forest[x, y, 1] = max(0, base_value + noise_fruit)

        return forest

    def _create_high_conflict_forest(self) -> np.ndarray:
        """Create forest where wood and fruit are negatively correlated"""
        forest = np.zeros((self.width, self.height, 2), dtype=int)

        for x in range(self.width):
            for y in range(self.height):
                total_resources = np.random.randint(8, 20)
                wood_fraction = np.random.random()

                forest[x, y, 0] = int(total_resources * wood_fraction)
                forest[x, y, 1] = int(total_resources * (1 - wood_fraction))

        return forest


# Helper functions for easy game creation and testing
def create_test_forest_game(size: int = 4,
                            conflict_level: str = "medium",
                            max_steps: int = 8) -> ForestCollectionMDP:
    """Create a test forest game with specified parameters"""

    game = ForestCollectionMDP(
        grid_size=(size, size),
        leader_start=(0, 0),
        follower_start=(size - 1, size - 1),
        max_steps_leader=max_steps,
        max_steps_follower=max_steps,
        leader_starts_first=True
    )

    game.create_strategic_forest(conflict_level)
    return game


def run_sequential_game_demo():
    """Demonstrate the sequential forest collection game"""

    print("Sequential Forest Collection Game Demo")
    print("=" * 50)

    # Create test game
    game = create_test_forest_game(size=4, conflict_level="medium", max_steps=6)

    print(f"Game setup:")
    print(f"  Grid size: {game.width}x{game.height}")
    print(f"  Leader starts at: {game.leader_start}")
    print(f"  Follower starts at: {game.follower_start}")
    print(f"  Max steps each: {game.max_steps_leader}")

    # Analyze forest structure
    analysis = game.analyze_cooperation_vs_conflict()
    print(f"\nForest analysis:")
    print(f"  Cooperation ratio: {analysis['cooperation_ratio']:.2f}")
    print(f"  Conflict ratio: {analysis['conflict_ratio']:.2f}")
    print(f"  Resource correlation: {analysis['resource_correlation']:.3f}")

    # Show initial state
    initial_state = game.get_initial_state()
    print(f"\nInitial state:")
    print(f"  {initial_state}")
    print(f"  First player to move: {'Leader' if initial_state.turn else 'Follower'}")

    # Visualize
    game.visualize_forest(state=initial_state)

    return game


# Example usage
if __name__ == "__main__":
    # Create and demonstrate the game
    demo_game = run_sequential_game_demo()

    # Test basic transitions
    initial_state = demo_game.get_initial_state()
    print(f"\nTesting transitions:")

    # Leader moves right
    if initial_state.is_leader_turn():
        next_state = demo_game.transition(initial_state, Action.RIGHT)
        print(f"After leader moves RIGHT: {next_state}")

        # Now follower moves
        if next_state.is_follower_turn():
            next_state2 = demo_game.transition(next_state, Action.LEFT)
            print(f"After follower moves LEFT: {next_state2}")