"""
Small Test Case for Mixed Strategy Stackelberg Equilibrium
2x2 grid, 1 step each, designed so optimal strategy is mixed

This creates a scenario where the leader benefits from randomizing
to keep the follower uncertain about which resource to target.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt


class Action(Enum):
    UP = (0, 1)
    DOWN = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    STAY = (0, 0)


@dataclass(frozen=True)
class GameState:
    leader_pos: Tuple[int, int]
    follower_pos: Tuple[int, int]
    leader_steps_left: int
    follower_steps_left: int
    leader_total_wood: int = 0
    follower_total_fruit: int = 0
    turn: Optional[bool] = None  # True = leader, False = follower

    def is_terminal(self) -> bool:
        return self.leader_steps_left <= 0 and self.follower_steps_left <= 0

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


class SimpleFores2x2MDP:
    """
    Simple 2x2 forest designed to require mixed strategies

    Forest layout:
    [W=8,F=2] [W=1,F=9]
    [W=0,F=0] [W=7,F=3]

    Leader starts at (0,0), Follower starts at (1,1)
    Each player gets exactly 1 step

    This setup creates strategic tension:
    - If leader goes RIGHT to (1,0), follower's best response is UP to (1,1) then stay
    - If leader goes UP to (0,1), follower's best response is LEFT to (0,1) then stay
    - Leader benefits from mixing to keep follower uncertain
    """

    def __init__(self):
        self.width = 2
        self.height = 2

        # Strategic forest layout
        self.forest_map = np.zeros((2, 2, 2))

        # Cell (0,0): High wood, low fruit - good for leader
        self.forest_map[0, 0, 0] = 8  # Wood
        self.forest_map[0, 0, 1] = 2  # Fruit

        # Cell (1,0): Low wood, very high fruit - great for follower
        self.forest_map[1, 0, 0] = 1  # Wood
        self.forest_map[1, 0, 1] = 9  # Fruit

        # Cell (0,1): Medium wood, high fruit - contested
        self.forest_map[0, 1, 0] = 7  # Wood
        self.forest_map[0, 1, 1] = 3  # Fruit

        # Cell (1,1): Empty - no reward
        self.forest_map[1, 1, 0] = 0  # Wood
        self.forest_map[1, 1, 1] = 0  # Fruit

        self.leader_start = (0, 0)
        self.follower_start = (1, 1)

    def get_initial_state(self) -> GameState:
        return GameState(
            leader_pos=self.leader_start,
            follower_pos=self.follower_start,
            leader_steps_left=1,
            follower_steps_left=1,
            turn=True  # Leader starts first
        )

    def get_valid_actions(self, pos: Tuple[int, int]) -> List[Action]:
        x, y = pos
        valid = [Action.STAY]

        if x > 0: valid.append(Action.LEFT)
        if x < 1: valid.append(Action.RIGHT)
        if y > 0: valid.append(Action.DOWN)
        if y < 1: valid.append(Action.UP)

        return valid

    def is_valid_action(self, state: GameState, action: Action) -> bool:
        if state.is_terminal():
            return False
        pos = state.get_current_player_pos()
        return action in self.get_valid_actions(pos)

    def apply_action(self, pos: Tuple[int, int], action: Action) -> Tuple[int, int]:
        dx, dy = action.value
        new_x = max(0, min(1, pos[0] + dx))
        new_y = max(0, min(1, pos[1] + dy))
        return (new_x, new_y)

    def get_cell_rewards(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        x, y = pos
        return int(self.forest_map[x, y, 0]), int(self.forest_map[x, y, 1])

    def transition(self, state: GameState, action: Action) -> GameState:
        if state.is_terminal():
            return state

        if state.is_leader_turn():
            # Leader's turn
            new_pos = self.apply_action(state.leader_pos, action)
            wood_reward, _ = self.get_cell_rewards(new_pos)

            return GameState(
                leader_pos=new_pos,
                follower_pos=state.follower_pos,
                leader_steps_left=state.leader_steps_left - 1,
                follower_steps_left=state.follower_steps_left,
                leader_total_wood=state.leader_total_wood + wood_reward,
                follower_total_fruit=state.follower_total_fruit,
                turn=False  # Switch to follower
            )
        else:
            # Follower's turn
            new_pos = self.apply_action(state.follower_pos, action)
            _, fruit_reward = self.get_cell_rewards(new_pos)

            return GameState(
                leader_pos=state.leader_pos,
                follower_pos=new_pos,
                leader_steps_left=state.leader_steps_left,
                follower_steps_left=state.follower_steps_left - 1,
                leader_total_wood=state.leader_total_wood,
                follower_total_fruit=state.follower_total_fruit + fruit_reward,
                turn=True if state.leader_steps_left > 0 else None  # Switch back if leader has steps
            )

    def visualize_forest(self):
        """Visualize the forest setup"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Wood map
        wood_map = self.forest_map[:, :, 0].T
        im1 = ax1.imshow(wood_map, cmap='YlOrBr', origin='lower')
        ax1.set_title('Wood Distribution')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, f'W:{int(wood_map[i, j])}',
                                ha="center", va="center", color="black", fontsize=12, weight='bold')

        # Mark start positions
        ax1.scatter(0, 0, c='blue', s=200, marker='s', alpha=0.7, label='Leader start')
        ax1.scatter(1, 1, c='red', s=200, marker='^', alpha=0.7, label='Follower start')
        ax1.legend()
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        plt.colorbar(im1, ax=ax1)

        # Fruit map
        fruit_map = self.forest_map[:, :, 1].T
        im2 = ax2.imshow(fruit_map, cmap='Greens', origin='lower')
        ax2.set_title('Fruit Distribution')

        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax2.text(j, i, f'F:{int(fruit_map[i, j])}',
                                ha="center", va="center", color="black", fontsize=12, weight='bold')

        # Mark start positions
        ax2.scatter(0, 0, c='blue', s=200, marker='s', alpha=0.7, label='Leader start')
        ax2.scatter(1, 1, c='red', s=200, marker='^', alpha=0.7, label='Follower start')
        ax2.legend()
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        return fig


def analyze_pure_strategies():
    """Analyze what happens with pure strategies"""
    print("=" * 60)
    print("ANALYZING PURE STRATEGIES")
    print("=" * 60)

    game = SimpleFores2x2MDP()
    initial_state = game.get_initial_state()

    print("Forest Layout:")
    print("  (0,1): W=7, F=3    (1,1): W=0, F=0  <- Follower starts")
    print("  (0,0): W=8, F=2    (1,0): W=1, F=9")
    print("         ^")
    print("      Leader starts")
    print()

    # Analyze each pure strategy for leader
    leader_actions = [Action.STAY, Action.UP, Action.RIGHT]

    results = []

    for leader_action in leader_actions:
        print(f"If Leader plays {leader_action.name}:")

        # Apply leader action
        state_after_leader = game.transition(initial_state, leader_action)
        print(f"  Leader position: {state_after_leader.leader_pos}")
        print(f"  Leader wood so far: {state_after_leader.leader_total_wood}")

        # Find follower's best response
        follower_actions = game.get_valid_actions(state_after_leader.follower_pos)
        best_follower_action = None
        best_follower_utility = -1
        best_final_state = None

        print("  Follower's options:")
        for follower_action in follower_actions:
            final_state = game.transition(state_after_leader, follower_action)
            follower_utility = final_state.follower_total_fruit
            leader_utility = final_state.leader_total_wood

            print(f"    {follower_action.name}: Follower gets {follower_utility}, Leader gets {leader_utility}")

            if follower_utility > best_follower_utility:
                best_follower_utility = follower_utility
                best_follower_action = follower_action
                best_final_state = final_state

        print(f"  → Follower's best response: {best_follower_action.name}")
        print(
            f"  → Final outcome: Leader={best_final_state.leader_total_wood}, Follower={best_final_state.follower_total_fruit}")
        print()

        results.append((leader_action, best_final_state.leader_total_wood, best_final_state.follower_total_fruit))

    # Find best pure strategy for leader
    best_pure_leader_utility = max(results, key=lambda x: x[1])
    print(f"Best pure strategy for leader: {best_pure_leader_utility[0].name}")
    print(f"  Leader utility: {best_pure_leader_utility[1]}")
    print(f"  Follower utility: {best_pure_leader_utility[2]}")

    return results


def analyze_mixed_strategy():
    """Analyze potential mixed strategy"""
    print("\n" + "=" * 60)
    print("ANALYZING MIXED STRATEGY POTENTIAL")
    print("=" * 60)

    print("Key insight: If leader mixes between UP and RIGHT,")
    print("follower faces uncertainty about where to go.")
    print()

    print("Suppose leader mixes: p * UP + (1-p) * RIGHT")
    print()

    # If follower goes LEFT (to (0,1)):
    print("If follower responds with LEFT:")
    print("  When leader plays UP: both at (0,1) → conflict!")
    print("    Leader gets wood from (0,1): 7")
    print("    Follower gets fruit from (0,1): 3")
    print("  When leader plays RIGHT: leader at (1,0), follower at (0,1)")
    print("    Leader gets wood from (1,0): 1")
    print("    Follower gets fruit from (0,1): 3")
    print("  Expected leader utility: p*7 + (1-p)*1 = 1 + 6p")
    print()

    # If follower goes UP (to (1,0)):
    print("If follower responds with UP:")
    print("  When leader plays UP: leader at (0,1), follower at (1,0)")
    print("    Leader gets wood from (0,1): 7")
    print("    Follower gets fruit from (1,0): 9")
    print("  When leader plays RIGHT: both at (1,0) → conflict!")
    print("    Leader gets wood from (1,0): 1")
    print("    Follower gets fruit from (1,0): 9")
    print("  Expected leader utility: p*7 + (1-p)*1 = 1 + 6p")
    print()

    print("Interesting! Leader gets same expected utility (1 + 6p) regardless")
    print("of follower's response. This suggests leader can mix optimally!")
    print()

    # Follower's perspective
    print("Follower's expected utilities:")
    print("  If follower plays LEFT: expected fruit = p*3 + (1-p)*3 = 3")
    print("  If follower plays UP: expected fruit = p*9 + (1-p)*9 = 9")
    print()
    print("Follower will always choose UP (expected fruit = 9)")
    print("So leader's optimal mixing gives utility: 1 + 6p")
    print("Leader should set p = 1 (always UP) for utility = 7")
    print()

    print("Wait... let me reconsider the game structure...")
    return analyze_game_tree()


def analyze_game_tree():
    """Let's manually trace through the extensive form game tree"""
    print("\n" + "=" * 60)
    print("MANUAL GAME TREE ANALYSIS")
    print("=" * 60)

    game = SimpleFores2x2MDP()
    initial_state = game.get_initial_state()

    print("Building complete game tree...")
    print()

    # Leader's first move options from (0,0)
    leader_actions = [Action.STAY, Action.UP, Action.RIGHT]

    all_outcomes = []

    for leader_action in leader_actions:
        print(f"Leader plays {leader_action.name}:")
        state1 = game.transition(initial_state, leader_action)

        # Follower's response options from (1,1)
        follower_actions = game.get_valid_actions(state1.follower_pos)

        for follower_action in follower_actions:
            state2 = game.transition(state1, follower_action)

            print(f"  Then follower plays {follower_action.name}:")
            print(f"    Final: Leader={state2.leader_total_wood}, Follower={state2.follower_total_fruit}")

            all_outcomes.append({
                'leader_action': leader_action,
                'follower_action': follower_action,
                'leader_utility': state2.leader_total_wood,
                'follower_utility': state2.follower_total_fruit,
                'leader_final_pos': state2.leader_pos,
                'follower_final_pos': state2.follower_pos
            })

    print("\nAll possible outcomes:")
    print("Leader Action | Follower Action | Leader Utility | Follower Utility")
    print("-" * 65)
    for outcome in all_outcomes:
        print(f"{outcome['leader_action'].name:12} | {outcome['follower_action'].name:14} | "
              f"{outcome['leader_utility']:13} | {outcome['follower_utility']}")

    # Group by leader action to find follower best responses
    print("\nFollower's best responses:")
    from collections import defaultdict
    by_leader_action = defaultdict(list)

    for outcome in all_outcomes:
        by_leader_action[outcome['leader_action']].append(outcome)

    stackelberg_outcomes = []

    for leader_action, outcomes in by_leader_action.items():
        # Find follower's best response (highest follower utility)
        best_for_follower = max(outcomes, key=lambda x: x['follower_utility'])
        stackelberg_outcomes.append(best_for_follower)
        print(
            f"If leader plays {leader_action.name}, follower best response is {best_for_follower['follower_action'].name}")
        print(
            f"  → Outcome: Leader={best_for_follower['leader_utility']}, Follower={best_for_follower['follower_utility']}")

    # Find leader's best commitment
    best_stackelberg = max(stackelberg_outcomes, key=lambda x: x['leader_utility'])
    print(f"\nBest pure strategy commitment for leader:")
    print(f"  Action: {best_stackelberg['leader_action'].name}")
    print(f"  Outcome: Leader={best_stackelberg['leader_utility']}, Follower={best_stackelberg['follower_utility']}")

    return all_outcomes, stackelberg_outcomes


def create_mixed_strategy_scenario():
    """Create a scenario where mixed strategy is definitely better"""
    print("\n" + "=" * 60)
    print("CREATING BETTER MIXED STRATEGY SCENARIO")
    print("=" * 60)

    # Let me modify the forest to create a true mixed strategy scenario
    class MixedStrategyForest(SimpleFores2x2MDP):
        def __init__(self):
            super().__init__()

            # New strategic layout where mixing helps
            self.forest_map = np.zeros((2, 2, 2))

            # Key insight: Create asymmetric payoffs that reward unpredictability

            # Cell (0,0): Starting position - moderate wood, no fruit
            self.forest_map[0, 0, 0] = 3  # Wood
            self.forest_map[0, 0, 1] = 0  # Fruit

            # Cell (1,0): High wood if uncontested, but follower can grab high fruit
            self.forest_map[1, 0, 0] = 10  # Wood
            self.forest_map[1, 0, 1] = 8  # Fruit

            # Cell (0,1): Moderate wood, low fruit
            self.forest_map[0, 1, 0] = 6  # Wood
            self.forest_map[0, 1, 1] = 2  # Fruit

            # Cell (1,1): Low wood, very high fruit
            self.forest_map[1, 1, 0] = 1  # Wood
            self.forest_map[1, 1, 1] = 12  # Fruit

    game = MixedStrategyForest()
    initial_state = game.get_initial_state()

    print("Modified Forest Layout:")
    print("  (0,1): W=6, F=2     (1,1): W=1, F=12  <- Follower starts")
    print("  (0,0): W=3, F=0     (1,0): W=10, F=8")
    print("         ^")
    print("      Leader starts")
    print()

    # Analyze this new scenario
    print("Pure strategy analysis:")

    outcomes = []
    for leader_action in [Action.STAY, Action.UP, Action.RIGHT]:
        state1 = game.transition(initial_state, leader_action)

        # Find follower's best response
        best_follower_utility = -1
        best_outcome = None

        for follower_action in game.get_valid_actions(state1.follower_pos):
            state2 = game.transition(state1, follower_action)
            if state2.follower_total_fruit > best_follower_utility:
                best_follower_utility = state2.follower_total_fruit
                best_outcome = state2

        outcomes.append((leader_action, best_outcome.leader_total_wood, best_outcome.follower_total_fruit))
        print(
            f"Leader {leader_action.name}: Leader gets {best_outcome.leader_total_wood}, Follower gets {best_outcome.follower_total_fruit}")

    best_pure = max(outcomes, key=lambda x: x[1])
    print(f"\nBest pure strategy: {best_pure[0].name} → Leader utility = {best_pure[1]}")

    # Now show why mixing might be better
    print(f"\nWhy mixing could help:")
    print(f"The follower is trying to predict leader's move to optimize their own collection.")
    print(f"If leader could make follower uncertain, leader might do better.")

    game.visualize_forest()
    plt.title("Modified Forest for Mixed Strategy Testing")
    plt.show()

    return game


def run_mixed_strategy_test():
    """Run the complete test"""
    print("MIXED STRATEGY STACKELBERG TEST CASE")
    print("=" * 60)

    # Create and visualize the game
    game = SimpleFores2x2MDP()
    fig = game.visualize_forest()
    plt.show()

    # Analyze pure strategies
    pure_results = analyze_pure_strategies()

    # Analyze mixed potential
    analyze_mixed_strategy()

    # Analyze game tree
    all_outcomes, stackelberg_outcomes = analyze_game_tree()

    # Try modified scenario
    modified_game = create_mixed_strategy_scenario()

    print(f"\n" + "=" * 60)
    print("SUMMARY FOR MANUAL TESTING")
    print("=" * 60)
    print(f"Game: 2x2 forest, 1 step each")
    print(f"Leader starts at (0,0), Follower starts at (1,1)")
    print(
        f"Best pure strategy gives Leader utility: {max(stackelberg_outcomes, key=lambda x: x['leader_utility'])['leader_utility']}")
    print(f"\nTo test mixed strategy algorithm:")
    print(f"1. Run the algorithm on this simple case")
    print(f"2. Check if it finds a mixed strategy")
    print(f"3. Compare with pure strategy result")
    print(f"4. If mixed strategy isn't better, try the modified forest")

    return game, modified_game


if __name__ == "__main__":
    game, modified_game = run_mixed_strategy_test()