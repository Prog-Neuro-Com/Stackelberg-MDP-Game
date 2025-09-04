import pytest
import numpy as np
from src.forest_game import ForestCollectionMDP, GameState, Action


def test_forest_initialization():
    """Test basic forest game initialization"""
    game = ForestCollectionMDP(grid_size=(3, 3))
    assert game.width == 3
    assert game.height == 3
    assert game.forest_map.shape == (3, 3, 2)


def test_state_transitions():
    """Test state transitions work correctly"""
    game = ForestCollectionMDP(grid_size=(3, 3))
    initial_state = game.get_initial_state()

    # Test moving right
    new_state = game.transition(initial_state, Action.RIGHT, Action.STAY)
    assert new_state.leader_pos == (1, 0)
    assert new_state.follower_pos == (0, 0)
    assert new_state.leader_steps_left == game.max_steps_leader - 1


def test_terminal_states():
    """Test terminal state detection"""
    state = GameState(
        leader_pos=(0, 0),
        follower_pos=(1, 1),
        leader_steps_left=0,
        follower_steps_left=5
    )
    assert state.is_terminal()


if __name__ == "__main__":
    pytest.main([__file__])