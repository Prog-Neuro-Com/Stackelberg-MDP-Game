from typing import List
import numpy as np
from matplotlib import pyplot as plt
from src.forest_game import GameState, Action, ForestCollectionMDP


def visualize_solution(actions: List[Action], initial_state: GameState, mdp: ForestCollectionMDP, title: str = "Game Solution"):
    """
    Visualize the solution path on the forest grid as a standalone function
    
    Args:
        actions: List of actions taken during the game
        initial_state: Initial game state
        mdp: The ForestCollectionMDP instance
        title: Optional title for the visualization
    """
    # Simulate to get full trajectory
    states = [initial_state]
    current_state = initial_state

    for action in actions:
        if current_state.is_terminal():
            break
        current_state = mdp.transition(current_state, action)
        states.append(current_state)

    # Determine who starts first
    first_player = "Leader" if initial_state.turn is True else "Follower" if initial_state.turn is False else "Simultaneous"
    
    # Create visualization with turn information
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Forest with paths
    wood_map = mdp.forest_map[:, :, 0].T
    fruit_map = mdp.forest_map[:, :, 1].T

    # Show combined resource map
    combined_map = wood_map + fruit_map
    im = ax1.imshow(combined_map, cmap='YlOrBr', origin='lower', alpha=0.7)

    # Extract leader and follower paths
    leader_path = [(state.leader_pos[0], state.leader_pos[1]) for state in states]
    follower_path = [(state.follower_pos[0], state.follower_pos[1]) for state in states]

    # Plot paths with turn indicators
    leader_x, leader_y = zip(*leader_path)
    follower_x, follower_y = zip(*follower_path)

    ax1.plot(leader_x, leader_y, 'b-o', linewidth=3, markersize=8,
             label='Leader Path', alpha=0.8)
    ax1.plot(follower_x, follower_y, 'r-s', linewidth=3, markersize=8,
             label='Follower Path', alpha=0.8)

    # Mark starting positions with special indicators for who goes first
    if first_player == "Leader":
        ax1.plot(leader_x[0], leader_y[0], 'b*', markersize=16, label='Leader Start (1st)')
        ax1.plot(follower_x[0], follower_y[0], 'ro', markersize=12, label='Follower Start (2nd)')
    elif first_player == "Follower":
        ax1.plot(leader_x[0], leader_y[0], 'bo', markersize=12, label='Leader Start (2nd)')
        ax1.plot(follower_x[0], follower_y[0], 'r*', markersize=16, label='Follower Start (1st)')
    else:
        ax1.plot(leader_x[0], leader_y[0], 'bo', markersize=12, label='Leader Start')
        ax1.plot(follower_x[0], follower_y[0], 'ro', markersize=12, label='Follower Start')

    # Add turn sequence annotations
    for i, state in enumerate(states[1:], 1):
        if i < len(states) - 1:  # Don't annotate the last state
            current_turn = "L" if state.turn is True else "F" if state.turn is False else "S"
            # Add small turn indicators along the paths
            if state.turn is True and i < len(leader_x):
                ax1.annotate(f'{i}L', (leader_x[i], leader_y[i]), 
                           xytext=(3, 3), textcoords='offset points', fontsize=8, color='blue')
            elif state.turn is False and i < len(follower_x):
                ax1.annotate(f'{i}F', (follower_x[i], follower_y[i]), 
                           xytext=(3, 3), textcoords='offset points', fontsize=8, color='red')

    enhanced_title = f"{title} ({first_player} starts first)"
    ax1.set_title(enhanced_title)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Payoffs over time with turn indicators
    leader_payoffs = [state.leader_total_wood for state in states]
    follower_payoffs = [state.follower_total_fruit for state in states]

    steps = list(range(len(states)))
    ax2.plot(steps, leader_payoffs, 'b-o', linewidth=2, label='Leader (Wood)')
    ax2.plot(steps, follower_payoffs, 'r-s', linewidth=2, label='Follower (Fruit)')

    # Add vertical lines to show when each player moves
    for i, state in enumerate(states[:-1]):  # Skip last state since no move follows it
        if state.turn is True:  # Leader's turn
            ax2.axvline(x=i, color='blue', alpha=0.3, linestyle='--', linewidth=1)
        elif state.turn is False:  # Follower's turn
            ax2.axvline(x=i, color='red', alpha=0.3, linestyle='--', linewidth=1)

    ax2.set_xlabel('Game Step')
    ax2.set_ylabel('Cumulative Reward')
    payoff_title = f'Payoff Accumulation ({first_player} starts)'
    ax2.set_title(payoff_title)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add final game info as text
    final_state = states[-1]
    info_text = f"Final: Leader={final_state.leader_total_wood}, Follower={final_state.follower_total_fruit}\n"
    info_text += f"Steps: Leader={initial_state.leader_steps_left - final_state.leader_steps_left}, "
    info_text += f"Follower={initial_state.follower_steps_left - final_state.follower_steps_left}"
    
    fig.suptitle(info_text, y=0.02, fontsize=10, ha='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for the info text
    plt.show()