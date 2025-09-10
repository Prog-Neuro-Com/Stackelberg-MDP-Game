"""
Test Cases for EPF and DP Solvers - Small Scale Forest Games
Each test case is designed to be manually verifiable with 2 steps per player
"""

import pytest
import numpy as np
from src.forest_game import ForestCollectionMDP, Action
from src.solvers.epf_solver import solve_forest_epf
from src.solvers.dynamic_programming_tree_solver import solve_forest_stackelberg


def create_test_case_1_simple_cooperation():
    """
    Test Case 1: Simple Cooperation
    A 2x2 grid where cooperation benefits both players

    Grid layout:
    (0,1) W=5, F=5    (1,1) W=2, F=8
    (0,0) W=8, F=2    (1,0) W=1, F=1

    Leader starts at (0,0), Follower starts at (1,1)
    Expected: Both should move to mutually beneficial cells
    """
    forest_map = np.zeros((2, 2, 2))

    # (0,0): High wood, low fruit - Leader's preferred
    forest_map[0, 0] = [8, 2]

    # (1,1): Low wood, high fruit - Follower's preferred
    forest_map[1, 1] = [2, 8]

    # (0,1): Balanced resources - Cooperation zone
    forest_map[0, 1] = [5, 5]

    # (1,0): Poor for both
    forest_map[1, 0] = [1, 1]

    mdp = ForestCollectionMDP(
        grid_size=(2, 2),
        forest_map=forest_map,
        leader_start=(0, 0),
        follower_start=(1, 1),
        max_steps_leader=2,
        max_steps_follower=2
    )

    return mdp, "Simple Cooperation Test"


def create_test_case_2_pure_conflict():
    """
    Test Case 2: Pure Conflict
    Resources are zero-sum: good for leader = bad for follower

    Grid layout:
    (0,1) W=10, F=0   (1,1) W=0, F=10
    (0,0) W=5, F=5    (1,0) W=3, F=3

    Tests how EPF handles pure conflict situations
    """
    forest_map = np.zeros((2, 2, 2))

    # (0,0): Starting position - moderate for both
    forest_map[0, 0] = [5, 5]

    # (0,1): Perfect for leader, terrible for follower
    forest_map[0, 1] = [10, 0]

    # (1,0): Mediocre for both
    forest_map[1, 0] = [3, 3]

    # (1,1): Perfect for follower, terrible for leader
    forest_map[1, 1] = [0, 10]

    mdp = ForestCollectionMDP(
        grid_size=(2, 2),
        forest_map=forest_map,
        leader_start=(0, 0),
        follower_start=(0, 0),  # Both start at same position
        max_steps_leader=2,
        max_steps_follower=2
    )

    return mdp, "Pure Conflict Test"


def create_test_case_3_threat_effectiveness():
    """
    Test Case 3: Threat Effectiveness
    Tests whether leader can use credible threats

    Grid layout:
    (0,1) W=1, F=9    (1,1) W=6, F=6
    (0,0) W=4, F=4    (1,0) W=9, F=1

    Leader can threaten to go to (1,0) which is bad for follower
    This should make follower cooperate more
    """
    forest_map = np.zeros((2, 2, 2))

    # (0,0): Starting - moderate
    forest_map[0, 0] = [4, 4]

    # (0,1): Great for follower, bad for leader
    forest_map[0, 1] = [1, 9]

    # (1,0): Great for leader, bad for follower (THREAT ZONE)
    forest_map[1, 0] = [9, 1]

    # (1,1): Good compromise
    forest_map[1, 1] = [6, 6]

    mdp = ForestCollectionMDP(
        grid_size=(2, 2),
        forest_map=forest_map,
        leader_start=(0, 0),
        follower_start=(0, 0),
        max_steps_leader=2,
        max_steps_follower=2
    )

    return mdp, "Threat Effectiveness Test"


def create_test_case_4_asymmetric_steps():
    """
    Test Case 4: Asymmetric Steps
    Leader has 2 steps, Follower has 1 step
    Tests how step advantage affects EPF

    Grid layout:
    (0,1) W=3, F=7    (1,1) W=7, F=3
    (0,0) W=5, F=5    (1,0) W=2, F=8
    """
    forest_map = np.zeros((2, 2, 2))

    forest_map[0, 0] = [5, 5]  # Start
    forest_map[0, 1] = [3, 7]  # Follower-favored
    forest_map[1, 0] = [2, 8]  # Strong follower
    forest_map[1, 1] = [7, 3]  # Leader-favored

    mdp = ForestCollectionMDP(
        grid_size=(2, 2),
        forest_map=forest_map,
        leader_start=(0, 0),
        follower_start=(0, 0),
        max_steps_leader=2,
        max_steps_follower=1  # Asymmetric steps
    )

    return mdp, "Asymmetric Steps Test"


def create_test_case_5_corner_dominance():
    """
    Test Case 5: Corner Dominance
    One corner is strictly dominant, testing EPF convergence

    Grid layout:
    (0,1) W=2, F=2    (1,1) W=15, F=15
    (0,0) W=3, F=3    (1,0) W=1, F=1

    (1,1) dominates everything - both players should want to go there
    Tests if EPF correctly identifies dominant strategies
    """
    forest_map = np.zeros((2, 2, 2))

    forest_map[0, 0] = [3, 3]  # Start - okay
    forest_map[0, 1] = [2, 2]  # Poor
    forest_map[1, 0] = [1, 1]  # Terrible
    forest_map[1, 1] = [15, 15]  # DOMINANT - great for both

    mdp = ForestCollectionMDP(
        grid_size=(2, 2),
        forest_map=forest_map,
        leader_start=(0, 0),
        follower_start=(1, 0),  # Different starting positions
        max_steps_leader=2,
        max_steps_follower=2
    )

    return mdp, "Corner Dominance Test"


def manual_verification_helper(mdp: ForestCollectionMDP, test_name: str):
    """
    Helper function to provide manual verification guidance
    """
    print(f"\n{'=' * 50}")
    print(f"MANUAL VERIFICATION: {test_name}")
    print(f"{'=' * 50}")

    print("\nForest Map Layout:")
    for y in range(mdp.height - 1, -1, -1):  # Top to bottom
        for x in range(mdp.width):
            wood = mdp.forest_map[x, y, 0]
            fruit = mdp.forest_map[x, y, 1]
            print(f"({x},{y}): W={wood:.0f}, F={fruit:.0f}", end="  ")
        print()

    print(f"\nStarting Positions:")
    print(f"Leader: {mdp.leader_start}")
    print(f"Follower: {mdp.follower_start}")
    print(f"Steps: Leader={mdp.max_steps_leader}, Follower={mdp.max_steps_follower}")

    initial_state = mdp.get_initial_state()
    print(f"Who goes first: {'Leader' if initial_state.turn else 'Follower'}")

    print(f"\nActions available: {[action.name for action in Action]}")
    print("Action effects: UP=(0,+1), DOWN=(0,-1), LEFT=(-1,0), RIGHT=(+1,0), STAY=(0,0)")

    print(f"\nTo manually verify:")
    print("1. Draw the game tree (2 levels deep for each player)")
    print("2. Compute payoffs at each leaf")
    print("3. Apply backward induction")
    print("4. For EPF: find all enforceable (follower_payoff, leader_payoff) combinations")

    return initial_state


def test_simple_cooperation():
    """Test Case 1: Simple Cooperation - Both players should find mutually beneficial cells"""
    mdp, test_name = create_test_case_1_simple_cooperation()
    
    # Run DP solver
    leader_payoff, follower_payoff, actions = solve_forest_stackelberg(mdp, max_depth=10)
    
    # Basic assertions
    assert leader_payoff > 0, "Leader should get positive payoff"
    assert follower_payoff > 0, "Follower should get positive payoff"
    assert len(actions) > 0, "Should have action sequence"
    
    # Run EPF solver
    epf, analysis = solve_forest_epf(mdp, max_depth=10)
    
    assert len(epf) > 0, "EPF should contain at least one point"
    assert 'optimal_point' in analysis, "Analysis should contain optimal point"
    
    # DP solution should be in EPF
    dp_point = (follower_payoff, leader_payoff)
    assert dp_point in epf or any(abs(dp_point[0] - p[0]) < 0.01 and abs(dp_point[1] - p[1]) < 0.01 for p in epf), \
        "DP solution should be in EPF set"


def test_pure_conflict():
    """Test Case 2: Pure Conflict - Zero-sum resources"""
    mdp, test_name = create_test_case_2_pure_conflict()
    
    # Run both solvers
    leader_payoff, follower_payoff, actions = solve_forest_stackelberg(mdp, max_depth=10)
    epf, analysis = solve_forest_epf(mdp, max_depth=10)
    
    # Basic assertions
    assert len(actions) > 0, "Should have action sequence"
    assert len(epf) > 0, "EPF should contain at least one point"
    
    # In pure conflict, leader should get better outcome than follower (Stackelberg advantage)
    assert leader_payoff >= follower_payoff, "Leader should have Stackelberg advantage"


def test_threat_effectiveness():
    """Test Case 3: Threat Effectiveness - Leader can use credible threats"""
    mdp, test_name = create_test_case_3_threat_effectiveness()
    
    # Run both solvers
    leader_payoff, follower_payoff, actions = solve_forest_stackelberg(mdp, max_depth=10)
    epf, analysis = solve_forest_epf(mdp, max_depth=10)
    
    # Basic assertions
    assert len(actions) > 0, "Should have action sequence"
    assert len(epf) > 0, "EPF should contain at least one point"
    assert leader_payoff > 0, "Leader should achieve positive payoff through threats"


def test_asymmetric_steps():
    """Test Case 4: Asymmetric Steps - Leader has 2 steps, Follower has 1"""
    mdp, test_name = create_test_case_4_asymmetric_steps()
    
    # Run both solvers
    leader_payoff, follower_payoff, actions = solve_forest_stackelberg(mdp, max_depth=10)
    epf, analysis = solve_forest_epf(mdp, max_depth=10)
    
    # Basic assertions
    assert len(actions) > 0, "Should have action sequence"
    assert len(epf) > 0, "EPF should contain at least one point"
    
    # Leader should benefit from having more steps
    assert leader_payoff > 0, "Leader should benefit from step advantage"


def test_corner_dominance():
    """Test Case 5: Corner Dominance - One corner dominates all other options"""
    mdp, test_name = create_test_case_5_corner_dominance()
    
    # Run both solvers
    leader_payoff, follower_payoff, actions = solve_forest_stackelberg(mdp, max_depth=10)
    epf, analysis = solve_forest_epf(mdp, max_depth=10)
    
    # Basic assertions
    assert len(actions) > 0, "Should have action sequence"
    assert len(epf) > 0, "EPF should contain at least one point"
    
    # Both players should get high payoffs from the dominant corner
    assert leader_payoff > 10, "Leader should reach high-value corner"
    assert follower_payoff > 10, "Follower should reach high-value corner"


def test_epf_contains_dp_solution():
    """Test that EPF always contains the DP solution for all test cases"""
    test_cases = [
        create_test_case_1_simple_cooperation,
        create_test_case_2_pure_conflict,
        create_test_case_3_threat_effectiveness,
        create_test_case_4_asymmetric_steps,
        create_test_case_5_corner_dominance
    ]
    
    for test_func in test_cases:
        mdp, test_name = test_func()
        
        # Run both solvers
        leader_payoff, follower_payoff, actions = solve_forest_stackelberg(mdp, max_depth=10)
        epf, analysis = solve_forest_epf(mdp, max_depth=10)
        
        # DP solution should be in EPF (allowing small numerical tolerance)
        dp_point = (follower_payoff, leader_payoff)
        found_in_epf = any(
            abs(dp_point[0] - p[0]) < 0.01 and abs(dp_point[1] - p[1]) < 0.01 
            for p in epf
        )
        assert found_in_epf, f"DP solution {dp_point} not found in EPF for {test_name}"


def test_solver_consistency():
    """Test that both solvers produce consistent results across multiple runs"""
    mdp, _ = create_test_case_1_simple_cooperation()
    
    # Run DP solver multiple times
    results = []
    for _ in range(3):
        leader_payoff, follower_payoff, actions = solve_forest_stackelberg(mdp, max_depth=10)
        results.append((leader_payoff, follower_payoff))
    
    # Results should be consistent
    first_result = results[0]
    for result in results[1:]:
        assert abs(result[0] - first_result[0]) < 0.01, "DP solver should be deterministic"
        assert abs(result[1] - first_result[1]) < 0.01, "DP solver should be deterministic"


@pytest.fixture
def sample_mdp():
    """Fixture providing a simple MDP for testing"""
    mdp, _ = create_test_case_1_simple_cooperation()
    return mdp


def test_mdp_basic_properties(sample_mdp):
    """Test basic properties of the MDP"""
    initial_state = sample_mdp.get_initial_state()
    
    assert initial_state is not None, "Should have initial state"
    assert hasattr(sample_mdp, 'forest_map'), "Should have forest map"
    assert sample_mdp.forest_map.shape[2] == 2, "Forest map should have 2 resource types"