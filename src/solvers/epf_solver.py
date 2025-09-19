"""
Correct EPF-Based Mixed Strategy Stackelberg Solver for Forest Collection Game

This implements the proper approach using Enforceable Payoff Frontiers (EPFs)
as piecewise linear concave functions with:
- Upper concave envelope (∨) operations for leader nodes
- Left-truncation (⊳) operations for follower nodes
- Proper handling of mixed strategies through convex combinations

Based on the SEFCE algorithm from:
- Bosansky et al. (2017): "Algorithmic Game Theory for Security"
- Function Approximation paper in your project
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
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

@dataclass
class EPFPoint:
    """Point on EPF: (follower_payoff, leader_payoff)"""
    follower_payoff: float
    leader_payoff: float

    def __str__(self):
        return f"({self.follower_payoff:.2f}, {self.leader_payoff:.2f})"

    def __eq__(self, other):
        if not isinstance(other, EPFPoint):
            return False
        return (abs(self.follower_payoff - other.follower_payoff) < 1e-6 and
                abs(self.leader_payoff - other.leader_payoff) < 1e-6)

    def __hash__(self):
        return hash((round(self.follower_payoff, 6), round(self.leader_payoff, 6)))

class EPF:
    """
    Enforceable Payoff Frontier as piecewise linear concave function
    Represents Us(μ2) = maximum leader payoff when follower gets μ2
    """

    def __init__(self, points: List[EPFPoint]):
        self.points = self._process_points(points)

    def _process_points(self, points: List[EPFPoint]) -> List[EPFPoint]:
        """Process points: remove duplicates, sort, ensure concavity"""
        if not points:
            return []

        # Remove duplicates
        unique_points = []
        seen = set()
        for point in points:
            if point not in seen:
                unique_points.append(point)
                seen.add(point)

        # Sort by follower payoff
        unique_points.sort(key=lambda p: p.follower_payoff)

        # Compute upper concave envelope
        return self._upper_concave_envelope(unique_points)

    def _upper_concave_envelope(self, points: List[EPFPoint]) -> List[EPFPoint]:
        """Compute upper concave envelope using Andrew's monotone chain algorithm"""
        if len(points) <= 1:
            return points

        # Build upper hull
        upper_hull = []
        for point in points:
            # Remove points that make the hull non-concave
            while (len(upper_hull) >= 2 and
                   self._cross_product(upper_hull[-2], upper_hull[-1], point) <= 0):
                upper_hull.pop()
            upper_hull.append(point)

        return upper_hull

    def _cross_product(self, o: EPFPoint, a: EPFPoint, b: EPFPoint) -> float:
        """Cross product for concavity check"""
        return ((a.follower_payoff - o.follower_payoff) * (b.leader_payoff - o.leader_payoff) -
                (a.leader_payoff - o.leader_payoff) * (b.follower_payoff - o.follower_payoff))

    def evaluate(self, follower_payoff: float) -> float:
        """Evaluate EPF at given follower payoff level"""
        if not self.points:
            return float('-inf')

        # Handle boundary cases
        if follower_payoff < self.points[0].follower_payoff:
            return float('-inf')
        if follower_payoff > self.points[-1].follower_payoff:
            return float('-inf')

        # Find the segment containing follower_payoff
        for i in range(len(self.points) - 1):
            p1, p2 = self.points[i], self.points[i + 1]

            if p1.follower_payoff <= follower_payoff <= p2.follower_payoff:
                # Linear interpolation
                if abs(p2.follower_payoff - p1.follower_payoff) < 1e-9:
                    return max(p1.leader_payoff, p2.leader_payoff)

                t = (follower_payoff - p1.follower_payoff) / (p2.follower_payoff - p1.follower_payoff)
                return p1.leader_payoff + t * (p2.leader_payoff - p1.leader_payoff)

        # Exact match
        for point in self.points:
            if abs(point.follower_payoff - follower_payoff) < 1e-9:
                return point.leader_payoff

        return float('-inf')

    def left_truncate(self, threshold: float) -> 'EPF':
        """Left-truncation: EPF⊳t keeps only μ2 ≥ threshold"""
        if not self.points:
            return EPF([])

        truncated_points = []

        for i, point in enumerate(self.points):
            if point.follower_payoff >= threshold - 1e-9:
                truncated_points.append(point)
            elif i < len(self.points) - 1:
                # Check if threshold crosses this segment
                next_point = self.points[i + 1]
                if point.follower_payoff < threshold < next_point.follower_payoff:
                    # Interpolate at threshold
                    leader_at_threshold = self.evaluate(threshold)
                    if leader_at_threshold > float('-inf'):
                        truncated_points.append(EPFPoint(threshold, leader_at_threshold))

        return EPF(truncated_points)

    def get_max_leader_payoff(self) -> float:
        """Get maximum leader payoff achievable"""
        if not self.points:
            return float('-inf')
        return max(p.leader_payoff for p in self.points)

    def get_optimal_point(self) -> Optional[EPFPoint]:
        """Get point that maximizes leader payoff"""
        if not self.points:
            return None
        return max(self.points, key=lambda p: p.leader_payoff)

    def get_domain(self) -> Tuple[float, float]:
        """Get domain of EPF"""
        if not self.points:
            return (0.0, 0.0)
        return (self.points[0].follower_payoff, self.points[-1].follower_payoff)

    def __str__(self):
        return f"EPF[{', '.join(str(p) for p in self.points)}]"

def upper_concave_envelope_multiple(epfs: List[EPF]) -> EPF:
    """Compute upper concave envelope of multiple EPFs (∨ operation)"""
    if not epfs:
        return EPF([])
    if len(epfs) == 1:
        return epfs[0]

    # Collect all points from all EPFs
    all_points = []
    for epf in epfs:
        all_points.extend(epf.points)

    # Also add intersections between EPFs
    for i in range(len(epfs)):
        for j in range(i + 1, len(epfs)):
            intersections = find_epf_intersections(epfs[i], epfs[j])
            all_points.extend(intersections)

    # Compute envelope
    if not all_points:
        return EPF([])

    # Sort by follower payoff
    all_points.sort(key=lambda p: p.follower_payoff)

    # For each follower payoff level, keep the point with highest leader payoff
    envelope_points = []
    current_follower = None
    current_max_leader = float('-inf')
    current_best_point = None

    for point in all_points:
        if current_follower is None or abs(point.follower_payoff - current_follower) > 1e-9:
            # New follower payoff level
            if current_best_point is not None:
                envelope_points.append(current_best_point)
            current_follower = point.follower_payoff
            current_max_leader = point.leader_payoff
            current_best_point = point
        else:
            # Same follower payoff level, keep best leader payoff
            if point.leader_payoff > current_max_leader:
                current_max_leader = point.leader_payoff
                current_best_point = point

    if current_best_point is not None:
        envelope_points.append(current_best_point)

    return EPF(envelope_points)

def find_epf_intersections(epf1: EPF, epf2: EPF) -> List[EPFPoint]:
    """Find intersection points between two EPFs"""
    intersections = []

    # Check intersections between all segment pairs
    for i in range(len(epf1.points) - 1):
        for j in range(len(epf2.points) - 1):
            p1a, p1b = epf1.points[i], epf1.points[i + 1]
            p2a, p2b = epf2.points[j], epf2.points[j + 1]

            intersection = line_intersection(p1a, p1b, p2a, p2b)
            if intersection is not None:
                intersections.append(intersection)

    return intersections

def line_intersection(p1a: EPFPoint, p1b: EPFPoint, p2a: EPFPoint, p2b: EPFPoint) -> Optional[EPFPoint]:
    """Find intersection of two line segments"""
    x1, y1 = p1a.follower_payoff, p1a.leader_payoff
    x2, y2 = p1b.follower_payoff, p1b.leader_payoff
    x3, y3 = p2a.follower_payoff, p2a.leader_payoff
    x4, y4 = p2b.follower_payoff, p2b.leader_payoff

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None  # Parallel lines

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        # Intersection point
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return EPFPoint(ix, iy)

    return None

class CorrectEPFMixedSolver:
    """Correct EPF-based mixed strategy Stackelberg solver"""

    def __init__(self, forest_mdp, max_depth: int = 10):
        self.mdp = forest_mdp
        self.max_depth = max_depth
        self.epf_cache: Dict[GameState, EPF] = {}
        self.grim_values: Dict[GameState, float] = {}  # V_underline(s)
        self.altruistic_values: Dict[GameState, float] = {}  # V_overline(s)
        self.debug = True

    def solve(self, initial_state: Optional[GameState] = None) -> Dict:
        """Solve for EPF using correct mixed strategy approach"""
        if initial_state is None:
            initial_state = self.mdp.get_initial_state()

        print("="*60)
        print("CORRECT EPF-BASED MIXED STRATEGY STACKELBERG SOLVER")
        print("="*60)

        # Phase 1: Compute grim and altruistic values
        print("Phase 1: Computing boundary values (grim and altruistic)...")
        self._compute_boundary_values(initial_state)

        # Phase 2: Compute EPFs via backward induction
        print("\nPhase 2: Computing EPFs via backward induction...")
        root_epf = self._compute_epf(initial_state, depth=0)

        print(f"\nRoot EPF: {root_epf}")
        print(f"Number of knots in root EPF: {len(root_epf.points)}")

        # Phase 3: Extract optimal outcome and strategy
        optimal_point = root_epf.get_optimal_point()
        if optimal_point is None:
            raise ValueError("No feasible outcomes found")

        print(f"Optimal outcome: {optimal_point}")
        max_leader_payoff = root_epf.get_max_leader_payoff()

        return {
            'root_epf': root_epf,
            'optimal_point': optimal_point,
            'max_leader_payoff': max_leader_payoff,
            'epf_cache': self.epf_cache.copy(),
            'num_epf_points': len(root_epf.points),
            'domain': root_epf.get_domain()
        }

    def _compute_boundary_values(self, initial_state: GameState):
        """Compute grim V_underline(s) and altruistic V_overline(s) values"""
        visited = set()

        def dfs(state: GameState, depth: int):
            if depth > self.max_depth or state in visited:
                return
            visited.add(state)

            if state.is_terminal():
                # Terminal state: follower gets their collected fruit
                self.grim_values[state] = float(state.follower_total_fruit)
                self.altruistic_values[state] = float(state.follower_total_fruit)
                return

            # Recursively process children first
            children = []
            if state.is_leader_turn():
                valid_actions = self.mdp.get_valid_actions(state.leader_pos)
            else:
                valid_actions = self.mdp.get_valid_actions(state.follower_pos)

            for action in valid_actions:
                if self.mdp.is_valid_action(state, action):
                    next_state = self.mdp.transition(state, action)
                    dfs(next_state, depth + 1)
                    children.append(next_state)

            # Compute values based on children
            if children:
                child_values = [self.grim_values.get(child, 0.0) for child in children]
                child_alt_values = [self.altruistic_values.get(child, 0.0) for child in children]

                if state.is_follower_turn():
                    # Follower maximizes their own payoff
                    self.grim_values[state] = max(child_values) if child_values else 0.0
                    self.altruistic_values[state] = max(child_alt_values) if child_alt_values else 0.0
                else:
                    # Leader chooses - for grim, minimize follower; for altruistic, maximize follower
                    self.grim_values[state] = min(child_values) if child_values else 0.0
                    self.altruistic_values[state] = max(child_alt_values) if child_alt_values else 0.0
            else:
                self.grim_values[state] = float(state.follower_total_fruit)
                self.altruistic_values[state] = float(state.follower_total_fruit)

        dfs(initial_state, 0)

        if self.debug:
            print(f"Computed boundary values for {len(self.grim_values)} states")

    def _compute_epf(self, state: GameState, depth: int) -> EPF:
        """Compute EPF for given state"""
        if state in self.epf_cache:
            return self.epf_cache[state]

        if depth > self.max_depth:
            # Fallback EPF
            terminal_point = EPFPoint(
                follower_payoff=float(state.follower_total_fruit),
                leader_payoff=float(state.leader_total_wood)
            )
            epf = EPF([terminal_point])
            self.epf_cache[state] = epf
            return epf

        if state.is_terminal():
            # Terminal state: degenerate EPF
            terminal_point = EPFPoint(
                follower_payoff=float(state.follower_total_fruit),
                leader_payoff=float(state.leader_total_wood)
            )
            epf = EPF([terminal_point])
            self.epf_cache[state] = epf

            if self.debug and depth <= 3:
                print(f"Terminal EPF at depth {depth}: {epf}")

            return epf

        # Get valid actions
        if state.is_leader_turn():
            valid_actions = self.mdp.get_valid_actions(state.leader_pos)
            player_str = "Leader"
        else:
            valid_actions = self.mdp.get_valid_actions(state.follower_pos)
            player_str = "Follower"

        # Compute child EPFs
        child_epfs = []
        for action in valid_actions:
            if self.mdp.is_valid_action(state, action):
                next_state = self.mdp.transition(state, action)
                child_epf = self._compute_epf(next_state, depth + 1)
                child_epfs.append((action, child_epf))

        if not child_epfs:
            # No valid children - create fallback EPF
            fallback_point = EPFPoint(
                follower_payoff=float(state.follower_total_fruit),
                leader_payoff=float(state.leader_total_wood)
            )
            epf = EPF([fallback_point])
            self.epf_cache[state] = epf
            return epf

        if state.is_leader_turn():
            # Leader node: take upper concave envelope (can mix freely)
            epfs = [child_epf for _, child_epf in child_epfs]
            epf = upper_concave_envelope_multiple(epfs)

            if self.debug and depth <= 2:
                print(f"Leader EPF at depth {depth}: {epf}")

        else:
            # Follower node: apply left-truncation for incentive compatibility
            truncated_epfs = []

            for action, child_epf in child_epfs:
                # Get minimum required incentive (grim threat)
                next_state = self.mdp.transition(state, action)
                threshold = self.grim_values.get(next_state, 0.0)

                # Apply left-truncation
                truncated_epf = child_epf.left_truncate(threshold)
                if truncated_epf.points:
                    truncated_epfs.append(truncated_epf)

            # Take upper concave envelope of incentive-compatible EPFs
            if truncated_epfs:
                epf = upper_concave_envelope_multiple(truncated_epfs)
            else:
                # Fallback
                fallback_point = EPFPoint(
                    follower_payoff=float(state.follower_total_fruit),
                    leader_payoff=float(state.leader_total_wood)
                )
                epf = EPF([fallback_point])

            if self.debug and depth <= 2:
                print(f"Follower EPF at depth {depth}: {epf}")

        self.epf_cache[state] = epf
        return epf

    def visualize_epf(self, solution: Dict, title: str = "EPF Visualization"):
        """Visualize the computed EPF"""
        root_epf = solution['root_epf']

        if not root_epf.points:
            print("No points to visualize")
            return None

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: EPF curve
        points = root_epf.points
        follower_payoffs = [p.follower_payoff for p in points]
        leader_payoffs = [p.leader_payoff for p in points]

        # Plot EPF as piecewise linear function
        ax1.plot(follower_payoffs, leader_payoffs, 'b-o', linewidth=2, markersize=8, label='EPF')

        # Highlight optimal point
        optimal = solution['optimal_point']
        ax1.scatter(optimal.follower_payoff, optimal.leader_payoff,
                   c='red', s=200, marker='*', label='Optimal for Leader', zorder=5)

        ax1.set_xlabel('Follower Payoff (Fruit)')
        ax1.set_ylabel('Leader Payoff (Wood)')
        ax1.set_title(f'{title}\nEPF with {len(points)} knots')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Annotate points
        for i, point in enumerate(points):
            ax1.annotate(f'{i}', (point.follower_payoff, point.leader_payoff),
                        xytext=(5, 5), textcoords='offset points')

        # Plot 2: EPF statistics
        stats = {
            'EPF Knots': len(points),
            'Max Leader Payoff': solution['max_leader_payoff'],
            'Domain Width': solution['domain'][1] - solution['domain'][0],
            'Cached EPFs': len(solution['epf_cache'])
        }

        ax2.bar(stats.keys(), stats.values())
        ax2.set_title('EPF Statistics')
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        return fig

# Simple test MDP for verification
class SimpleTestMDP:
    """Simple test MDP to verify EPF computation"""

    def __init__(self):
        self.width = 2
        self.height = 2

        # Simple forest: wood at (0,0), fruit at (1,1)
        self.forest_map = np.zeros((2, 2, 2))
        self.forest_map[0, 0, 0] = 10  # Wood at leader start
        self.forest_map[1, 1, 1] = 10  # Fruit at follower start
        self.forest_map[1, 0, 0] = 5   # Some wood
        self.forest_map[1, 0, 1] = 5   # Some fruit
        self.forest_map[0, 1, 0] = 3   # Less wood
        self.forest_map[0, 1, 1] = 8   # More fruit

    def get_initial_state(self) -> GameState:
        return GameState(
            leader_pos=(0, 0),
            follower_pos=(1, 1),
            leader_steps_left=1,
            follower_steps_left=1,
            turn=True  # Leader starts
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
        return self.forest_map[x, y, 0], self.forest_map[x, y, 1]

    def transition(self, state: GameState, action: Action) -> GameState:
        if state.is_terminal():
            return state

        if state.is_leader_turn():
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
            new_pos = self.apply_action(state.follower_pos, action)
            _, fruit_reward = self.get_cell_rewards(new_pos)

            return GameState(
                leader_pos=state.leader_pos,
                follower_pos=new_pos,
                leader_steps_left=state.leader_steps_left,
                follower_steps_left=state.follower_steps_left - 1,
                leader_total_wood=state.leader_total_wood,
                follower_total_fruit=state.follower_total_fruit + fruit_reward,
                turn=None  # Game ends
            )

def create_mixed_strategy_beneficial_case():
    """Create a scenario where mixed strategies clearly help"""
    print("\n" + "="*60)
    print("CREATING SCENARIO WHERE MIXED STRATEGIES HELP")
    print("="*60)

    class MixedBenefitMDP(SimpleTestMDP):
        def __init__(self):
            super().__init__()

            # Redesign to create clear mixed strategy benefits
            self.forest_map = np.zeros((2, 2, 2))

            # Strategic layout where leader benefits from unpredictability
            self.forest_map[0, 0, 0] = 2   # Low wood at start
            self.forest_map[0, 0, 1] = 1   # Low fruit at start

            self.forest_map[1, 0, 0] = 8   # High wood RIGHT
            self.forest_map[1, 0, 1] = 9   # High fruit RIGHT (contested!)

            self.forest_map[0, 1, 0] = 7   # Good wood UP
            self.forest_map[0, 1, 1] = 2   # Low fruit UP

            self.forest_map[1, 1, 0] = 1   # Low wood at follower start
            self.forest_map[1, 1, 1] = 3   # Medium fruit at follower start

    print("Testing with modified forest designed for mixed strategy benefits...")

    game = MixedBenefitMDP()


    print("Modified forest layout:")
    for y in range(1, -1, -1):
        for x in range(2):
            wood, fruit = game.get_cell_rewards((x, y))
            print(f"({x},{y}): W={wood:2}, F={fruit:2}  ", end="")
        print()
    print()

    # Analyze this cas
    solver = CorrectEPFMixedSolver(game)
    solution = solver.solve()
    fig = solver.visualize_epf(solution, "Correct EPF-Based Mixed Strategy")
    plt.show()
    return solution


def run_comprehensive_test():
    """Run comprehensive test of the EPF mixed strategy solver"""
    print("COMPREHENSIVE EPF MIXED STRATEGY SOLVER TEST")
    print("=" * 70)

    # Test 1: Basic EPF solver
    # print("TEST 1: Basic EPF Solver")
    # solution1, solver1 = test_correct_epf_solver()

    # Test 2: Compare with pure strategies
    # print("\nTEST 2: Comparison with Pure Strategy Analysis")
    # solution2, stackelberg_outcomes = compare_with_pure_strategy_analysis()

    # Test 3: Modified scenario
    print("\nTEST 3: Modified Scenario for Mixed Strategy Benefits")
    solution3 = create_mixed_strategy_beneficial_case()

    print("\n" + "="*70)
    print("COMPREHENSIVE TEST SUMMARY")
    print("="*70)

    return solution3

if __name__ == "__main__":
    solution3 = run_comprehensive_test()