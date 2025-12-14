from typing import Tuple

from .rules import GameRules, Payoff
from .state import GameState, Action

def step(rules: GameRules, state: GameState, actions: Tuple[Action, Action]) -> Tuple[GameState, Payoff]:
    """Execute a game step with given actions and return updated state and payoff."""
    action1, action2 = actions
    payoff = rules.get_payoff(action1, action2)
    state.add_round(action1, action2)
    return state, payoff