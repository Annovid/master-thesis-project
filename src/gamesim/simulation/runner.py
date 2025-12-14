import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

from ..game.state import GameState, RoundActions
from ..games.base_game import Game
from ..agents.base import Agent

@dataclass
class SimulationResults:
    """Results of a simulation run."""
    history: List[RoundActions]
    total_payoffs: List[float]
    round_payoffs: List[List[float]]
    agent_names: List[str]
    config: dict

def run_simulation(game: Game, agents: List[Agent], n_rounds: int, config: dict) -> SimulationResults:
    """Run the simulation for the given number of rounds."""
    # Create artifacts directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    game_name = game.__class__.__name__.lower()
    artifacts_dir = Path("artifacts") / "games" / f"{game_name}-{timestamp}"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    state = GameState(n_rounds)
    total_payoffs = [0.0] * game.num_players
    round_payoffs: List[List[float]] = []
    round_details = []

    round_num = 0
    while not state.is_game_over():
        round_num += 1
        actions_and_details = [agent.act(state, game, i) for i, agent in enumerate(agents)]
        actions = [ad[0] for ad in actions_and_details]
        details = [ad[1] for ad in actions_and_details]
        
        payoffs = game.compute_payoffs(actions)
        state.add_round(actions)
        
        for i in range(game.num_players):
            total_payoffs[i] += payoffs[i]
        round_payoffs.append(payoffs)

        # Save round details
        round_detail = {
            "round": round_num,
            "actions": actions,
            "payoffs": payoffs,
            "details": details
        }
        round_details.append(round_detail)
        
        # Save to file
        with open(artifacts_dir / f"round_{round_num}.json", "w") as f:
            json.dump(round_detail, f, indent=2)

    # Save final report
    final_report = {
        "config": config,
        "game": game_name,
        "agent_names": [agent.name for agent in agents],
        "total_payoffs": total_payoffs,
        "round_payoffs": round_payoffs,
        "history": [r.actions for r in state.get_history()],
        "round_details": round_details
    }
    with open(artifacts_dir / "final_report.json", "w") as f:
        json.dump(final_report, f, indent=2)

    return SimulationResults(
        history=state.get_history(),
        total_payoffs=total_payoffs,
        round_payoffs=round_payoffs,
        agent_names=[agent.name for agent in agents],
        config=config
    )