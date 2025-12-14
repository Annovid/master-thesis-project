import json
import logging
from pathlib import Path
from typing import Optional

from ..simulation.runner import SimulationResults

logger = logging.getLogger(__name__)

def write_report(results: SimulationResults, output_file: Optional[str] = None) -> None:
    """Write the simulation results to console or file."""
    if output_file:
        path = Path(output_file)
        with path.open("w") as f:
            json.dump({
                "agent_names": results.agent_names,
                "total_payoffs": results.total_payoffs,
                "round_payoffs": results.round_payoffs,
                "history": [r.actions for r in results.history] if results.history else []
            }, f, indent=2)
        logger.info(f"Results written to {path}")
    else:
        logger.info("Simulation Results:")
        logger.info(f"Agents: {results.agent_names}")
        logger.info(f"Total Payoffs: {results.total_payoffs}")
        logger.info("Round Payoffs:")
        for i, payoffs in enumerate(results.round_payoffs, 1):
            logger.info(f"Round {i}: {payoffs}")
        if results.history:
            logger.info("History:")
            for i, round_actions in enumerate(results.history, 1):
                logger.info(f"Round {i}: {round_actions.actions}")