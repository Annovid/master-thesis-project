import logging
import argparse

from dotenv import load_dotenv

from src.gamesim.config import load_config
from src.gamesim.simulation.runner import run_simulation
from src.gamesim.reporting.report_writer import write_report
from src.services.analyze.analyzer import analyze
from src.gamesim.agents.simple_agents import AlwaysCooperate, AlwaysDefect, RandomAgent
from src.gamesim.agents.llm_agent import LLMAgent
from src.connectors.openai_connector import OpenAIConnector
from src.connectors.openrouter_connector import OpenRouterConnector
from src.connectors.gateway_connector import LLMApiGatewayConnector
from src.connectors.mock_connector import MockConnector
from src.connectors.model_aliases import resolve_model_alias

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

def create_agent(agent_config: dict, name: str, default_provider: str = "auto"):
    """Create an agent based on config."""
    agent_type = agent_config.get("type")
    if agent_type is None:
        raise ValueError("Missing 'type' in agent config")
    if agent_type == "simple":
        strategy = agent_config.get("strategy")
        if strategy is None:
            raise ValueError("Missing 'strategy' for simple agent")
        if strategy == "always_cooperate":
            return AlwaysCooperate(name)
        elif strategy == "always_defect":
            return AlwaysDefect(name)
        elif strategy == "random":
            return RandomAgent(name)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    elif agent_type == "llm":
        model = agent_config.get("model")
        if model is None:
            raise ValueError("Missing 'model' for llm agent")
        model = resolve_model_alias(model)
        temperature = agent_config.get("temperature")
        if temperature is None:
            raise ValueError("Missing 'temperature' for llm agent")
        reasoning = agent_config.get("reasoning", False)
        provider = agent_config.get("provider", default_provider)

        # provider overrides routing when explicitly set
        if provider == "openai":
            connector = OpenAIConnector(model=model, temperature=temperature)
        elif provider == "openrouter":
            connector = OpenRouterConnector(model=model, temperature=temperature)
        elif provider == "gateway":
            connector = LLMApiGatewayConnector(model=model, temperature=temperature)
        else:  # auto
            if model.startswith("gpt-"):
                connector = OpenAIConnector(model=model, temperature=temperature)
            else:
                connector = OpenRouterConnector(model=model, temperature=temperature)

        return LLMAgent(name, connector, temperature, reasoning)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def main(config_path: str) -> int:
    """Main entry point for the simulation."""
    load_dotenv()
    config = load_config(config_path)

    agents = [
        create_agent(agent_config, f"Player{i+1}", default_provider=config.llm_provider)
        for i, agent_config in enumerate(config.agent_configs)
    ]

    # Create config dict for reporting
    config_dict = {
        "game": config.game.__class__.__name__.lower(),
        "rounds": config.rounds,
        "agents": config.agent_configs
    }
    if hasattr(config.game, 'payoff_matrix'):
        config_dict["payoff_matrix"] = [
            [[p.player1, p.player2] for p in row] for row in config.game.payoff_matrix
        ]
    if hasattr(config.game, 'endowment'):
        config_dict.update({
            "num_players": config.game.num_players,
            "endowment": config.game.endowment,
            "multiplier": config.game.multiplier,
            "transparency": config.transparency
        })

    results = run_simulation(config.game, agents, config.rounds, config_dict)

    write_report(results)
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Game Simulation and Analysis")
    parser.add_argument('--config', type=str, help='Path to YAML config file for simulation')
    parser.add_argument('--game', type=str, choices=['pd', 'pg'], help='Quick run with default config for game type')
    parser.add_argument('--analyze', type=str, help='Path to JSON file for analysis')
    args = parser.parse_args()
    
    if args.analyze:
        analyze(args.analyze)
    elif args.config:
        raise SystemExit(main(args.config))
    elif args.game:
        config_path = f'configs/{args.game}_config.yaml'
        raise SystemExit(main(config_path))
    else:
        print("Use --config or --game for simulation, or --analyze for analysis")
