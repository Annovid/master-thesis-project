import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from src.connectors.openai_connector import OpenAIConnector
from src.connectors.openrouter_connector import OpenRouterConnector
from src.connectors.gateway_connector import LLMApiGatewayConnector
from src.connectors.mock_connector import MockConnector


def build_connector(agent_cfg: dict, default_provider: str = "auto"):
    model = agent_cfg.get("model", "gpt-3.5-turbo-0125")
    temperature = agent_cfg.get("temperature", 0.0)
    connector_type = agent_cfg.get("type", "openai")
    provider = agent_cfg.get("provider", default_provider)

    if connector_type == "mock":
        return MockConnector()

    # provider overrides routing when explicitly set
    if provider == "openai":
        return OpenAIConnector(model=model, temperature=temperature)
    if provider == "openrouter":
        return OpenRouterConnector(model=model, temperature=temperature)
    if provider == "gateway":
        return LLMApiGatewayConnector(model=model, temperature=temperature)

    # auto: route native OpenAI models -> OpenAI; otherwise -> OpenRouter
    if connector_type == "openai" or model.startswith("gpt-"):
        return OpenAIConnector(model=model, temperature=temperature)

    return OpenRouterConnector(model=model, temperature=temperature)


def run_single(prompt_path: Path, config_path: Path, output_dir: Path) -> None:
    """Run LLM requests for all agents in config against a single prompt (no game)."""

    load_dotenv()

    config = json.loads(config_path.read_text())
    agents = config.get("agents", [])
    llm_provider = config.get("llm_provider") or os.getenv("LLM_PROVIDER") or "auto"
    if not agents:
        raise ValueError("config must contain at least one agent in 'agents'")

    prompt = prompt_path.read_text()

    # Ensure output directory and persist prompt once
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.txt").write_text(prompt)

    agents_data = []

    for idx, agent_cfg in enumerate(agents):
        connector = build_connector(agent_cfg, default_provider=llm_provider)
        response, meta = connector.query(prompt)

        chat_id = meta.get("chat_id") if isinstance(meta, dict) else None
        model = meta.get("model") if isinstance(meta, dict) else agent_cfg.get("model")
        name = agent_cfg.get("name", f"agent_{idx}")

        log_path = output_dir / f"chat_{idx}_{name}.txt"
        with log_path.open("w") as f:
            f.write(f"chat_id: {chat_id}\n")
            f.write(f"model: {model}\n")
            f.write(f"name: {name}\n\n")
            f.write("PROMPT:\n")
            f.write(prompt)
            f.write("\n\nRESPONSE:\n")
            f.write(response)
            f.write("\n")

        agents_data.append(
            {
                "name": name,
                "model": model,
                "chat_id": chat_id,
                "log_path": str(log_path),
                "temperature": agent_cfg.get("temperature"),
            }
        )

        # Single-line status per agent
        print(f"OK {log_path} agent={name} model={model}")

    # Write agents.json into output_dir
    agents_path = output_dir / "agents.json"
    agents_path.write_text(json.dumps({"agents": agents_data}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single raw LLM request tester (no game)")
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("/home/annovid/personal/mipt/master-thesis/resources/single/prompt.txt"),
        help="Path to prompt.txt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/annovid/personal/mipt/master-thesis/resources/single/config.json"),
        help="Path to config.json with agents",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/annovid/personal/mipt/master-thesis/resources/single/last_run"),
        help="Directory to store outputs",
    )
    args = parser.parse_args()

    run_single(args.prompt, args.config, args.output)
