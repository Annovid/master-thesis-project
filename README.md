# Game Simulation

A project for simulating economic games with LLM agents.

## Installation

```bash
pip install -e .
```

## Setup

1. Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   OPENROUTER_API_KEY=your-openrouter-api-key-here
   LLM_GATEWAY_API_KEY=your-gateway-api-key-here
   LLM_GATEWAY_BASE_URL=https://api.llm-gateway.local
   LLM_PROVIDER=auto  # auto|openai|openrouter|gateway (default: auto)
   ```

2. Install dependencies:
   ```bash
   poetry install
   ```

## Usage

Run simulation with config file:

```bash
python main.py --config configs/pd_config.yaml
```

Example configs:
- `configs/pd_config.yaml` - Prisoner's Dilemma
- `configs/pg_config.yaml` - Public Goods Game

Config fields:
- `llm_provider`: default provider for all LLM agents (`auto|openai|openrouter|gateway`)
- Each `agent.provider` overrides the global provider.
