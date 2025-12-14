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
