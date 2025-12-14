Хорошо, сделаем **осознанную, “взрослую” архитектуру с `main.py` в корне**, без фанатизма, но так, чтобы проект **масштабировался, тестировался и не превратился в спагетти**.

Контекст я учитываю:

* теория игр (Public Goods / CPR)
* правила в **текстовом формате**
* **LLM-агенты**
* **API запросы**
* парсинг ответов
* **отчёты по играм**

---

# 🎯 Общая идея архитектуры

**`main.py` — orchestration / entrypoint**

* парсит аргументы
* загружает правила
* инициализирует агентов
* запускает симуляции
* сохраняет отчёты

**Вся логика — в пакетах**

* игра ≠ LLM
* LLM ≠ отчёты
* отчёты ≠ формат хранения

---

# 📁 Структура проекта (с `main.py` в корне)

```
game-sim/
├─ main.py
├─ pyproject.toml
├─ README.md
├─ rules/
│  ├─ public_goods.txt
│  └─ cpr.txt
├─ src/
│  └─ gamesim/
│     ├─ __init__.py
│
│     ├─ config.py
│
│     ├─ game/
│     │  ├─ __init__.py
│     │  ├─ rules.py
│     │  ├─ state.py
│     │  ├─ engine.py
│     │  └─ payoff.py
│
│     ├─ agents/
│     │  ├─ __init__.py
│     │  ├─ base.py
│     │  ├─ llm_agent.py
│     │  └─ prompts.py
│
│     ├─ llm/
│     │  ├─ __init__.py
│     │  ├─ client.py
│     │  ├─ openai_client.py
│     │  └─ schemas.py
│
│     ├─ parsing/
│     │  ├─ __init__.py
│     │  └─ action_parser.py
│
│     ├─ simulation/
│     │  ├─ __init__.py
│     │  ├─ runner.py
│     │  └─ episode.py
│
│     ├─ reporting/
│     │  ├─ __init__.py
│     │  ├─ metrics.py
│     │  ├─ tables.py
│     │  └─ report_writer.py
│
│     └─ utils/
│        ├─ __init__.py
│        └─ logging.py
│
└─ tests/
   ├─ test_game_engine.py
   ├─ test_parser.py
   └─ test_simulation.py
```

---

# ▶️ `main.py` (корень)

**Минимум логики. Только orchestration.**

```python
from gamesim.config import load_config
from gamesim.game.rules import load_rules
from gamesim.simulation.runner import run_simulation
from gamesim.reporting.report_writer import write_report


def main() -> int:
    config = load_config()

    rules = load_rules(config.rules_path)

    results = run_simulation(
        rules=rules,
        model=config.model,
        n_agents=config.n_agents,
        n_rounds=config.n_rounds,
    )

    write_report(results, output_dir=config.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

👉 **`main.py` можно читать как сценарий эксперимента**

---

# 🧠 Слои и ответственность

## 1️⃣ Game (чистая теория игр)

```
game/
├─ rules.py     # парсинг текстовых правил
├─ state.py     # состояние игры (раунд, ресурсы, вклады)
├─ engine.py    # шаг игры
└─ payoff.py    # расчёт выигрышей
```

```python
# engine.py
def step(state, actions):
    ...
    return new_state, payoffs
```

❗ **Не знает ничего про LLM**

---

## 2️⃣ Agents (поведение игроков)

```
agents/
├─ base.py
├─ llm_agent.py
└─ prompts.py
```

```python
class Agent:
    def act(self, state) -> str:
        raise NotImplementedError
```

```python
class LLMAgent(Agent):
    def act(self, state) -> str:
        response = self.client.complete(prompt)
        return response
```

---

## 3️⃣ LLM (инфраструктура)

```
llm/
├─ client.py        # абстракция
├─ openai_client.py # конкретная реализация
└─ schemas.py       # ожидаемый формат ответа
```

Позволяет легко:

* сменить OpenAI → Anthropic → local LLM
* мокать в тестах

---

## 4️⃣ Parsing (критически важный слой)

```
parsing/action_parser.py
```

```python
def parse_action(text: str) -> Action:
    """
    Превращает LLM-ответ в формальное действие:
    - вклад
    - извлечение ресурса
    """
```

👉 **Изолирует весь “LLM хаос”**

---

## 5️⃣ Simulation (оркестрация)

```
simulation/
├─ episode.py
└─ runner.py
```

```python
def run_simulation(...):
    for round in rounds:
        actions = collect_actions()
        state, payoffs = engine.step(state, actions)
```

---

## 6️⃣ Reporting (выход)

```
reporting/
├─ metrics.py
├─ tables.py
└─ report_writer.py
```

Поддержка:

* CSV
* JSON
* Markdown
* LaTeX / PDF (если надо)

---

# 🧪 Тестируемость

* `game/` — чистые unit-тесты
* `parsing/` — property tests
* `llm/` — мок клиента
* `simulation/` — deterministic runs

---

# 🧠 Почему `main.py` здесь оправдан

✔ Исследовательский / экспериментальный проект
✔ Один сценарий запуска
✔ Ясная точка входа
✔ Удобно запускать:

```bash
python main.py --rules rules/cpr.txt --model gpt-4o
```

 В main.py только точка запуска
