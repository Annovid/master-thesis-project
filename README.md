# LLM Public Goods Game Experiments

Репозиторий содержит код и результаты экспериментов для дипломной работы о
поведении LLM-агентов в экономических играх. Основной фокус сейчас — Public
Goods Game (PGG): вклад в общественный фонд, устойчивость числовых решений,
влияние температуры, многораундовая динамика и чувствительность к формулировке
prompt.

Проект устроен как два связанных слоя:

- `src/gamesim/` — общий симулятор экономических игр и агентов;
- `src/services/` и `scripts/` — воспроизводимые экспериментальные запуски,
  анализ и экспорт результатов для диплома.

## Структура проекта

```text
.
├── README.md
├── main.py
├── pyproject.toml
├── configs/
│   ├── pd_config.yaml
│   └── pg_config.yaml
├── src/
│   ├── connectors/
│   ├── gamesim/
│   └── services/
├── scripts/
├── docs/
│   └── experiments/
├── outputs/
├── artifacts/
└── resources/
```

## Основные папки

`configs/` — YAML-конфиги для обычного запуска симулятора через `main.py`.
Сейчас есть примеры для Prisoner's Dilemma и Public Goods Game.

`src/connectors/` — адаптеры к LLM-провайдерам:

- `openai_connector.py` — прямой OpenAI API;
- `openrouter_connector.py` — OpenRouter;
- `gateway_connector.py` — внутренний gateway-совместимый API;
- `mock_connector.py` — тестовый connector без внешнего API.

`src/gamesim/` — ядро симулятора:

- `agents/` — базовый агент, LLM-агент и простые стратегии;
- `games/` — реализации игр (`public_goods_game.py`, `pd_game.py`);
- `game/` — состояние игры, правила и вспомогательные структуры;
- `simulation/runner.py` — общий цикл симуляции;
- `reporting/` — запись отчётов;
- `config.py` — загрузка YAML-конфигов.

`src/services/` — экспериментальные сервисы, которые используются для диплома:

- `experiment0.py` — эксперимент 0: детерминированность one-shot PGG при `T=0`;
- `analyze_exp0.py` — анализ результатов эксперимента 0;
- `analyze_exp1_1.py` — анализ распределений вкладов по температуре;
- `experiment1_2.py` — эксперимент 1.2: repeated PGG и decay of cooperation;
- `analyze_exp1_2.py` — анализ многораундовых сессий;
- `experiment2_1.py` — эксперимент 2.1: sweep prompt-условий в one-shot PGG;
- `analyze_exp2_1.py` — анализ prompt-условий;
- `single_request.py` — общий helper для единичных LLM-запросов;
- `cost_estimate.py` — оценка стоимости прогонов.

`scripts/` — небольшие CLI-утилиты поверх сервисов:

- `exp1_1_runner.py`, `exp1_1_analyze.py` — запуск и анализ температурного грида;
- `collect_exp1_2.py` — сбор результатов эксперимента 1.2 в таблицы;
- `sessions_to_md.py`, `sessions_to_excel.py` — экспорт сессий в Markdown/XLSX;
- `exp_status.py`, `exp2_1_status.py` — проверка статуса прогонов.

`docs/experiments/` — человекочитаемые описания экспериментов и индексы
результатов. Эти документы связывают код, raw outputs и текст диплома.

`outputs/` — результаты запусков. Обычно внутри каждого `outputs/<run_id>/`
лежат:

- `run_config.json` — параметры запуска;
- `run_summary.json` — все вызовы, распарсенные действия, ошибки и метаданные;
- `<model_safe>/.../prompt.txt` — точный prompt;
- `<model_safe>/.../sample_<i>.txt` или `player_<p>.txt` — полный ответ модели;
- `analysis.json`, `exp*_analysis.json` или `exp*_report.md` — агрегированный анализ;
- `plots/` — графики, если они были построены.

`artifacts/` — рабочие артефакты, временные логи, сохранённые ответы и ноутбуки.
Это вспомогательная зона, а не основной источник результатов для диплома.

`resources/` — дополнительные prompt-шаблоны, email-черновики и одиночные
материалы, которые не являются кодом экспериментов.

## Установка

Требуется Python 3.8+.

```bash
poetry install
```

Или в editable-режиме:

```bash
pip install -e .
```

Для реальных LLM-запросов нужен `.env` в корне проекта:

```text
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
LLM_GATEWAY_API_KEY=...
LLM_GATEWAY_BASE_URL=...
LLM_PROVIDER=auto
```

`LLM_PROVIDER` может быть `auto`, `openai`, `openrouter` или `gateway`.
Если provider не задан явно, `gpt-*` модели идут через OpenAI, остальные — через
OpenRouter.

## Быстрый запуск симулятора

Запуск по готовому конфигу:

```bash
python main.py --config configs/pg_config.yaml
```

Короткая форма:

```bash
python main.py --game pg
python main.py --game pd
```

Анализ JSON-результата обычной симуляции:

```bash
python main.py --analyze path/to/results.json
```

## Эксперименты

Экспериментальные сервисы запускаются как Python-модули. Конкретные параметры
задаются CLI-аргументами самих файлов.

Примеры:

```bash
python -m src.services.experiment0 --help
python -m src.services.experiment1_2 --help
python -m src.services.experiment2_1 --help
```

Текущая карта экспериментов:

- `exp0` — one-shot PGG: насколько стабилен числовой ответ при `T=0` и разных
  горизонтах в prompt;
- `exp1_1` — one-shot PGG: распределение вкладов при разных температурах;
- `exp1_2` — repeated PGG: динамика вкладов по раундам и decay of cooperation;
- `exp2_1` — one-shot PGG: влияние prompt-условий на близость к человеческому
  baseline.

## Связь с дипломом

Этот репозиторий хранит исполняемый код и raw data. Текст диплома, LaTeX и
обобщающие литературные материалы находятся уровнем выше в основном workspace
диплома. При переносе результатов в диплом обычно используются:

- агрегированные таблицы из `outputs/*/run_summary.json`;
- Markdown-экспорты из `outputs/exp1_2_md/`;
- отчёты из `docs/experiments/`;
- графики из `outputs/*/plots/`.

## Что не является частью архитектуры

`venv/`, `__pycache__/`, `.git/`, `.claude/`, `.codex/` и другие служебные
директории могут присутствовать в рабочем дереве, но не являются частью
структуры проекта и не нужны для понимания кода.
