# Эксперимент 0 — детерминированность при T = 0

**Цель.** Измерить, насколько ответ LLM воспроизводим при `T = 0` в PGG: числовая
стабильность распарсенного вклада (а) и текстовая стабильность полного ответа
(б, SHA-256 по байтам).

**Дизайн.** PGG (4 игрока, endowment 20, multiplier 1.6), one-shot. В промпте
варьируется заявленный горизонт: `1 round` vs `10 rounds`. n = 10 повторов на
ячейку. Все запросы через OpenRouter, `temperature = 0.0`. `max_tokens`: 4000 для
Opus / GPT-5.5-pro / Llama; 16000 для Gemini (иначе CoT обрезается).

Подробный отчёт с интерпретацией: `docs/06/4. Результаты эксперимента 0.md`
(в репозитории тезиса).

## Прогоны и где лежит raw data

### `outputs/exp0_smoke/` — smoke-тест

Цель: проверить пайплайн на дешёвой модели до полного прогона.

- Модель: `gpt-4o-mini` (provider `openai`, прямой OpenAI API).
- Грид: `horizons = [1, 10]`, `T = 0.0`, n = 10 → 20 запросов.
- Артефакты:
  - `run_config.json` — конфиг.
  - `run_summary.json` — runs.
  - `analysis.json` — метрики (h=1: mode=10, n_unique_actions=1; h=10: mode=10, n_unique_actions=1).
  - `gpt-4o-mini/horizon_<1|10>/temp_0.0/sample_<0..9>.txt` — полные ответы.

### `outputs/exp0_probe/` — кросс-провайдерный probe

Цель: убедиться, что 4 целевых модели вообще отвечают через OpenRouter и парсятся.

- Модели: `anthropic/claude-opus-4.7`, `openai/gpt-5.5-pro`,
  `google/gemini-2.5-pro`, `meta-llama/llama-4-maverick` (provider `openrouter`).
- Грид: `horizons = [1]`, `T = 0.0`, n = 1 → 4 запроса.
- Артефакты: `run_config.json`, `run_summary.json`, `<model_safe>/horizon_1/temp_0.0/sample_0.txt`.

### `outputs/exp0_probe2/` — точечная перепроверка Gemini

- Модель: `google/gemini-2.5-pro`, `horizons = [1]`, `T = 0.0`, n = 1.
- Артефакты: те же ключи.

### `outputs/exp0_iter1/` — основной грид Эксп. 0

Цель: полный n = 10 по 4 моделям × 2 горизонтам = 80 запросов.

- Модели: `anthropic/claude-opus-4.7`, `openai/gpt-5.5-pro`,
  `google/gemini-2.5-pro`, `meta-llama/llama-4-maverick`.
- Грид: `horizons = [1, 10]`, `T = 0.0`, n = 10.
- Артефакты:
  - `run_config.json`, `run_summary.json` (финальный с мерджем всех догонов).
  - `analysis.json` — метрики по ячейкам.
  - Промежуточные снапшоты (можно удалить): `run_summary_part1.json`,
    `run_summary_part2.json`, `run_summary_pre_topup.json`.
  - `<model_safe>/horizon_<1|10>/prompt.txt` + `temp_0.0/sample_<0..9>.txt` для
    каждой из 4 моделей.
- Логи фоновых прогонов:
  - `outputs/exp0_iter1.log` (пуст — поток шёл в `_part2`),
  - `outputs/exp0_iter1_part2.log`,
  - `outputs/exp0_iter1_retry.log`,
  - `outputs/exp0_iter1_topup.log`,
  - `outputs/exp0_iter1_topup_gpt.log`.

## Ключевые результаты (по `outputs/exp0_iter1/analysis.json`)

| Модель | h | uniq_a | uniq_sha | mode | mean | std | errs |
|---|---:|---:|---:|---:|---:|---:|---:|
| anthropic/claude-opus-4.7 | 1 | 1 | 9 | 0 | 0.00 | 0.00 | 0 |
| anthropic/claude-opus-4.7 | 10 | 1 | 10 | 15 | 15.00 | 0.00 | 0 |
| openai/gpt-5.5-pro | 1 | 1 | 9 | 0 | 0.00 | 0.00 | 0 |
| openai/gpt-5.5-pro | 10 | 2 | 5 | 20 | 13.33 | 9.43 | 1 |
| google/gemini-2.5-pro | 1 | 1 | 4 | 0 | 0.00 | 0.00 | 0 |
| google/gemini-2.5-pro | 10 | 2 | 5 | 20 | 18.00 | 4.00 | 0 |
| meta-llama/llama-4-maverick | 1 | 1 | 10 | 0 | 0.00 | 0.00 | 0 |
| meta-llama/llama-4-maverick | 10 | 1 | 10 | 10 | 10.00 | 0.00 | 0 |

Содержательно:

- В one-shot (h=1) **все 4 модели вкладывают 0** — равновесие по Нэшу.
- При замене промпта на `10 rounds` shadow of the future включается сразу:
  Llama → 10, Opus → 15, Gemini → 20 (mean 18), GPT-5.5-pro → 20 (mean 13.3).
- Числовая детерминированность **нарушается** у Gemini h=10 (`20 × 8, 10 × 2`)
  и GPT-5.5-pro h=10 (`20 × 6, 0 × 3` + 1 пустой ответ от API).
- Текстовая детерминированность практически нулевая везде.
