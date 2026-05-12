# Эксперимент 1.1 — распределение вклада по температуре

**Цель.** Измерить, как форма распределения вклада в one-shot PGG зависит от
температуры сэмплинга.

**Дизайн.** PGG (4 игрока, endowment 20, multiplier 1.6), one-shot, горизонт в
промпте = 10 раундов. Варьируем температуру; n = 10 на ячейку (модель × T).
Метрики: гистограмма по 10 бинам, mean / median / std, доля «угловых» (0 и 20),
энтропия распределения.

Примечание про шкалу T: Anthropic ограничен `[0, 1]`, OpenAI/Google допускают
`[0, 2]`. Полный грид в реальности собирался по каждой модели отдельно.

## Прогоны и где лежит raw data

Эксп. 1.1 фактически собирался **итеративно**, по одному probe-прогону на модель,
а потом мерджился в общий датасет.

### `outputs/exp1_1/` — первый запуск (только GPT-5.5-pro по подгриду T)

- Модель: `openai/gpt-5.5-pro` (provider `openrouter`,
  `OPENROUTER_MAX_TOKENS = 4000`).
- Грид: `horizons = [10]`, `T = [0.6, 1.4]`, n = 10.
- Артефакты: `run_config.json`, `run_summary.json`, `<model>/horizon_10/temp_<T>/sample_<i>.txt`.
- Лог: `outputs/exp1_1_subgrid.log`.

### `outputs/exp1_1_llama_probe/` — Llama 4 Maverick по полному T-гриду

- Модель: `meta-llama/llama-4-maverick` (provider `openrouter`,
  `max_tokens = 4000`).
- Грид: `T = [0.0, 0.3, 0.6, 0.7, 1.0, 1.4, 2.0]`, h = 10, n = 10.
- Артефакты: `run_config.json`, `run_summary.json`, `exp1_1_analysis.json`,
  `exp1_1_report.md`, `meta-llama_llama-4-maverick/horizon_10/temp_<T>/sample_<i>.txt`,
  `plots/`.
- Лог: `outputs/exp1_1_llama_probe.log`.

### `outputs/exp1_1_gpt4o_probe/` — GPT-4o через OpenRouter

- Модель: `openai/gpt-4o` (provider `openrouter`, `max_tokens = 4000`).
- Грид: `T = [0.6, 1.4, 2.0]`, h = 10, n = 10.
- Артефакты: `run_config.json`, `run_summary.json`, `exp1_1_analysis.json`,
  `exp1_1_report.md`, `openai_gpt-4o/...`, `plots/`.
- Лог: `outputs/exp1_1_gpt4o_probe.log`.

### `outputs/exp1_1_gpt4o_openai/` — GPT-4o через прямой OpenAI API

Параллельный прогон GPT-4o напрямую через OpenAI, чтобы добить недостающие T и
сравнить с поведением через OpenRouter.

- Модель: `gpt-4o` (provider `openai`, без `max_tokens`).
- Грид: `T = [0.0, 0.3, 0.6, 0.7, 1.0, 1.4, 2.0]`, h = 10, n = 10.
- Артефакты: `run_config.json`, `run_summary.json`, `exp1_1_analysis.json`,
  `exp1_1_report.md`, `gpt-4o/horizon_10/temp_<T>/sample_<i>.txt`, `plots/`.
- Лог: `outputs/exp1_1_gpt4o_openai.log` (+ пустой `exp1_1_gpt4o_t2.log`).

### `outputs/exp1_1_bare_number_probe/` — sanity-чек «голого числа»

Два маленьких параллельных прогона GPT-4o по верхним T (`1.2, 1.4, 1.6, 1.8, 2.0`),
n = 10, h = 10 — чтобы посмотреть на устойчивость при высоких T. Делалось
дважды: через прямой OpenAI и через OpenRouter.

- `openai_direct/` — `provider = openai`, `gpt-4o`. Подпапки: `run_config.json`,
  `run_summary.json`, `gpt-4o/horizon_10/...`.
- `openrouter/` — `provider = openrouter`, `openai/gpt-4o`. Подпапки:
  `run_config.json`, `run_summary.json`, `openai_gpt-4o/horizon_10/...`.

### `outputs/exp1_1_remaining.log` — лог дотопа Opus / Gemini / GPT-5.5-pro

Содержит догонные прогоны для Эксп. 1.1 по моделям Opus 4.7, Gemini 2.5 Pro и
GPT-5.5-pro по нужным T. Соответствующие sample-файлы пишутся в подпапки моделей
внутри `outputs/exp1_1/`.

### `outputs/exp1_1_merged/` — финальная сводка Эксп. 1.1

Сводный датасет и отчёт по всем моделям, использовавшим OpenRouter
(Opus 4.7, Gemini 2.5 Pro, Llama 4 Maverick, GPT-4o, GPT-5.5-pro).

- `run_summary.json` — мердж всех runs.
- `exp1_1_analysis.json` — метрики по ячейкам.
- `exp1_1_report.md` — авто-репорт со сводной таблицей и списком графиков.
- `plots/` — гистограммы (`hist_<model>_h10.png`) и summary-картинки
  (`summary_<model>_h10.png`) по 5 моделям.

## Ключевые результаты (по `outputs/exp1_1_merged/exp1_1_analysis.json`)

Существенные ячейки сведены в `exp1_1_merged/exp1_1_report.md`. Содержательно:

- **Opus 4.7** (T = 0.3 / 0.7 / 1.0): mode = 15, mean = 15, std = 0 — стенка по T.
- **Gemini 2.5 Pro** (T = 0.6 / 1.4): mode = 20, mean ≈ 19–20, p(E) = 0.9–1.0;
  при T = 2.0 — парсер падает (n = 2, оба errs).
- **GPT-4o через OpenRouter**: mode = 10 во всём диапазоне `[0.0..1.0]`, при
  T = 1.4 std = 0.8 (mean 9.6), при T = 2.0 — 6/6 errs.
- **Llama 4 Maverick**: mode = 10 на всём диапазоне `[0.0..2.0]`, разброс
  появляется при T ≥ 1.0 (std = 1.5 при T = 1.0 / 1.4; std = 0.6 при T = 2.0).
- **GPT-5.5-pro** (T = 0.6 / 1.4): биполярное распределение между 0 и 20
  (p(0) = p(E) = 0.5, std = 10, entropy ≈ 0.69), что согласуется с
  ненулевой недетерминированностью при T = 0 из Эксп. 0.
