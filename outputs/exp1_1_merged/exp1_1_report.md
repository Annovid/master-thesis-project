# Эксперимент 1.1 — распределение вклада по температуре

Эндоумент: 20.0. Метрики из дизайна (`docs/06/3. Дизайн эксперимента.md`).

## Сводная таблица

| model | h | T | n | mean | median | std | p(0) | p(E) | entropy | mode | errs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anthropic/claude-4.7-opus-20260416 | 10 | 0.3 | 10 | 15.00 | 15.00 | 0.00 | 0.00 | 0.00 | -0.00 | 15.0 | 0 |
| anthropic/claude-4.7-opus-20260416 | 10 | 0.7 | 10 | 15.00 | 15.00 | 0.00 | 0.00 | 0.00 | -0.00 | 15.0 | 0 |
| anthropic/claude-4.7-opus-20260416 | 10 | 1.0 | 10 | 15.00 | 15.00 | 0.00 | 0.00 | 0.00 | -0.00 | 15.0 | 0 |
| google/gemini-2.5-pro | 10 | 0.6 | 10 | 20.00 | 20.00 | 0.00 | 0.00 | 1.00 | -0.00 | 20.0 | 0 |
| google/gemini-2.5-pro | 10 | 1.4 | 10 | 19.00 | 20.00 | 3.00 | 0.00 | 0.90 | 0.33 | 20.0 | 0 |
| google/gemini-2.5-pro | 10 | 2.0 | 2 | — | — | — | — | — | — | — | 2 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 0.0 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 0.3 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 0.6 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 0.7 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 1.0 | 10 | 10.50 | 10.00 | 1.50 | 0.00 | 0.00 | 0.33 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 1.4 | 10 | 10.50 | 10.00 | 1.50 | 0.00 | 0.00 | 0.33 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 2.0 | 10 | 10.20 | 10.00 | 0.60 | 0.00 | 0.00 | 0.33 | 10.0 | 0 |
| openai/gpt-4o | 10 | 0.0 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| openai/gpt-4o | 10 | 0.3 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| openai/gpt-4o | 10 | 0.6 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| openai/gpt-4o | 10 | 0.7 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| openai/gpt-4o | 10 | 1.0 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| openai/gpt-4o | 10 | 1.4 | 10 | 9.60 | 10.00 | 0.80 | 0.00 | 0.00 | 0.50 | 10.0 | 0 |
| openai/gpt-4o | 10 | 2.0 | 6 | — | — | — | — | — | — | — | 6 |
| openai/gpt-5.5-pro-20260423 | 10 | 0.6 | 10 | 10.00 | 10.00 | 10.00 | 0.50 | 0.50 | 0.69 | 20.0 | 0 |
| openai/gpt-5.5-pro-20260423 | 10 | 1.4 | 10 | 10.00 | 10.00 | 10.00 | 0.50 | 0.50 | 0.69 | 20.0 | 0 |
| openai/gpt-5.5-pro-20260423 | 10 | 2.0 | 2 | — | — | — | — | — | — | — | 2 |

Колонки: `p(0)` — доля вкладов 0; `p(E)` — доля вкладов = endowment; `entropy` — Shannon entropy 10-биновой гистограммы (натов); `mode` — наиболее частое значение.

## Графики

- `plots/hist_anthropic_claude-4.7-opus-20260416_h10.png`
- `plots/summary_anthropic_claude-4.7-opus-20260416_h10.png`
- `plots/hist_google_gemini-2.5-pro_h10.png`
- `plots/summary_google_gemini-2.5-pro_h10.png`
- `plots/hist_meta-llama_llama-4-maverick-17b-128e-instruct_h10.png`
- `plots/summary_meta-llama_llama-4-maverick-17b-128e-instruct_h10.png`
- `plots/hist_openai_gpt-4o_h10.png`
- `plots/summary_openai_gpt-4o_h10.png`
- `plots/hist_openai_gpt-5.5-pro-20260423_h10.png`
- `plots/summary_openai_gpt-5.5-pro-20260423_h10.png`
