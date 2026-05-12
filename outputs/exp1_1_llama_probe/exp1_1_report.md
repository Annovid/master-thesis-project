# Эксперимент 1.1 — распределение вклада по температуре

Эндоумент: 20.0. Метрики из дизайна (`docs/06/3. Дизайн эксперимента.md`).

## Сводная таблица

| model | h | T | n | mean | median | std | p(0) | p(E) | entropy | mode | errs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 0.0 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 0.3 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 0.6 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 0.7 | 10 | 10.00 | 10.00 | 0.00 | 0.00 | 0.00 | -0.00 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 1.0 | 10 | 10.50 | 10.00 | 1.50 | 0.00 | 0.00 | 0.33 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 1.4 | 10 | 10.50 | 10.00 | 1.50 | 0.00 | 0.00 | 0.33 | 10.0 | 0 |
| meta-llama/llama-4-maverick-17b-128e-instruct | 10 | 2.0 | 10 | 10.20 | 10.00 | 0.60 | 0.00 | 0.00 | 0.33 | 10.0 | 0 |

Колонки: `p(0)` — доля вкладов 0; `p(E)` — доля вкладов = endowment; `entropy` — Shannon entropy 10-биновой гистограммы (натов); `mode` — наиболее частое значение.

## Графики

- `plots/hist_meta-llama_llama-4-maverick-17b-128e-instruct_h10.png`
- `plots/summary_meta-llama_llama-4-maverick-17b-128e-instruct_h10.png`
