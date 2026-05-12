# Эксперимент 1.2 — decay of cooperation (pilot)

Endowment: 20.0. Полная PGG, 10 раундов, 4 копии модели, T=0.7, transparency=on. Дизайн: `docs/06/3. Дизайн эксперимента.md`.

## Сводная таблица по сессиям

| model | session | rounds | R1 | R_last | R1−R_last | slope | cc_r | mean_std_within_round | incomplete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anthropic/claude-opus-4.7 | 0 | 10 | 15.00 | 10.00 | 5.00 | 0.132 | 0.08 | 1.30 | False |
| anthropic/claude-opus-4.7 | 1 | 10 | 15.25 | 10.00 | 5.25 | 0.027 | 0.10 | 1.09 | False |
| google/gemini-2.5-pro | 0 | 5 | 20.00 | 20.00 | 0.00 | 0.000 | — | 0.00 | True |
| google/gemini-2.5-pro | 1 | 10 | 20.00 | 0.00 | 20.00 | -1.727 | 0.71 | 0.87 | False |
| meta-llama/llama-4-maverick | 0 | 10 | 11.75 | 0.00 | 11.75 | -1.232 | 0.88 | 0.99 | False |
| meta-llama/llama-4-maverick | 1 | 10 | 10.00 | 0.00 | 10.00 | -0.758 | 0.75 | 0.50 | False |
| openai/gpt-4o | 0 | 10 | 10.00 | 18.25 | -8.25 | 0.938 | 0.96 | 0.48 | False |
| openai/gpt-4o | 1 | 5 | 8.75 | 9.75 | -1.00 | 0.325 | -0.08 | 1.00 | True |

Колонки: `R1` и `R_last` — средний по 4 игрокам вклад в первом и последнем сыгранном раунде; `R1−R_last` — простая разность (положительная = есть decay); `slope` — OLS-наклон `mean_contribution(round) ~ round` по всем сыгранным раундам сессии (отрицательный = decay); `cc_r` — корреляция Пирсона между вкладом игрока в раунде r+1 и средним вкладом остальных в раунде r (>0 = conditional cooperation); `mean_std_within_round` — средняя дисперсия вкладов между 4 игроками внутри одного раунда, усреднённая по раундам (индикатор гетерогенности).

## Сводка по моделям

| model | n_sessions | n_incomplete | mean_slope | slope_sign_consistent | mean R1−R_last | mean cc_r |
|---|---:|---:|---:|---:|---:|---:|
| anthropic/claude-opus-4.7 | 2 | 0 | 0.080 | True | 5.12 | 0.09 |
| google/gemini-2.5-pro | 2 | 1 | -0.864 | False | 10.00 | 0.71 |
| meta-llama/llama-4-maverick | 2 | 0 | -0.995 | True | 10.88 | 0.81 |
| openai/gpt-4o | 2 | 1 | 0.631 | True | -4.62 | 0.44 |

## Графики

- `plots/rounds_anthropic_claude-opus-4.7.png`
- `plots/rounds_google_gemini-2.5-pro.png`
- `plots/rounds_meta-llama_llama-4-maverick.png`
- `plots/rounds_openai_gpt-4o.png`
- `plots/decay_summary.png`
