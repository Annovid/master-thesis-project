# Эксперимент 1.2 — decay of cooperation (pilot)

Endowment: 20.0. Полная PGG, 10 раундов, 4 копии модели, T=0.7, transparency=on. Дизайн: `docs/06/3. Дизайн эксперимента.md`.

## Сводная таблица по сессиям

| model | session | rounds | R1 | R_last | R1−R_last | slope | cc_r | mean_std_within_round | incomplete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-4o-mini | 0 | 10 | 8.25 | 7.75 | 0.50 | -0.158 | 0.11 | 1.09 | False |

Колонки: `R1` и `R_last` — средний по 4 игрокам вклад в первом и последнем сыгранном раунде; `R1−R_last` — простая разность (положительная = есть decay); `slope` — OLS-наклон `mean_contribution(round) ~ round` по всем сыгранным раундам сессии (отрицательный = decay); `cc_r` — корреляция Пирсона между вкладом игрока в раунде r+1 и средним вкладом остальных в раунде r (>0 = conditional cooperation); `mean_std_within_round` — средняя дисперсия вкладов между 4 игроками внутри одного раунда, усреднённая по раундам (индикатор гетерогенности).

## Сводка по моделям

| model | n_sessions | n_incomplete | mean_slope | slope_sign_consistent | mean R1−R_last | mean cc_r |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4o-mini | 1 | 0 | -0.158 | False | 0.50 | 0.11 |

## Графики

- `plots/rounds_gpt-4o-mini.png`
- `plots/decay_summary.png`
