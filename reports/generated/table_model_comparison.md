| Model | Split | MAE | RMSE | R2 | Dir. acc | Majority rate | Dir. skill |
|---|---|---|---|---|---|---|---|
| Global mean baseline | test | 1.90 | 2.84 | -0.00 | 0.81 | 0.81 | 0.00 |
| Persistence baseline | test | 2.23 | 4.52 | -1.54 | 0.77 | 0.81 | -0.03 |
| Country historical mean baseline | test | 1.94 | 2.88 | -0.03 | 0.78 | 0.81 | -0.03 |
| HistGradientBoostingRegressor (deployed) | test | 1.82 | 2.79 | 0.03 | 0.81 | 0.81 | 0.00 |
| Global mean baseline | validation | 4.04 | 6.14 | -0.18 | 0.51 | 0.51 | 0.00 |
| Persistence baseline | validation | 5.02 | 8.27 | -1.14 | 0.55 | 0.51 | 0.03 |
| Ridge (CV-best) | validation | 4.00 | 6.08 | -0.16 | 0.51 | 0.51 | 0.00 |
| HGB (CV-best) | validation | 3.89 | 5.97 | -0.11 | 0.53 | 0.51 | 0.01 |
