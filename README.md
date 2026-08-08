# RL Poker Investigations

Supervised Texas Hold'em equity prediction plus a CFR half. The supervised
side benchmarks six models (mean baseline, Ridge, Random Forest, LightGBM,
CatBoost, MLP) on bucketed Monte Carlo equity, with split-conformal and
quantile-forest uncertainty and three feature-importance measures. The CFR
side trains tabular CFR, Deep CFR, and Diffusion-CFR (a Deep CFR variant
whose advantage head is a conditional denoising-diffusion model) on Kuhn and
Leduc poker via OpenSpiel. 

The full write-up and discussion are published on
my personal website; every table and figure in it is reproducible from the
code in this repository.

## Why this project

This is a personal continuation of a final project from my last spring's Machine Learning course, I trained the original supervised equity
model. I kept putting off the two questions we didn't have time for that
semester, and eventually sat down to actually answer them after the semester ended. The pull toward
the CFR half specifically came from my ML professor's suggestion to look into
game-theoretic, imperfect-information-game methods, which is what got me
digging into CFR in the first place and led to building Diffusion-CFR.

## Main results

- LightGBM cuts weighted test MAE roughly 50% vs the Random Forest baseline
  at every street (river: 0.0141 to 0.0055).
- Nominal 90% split-conformal intervals cover about 81% of held-out buckets;
  weighted training breaks the exchangeability the guarantee needs.
- Diffusion-CFR's exploitability trends downward on Kuhn and Leduc over three
  seeds; on Leduc it is competitive with Deep CFR (1.874 ± 0.192 vs
  1.986 ± 0.048), on Kuhn behind (0.561 ± 0.062 vs 0.337 ± 0.004), and
  neither neural variant approaches tabular CFR at laptop-scale budgets.

## Setup

Python 3.10+.

```bash
pip install -r requirements.txt
```

Data is the [Texas Hold'em Monte Carlo dataset](https://www.kaggle.com/datasets/benjaminniesmertelny/texas-holdem-monte-carlo-data)
(Niesmertelny). On first use `kagglehub` downloads it automatically; the code
expects the CSVs at `data/raw/data/{preflop,flop,turn,river}_equity.csv`.

## Run

```bash
python scripts/train_equity.py --stage all                # equity benchmark: Tables 1-2, metrics CSVs
python scripts/train_tabular_cfr.py --game leduc --iters 300   # exact-CFR reference curve
python scripts/train_deep_cfr.py --game leduc --iters 200 --seed 42        # Deep CFR curve
python scripts/train_diffusion_cfr.py --game leduc --iters 100 --seed 42   # Diffusion-CFR curve
python scripts/evaluate_agents.py --game leduc --agents random equity_threshold --n-hands 2000  # head-to-head sanity check
python scripts/make_all_figures.py                        # rebuilds all figures from cached metrics
```

The full grid repeats the CFR runs with `--game kuhn` and seeds 42, 43, 44.

The Random Forest baseline is reproduced from the original class project
(supervised equity prediction); everything else here is a new addition to the
codebase.

*Dataset credit: Benjamin Niesmertelny (Kaggle).*
