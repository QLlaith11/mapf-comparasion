# MAPF Comparison: CBS-TA vs TSWAP vs ECBS

A Python implementation and Monte Carlo evaluation of three Multi-Agent Path Finding
(MAPF) algorithms on random 4-connected grids with obstacles:

| Algorithm | Idea | Guarantee | Source |
|-----------|------|-----------|--------|
| **CBS-TA** | Conflict-Based Search over a search *forest* of task assignments (Murty k-best) | Complete, sum-of-costs optimal | Hönig et al. (2018) |
| **TSWAP** | One-timestep planning with target swapping and deadlock rotation | Complete, sub-optimal | Okumura & Défago (2022) |
| **ECBS** | CBS with focal search (single Hungarian-assigned root) | Bounded sub-optimal (w) | Barer et al. (2014) |

Every experiment is preceded by a **pre-run validation**: a 15-run pilot study, sample-size
calculation (±10 SIC at 95%), power analysis (Cohen's d, 80% power), and a Shapiro-Wilk test that
selects a t-distribution CI (normal data) or a 5,000-resample bootstrap CI (non-normal data).

## Results at a glance

Sum of Individual Costs (SIC), mean ± std over *successful* runs. Full tables: [`results/`](results/validated_run_summary.txt).

| Config | TSWAP | CBS-TA | ECBS (w=1.5) |
|--------|-------|--------|--------------|
| 20×20, 20 agents | 100% · 111.3 | 62.5% · 100.1 | 80.0% · 105.6 |
| 40×40, 60 agents | 100% · 412.5 | 20.0% · 387.2 | 60.0% · 378.8 |
| 60×60, 100 agents | 100% · 858.0 | 0% | 55.0% · 794.4 |
| 40×40, 20 agents | 100% · 208.0 | 91.4% · 200.4 | 94.3% · 202.9 |
| 40×40, 60 agents | 100% · 412.5 | 23.3% · 383.7 | 56.7% · 377.6 |
| 40×40, 80 agents | 100% · 500.8 | 8.0% · 430.5 (n=2) | 48.0% · 439.8 |

Cohen's d (TSWAP vs ECBS) grows from 0.195 (20 agents) to 0.905 (80 agents) — see
[`docs/figures/effect_size_cohens_d.png`](docs/figures/effect_size_cohens_d.png).

> **Caveat:** SIC and runtime are conditioned on success. For CBS-TA/ECBS at low success rates the
> successful runs are the easier instances, so their means understate cost on unselected instances.

## Repository layout

```
src/mapf_experiments.py     all algorithms + Monte Carlo/validation framework
results/                    verbatim summary tables from the validated run
docs/MAPF_Comparative_Report.docx   full academic report (APA references, flowcharts, paper comparison)
docs/figures/               effect-size chart
requirements.txt
```

## Quick start

```bash
git clone 
cd mapf-comparison
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/mapf_experiments.py
```

**Runtime:** the full validated suite took roughly 1.5–2 hours on the author's machine
(pure Python; CBS-TA/ECBS hit their timeouts on hard instances). To try it quickly, edit
`scalability_configs` / `density_configs` in `generate_academic_report_with_validation()` to use
smaller grids and fewer `num_runs`.

The run writes `mapf_comprehensive_results_with_validation.csv` (success rate, SIC mean/std/CV, CI
method, mean/std runtime per config and algorithm).

## Implementation notes

- ECBS here uses a **single** Hungarian-optimal root, not the full ECBS-TA forest (CBS-TA-style /
  MinRoot) from Hönig et al. This is the main reason its success rate trails the published ECBS-TA.
- Randomness: `random` and `numpy` are seeded per test (seed 42, pilot seeds 42+i / 100+i). Runs
  within a test are not re-seeded, so results are reproducible for a given configuration list.
- `pygame` is imported by the source file but no visualisation is included in this module.
- Pure Python is roughly 50–100× slower than the C++ implementations in the source papers; compare
  trends and density-normalised behaviour, not absolute runtimes.

## References

- Hönig, W., Kiesel, S., Tinka, A., Durham, J. W., & Ayanian, N. (2018). Conflict-based search with
  optimal task assignment. *AAMAS 2018*.
- Okumura, K., & Défago, X. (2022). Solving simultaneous target assignment and path planning
  efficiently with time-independent execution. *AAAI 2022*.
- Barer, M., Sharon, G., Stern, R., & Felner, A. (2014). Suboptimal variants of the conflict-based
  search algorithm for the multi-agent pathfinding problem. *SOCS 2014*.
- Sharon, G., Stern, R., Felner, A., & Sturtevant, N. R. (2015). Conflict-based search for optimal
  multi-agent pathfinding. *Artificial Intelligence, 219*, 40–66.

The source papers are not redistributed here; please obtain them from the publishers.

