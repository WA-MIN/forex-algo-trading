# Findings and results

<!-- TODO: fill in after the final master evaluation run completes. -->
<!-- This file is structural scaffolding only. No fabricated metrics. -->

## Executive summary

<!-- TODO: 3 to 5 bullet points of the highest-impact discoveries. -->
<!-- Examples of the form to use:                                     -->
<!-- - Session conditioning: helped on N pairs, hurt on M pairs       -->
<!-- - LSTM vs LR: significantly better on K pairs (DM p < 0.05)      -->
<!-- - Best overall strategy: <name>, <pair>, composite <score>       -->

## Quantitative results

| Metric | Value | Baseline | Delta | Significance |
|--------|-------|----------|-------|--------------|
| <!-- TODO --> |  |  |  |  |

<!-- Recommended row set (fill in from output/master_eval/master_report.txt): -->
<!-- - Best rule-based composite (pair, strategy, score, BAH delta, DM p)     -->
<!-- - Best ML composite (pair, strategy, score, BAH delta, DM p)             -->
<!-- - Champion vs runner-up DM p-value (rule-based)                          -->
<!-- - Champion vs runner-up DM p-value (ML)                                  -->
<!-- - Average in-domain Sharpe vs average transfer Sharpe (LR)                -->
<!-- - Average in-domain Sharpe vs average transfer Sharpe (LSTM)              -->

## Key findings

<details>
<summary>Finding 1: TODO title</summary>

**What was observed:** <!-- TODO -->

**Why it matters:** <!-- TODO -->

**Supporting evidence:**

| Run | Config | Result |
|-----|--------|--------|
| <!-- TODO --> |  |  |

**Interpretation:** <!-- TODO -->

</details>

<details>
<summary>Finding 2: TODO title</summary>

**What was observed:** <!-- TODO -->

**Why it matters:** <!-- TODO -->

**Supporting evidence:**

| Run | Config | Result |
|-----|--------|--------|
| <!-- TODO --> |  |  |

**Interpretation:** <!-- TODO -->

</details>

<details>
<summary>Finding 3: TODO title</summary>

**What was observed:** <!-- TODO -->

**Why it matters:** <!-- TODO -->

**Supporting evidence:**

| Run | Config | Result |
|-----|--------|--------|
| <!-- TODO --> |  |  |

**Interpretation:** <!-- TODO -->

</details>

<details>
<summary>Finding 4: TODO title</summary>

**What was observed:** <!-- TODO -->

**Why it matters:** <!-- TODO -->

**Supporting evidence:**

| Run | Config | Result |
|-----|--------|--------|
| <!-- TODO --> |  |  |

**Interpretation:** <!-- TODO -->

</details>

## What worked exceptionally well

<details>
<summary>Standout result: TODO name</summary>

<!-- TODO: detailed explanation with numbers, comparisons, and why it stands out. -->
<!-- Reference the specific (pair, strategy, session, spread) cell.               -->
<!-- Cite the exact composite score and the corresponding row in results_all.csv. -->

</details>

## What did not work, and why

<!-- TODO: be honest. Failed attempts are evidence of scientific rigour, not weakness. -->
<!-- Examples of items worth documenting:                                              -->
<!-- - Strategies that screened well on validation but collapsed on test               -->
<!-- - Session-conditional models that underperformed their global counterpart        -->
<!-- - Pairs where no strategy in the system produced positive net Sharpe              -->
<!-- - Architectures or features that were tried earlier and abandoned                 -->

## Visualisations

<!-- TODO: reference plots in docs/assets/ or output/master_eval/plots/ with captions. -->
<!-- Suggested plots:                                                                  -->
<!-- - Per-pair composite-score bar chart (rule-based vs ML)                          -->
<!-- - 4 by 4 transfer matrix heatmaps per pair                                       -->
<!-- - Equity curves for the top three strategies on the test window                   -->
<!-- - Drawdown trajectories for the same three strategies                            -->
