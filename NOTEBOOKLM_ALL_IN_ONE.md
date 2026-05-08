# SVRAP All-in-One Knowledge Pack (Final Paper Version, 2026-04)

This document is a unified learning source for downstream tools such as NotebookLM.
It consolidates the current repository state, implementation logic, interfaces, runbooks,
result governance, and archived branch context.

Scope of this pack:
1. Current final pipeline and baseline definition
2. End-to-end Python + C++ execution logic
3. Key code-level behavior from main implementation files
4. Dataset and experiment organization
5. Output artifacts and result interpretation
6. Reproducible command cookbook
7. Known constraints and maintenance guidance
8. Archived historical branch context and migration notes

-------------------------------------------------------------------------------

## 1) One-Page Executive Summary

Project identity:
- Hybrid SVRAP solver for paper-ready reproducibility
- Neural prior (Python) + metaheuristic optimization (C++)

Final decisions already applied:
1. Baseline is unified as greedy top-k over p_route
2. Gumbel-based baseline/training branch is retired from the mainline
3. Historical divergent variants are archived, not active
4. results directory keeps only files coupled to current workflow

Main entry points:
- Python training/inference: svrap_solver.py
- C++ optimization entry: main.cpp
- Core C++ optimization logic: tabu_search.cpp, greedy.cpp

Core data exchange:
- Python exports node probabilities to attention_probs.csv
- C++ reads attention_probs.csv, builds initial on-tour set with top-k, then runs tabu search

-------------------------------------------------------------------------------

## 2) Canonical Source-of-Truth Files

Primary docs:
1. README.md
2. README_TECHNICAL.md
3. PRETRAINING_PIPELINE_QUANTIFIED.md

Primary runtime files:
1. svrap_solver.py
2. main.cpp
3. input.cpp / input.h
4. greedy.cpp / greedy.h
5. tabu_search.cpp / tabu_search.h

Operational directories:
1. formatted_dataset/
2. pretraining_dataset/
3. main_dataset/
4. models/
5. results/
6. archive/

-------------------------------------------------------------------------------

## 3) Current Final Pipeline

### 3.1 Stage A: Python training/inference

High-level behavior:
1. Load dataset and compute distance-derived feature matrices
2. Run dual-stream attention network
3. Train policy with sampled binary actions and top-k greedy baseline cost
4. Optionally include counterfactual node-guidance loss
5. Export probabilities and top-k backbone indices

Key outputs:
- attention_probs.csv
- backbone_indices.txt
- models/svrap_best_model_<dataset><variant>.pth
- training_log_<dataset><variant>.csv

### 3.2 Stage B: C++ tabu optimization

High-level behavior:
1. Read coordinates from dataset
2. Read attention_probs.csv if available
3. If neural init enabled and matching succeeds:
   - sort by p_route
   - choose top 20 percent (at least 2) as initial on-tour
4. Else fallback to legacy initialization
5. Run greedy local search
6. Run tabu search with optional strategy profile
7. Print best cost and runtime

Optional label export behavior:
- main.cpp supports an optional label_output_path parameter
- writes one integer per line (1 for on-tour, 0 for off-tour) from best solution

-------------------------------------------------------------------------------

## 4) Baseline and Strategy Definitions

### 4.1 Baseline used in final training logic

Inside svrap_solver.py training loop:
1. probs = softmax(logits)
2. route_scores = probs[:, 1]
3. choose k = max(2, int(n * TOP_K_ROUTE_RATIO))
4. greedy_actions[topk(route_scores)] = 1
5. baseline_cost = evaluate_solution(greedy_actions)

Advantage normalization:
- advantage = (cost - baseline_cost) / max(abs(baseline_cost), 1.0)

This is the active and final baseline behavior.

### 4.2 C++ strategy selector in main.cpp

Strategy names currently interpreted in main.cpp:
1. paper_baseline
2. baseline
3. no_nn
4. no_entropy
5. no_knn
6. simple_div

paper_baseline behavior:
- locks paper-aligned defaults
- ignores override arguments for tuned fields

-------------------------------------------------------------------------------

## 5) Objective and Cost Semantics

SVRAP objective approximation:
1. Routing term based on c_ij = a * l_ij
2. Allocation term based on d_ij = (10 - a) * l_ij
3. Isolation term with lambda_isol * D_i

Dynamic isolation coefficient:
- lambda_isol = 0.5 + 0.0004 * a^2 * n

Off-route node decision rule:
- per node, choose min(assign_candidate, loss_candidate)

Degeneracy guard in Python stage:
- if no route nodes selected, apply NO_ROUTE_PENALTY

-------------------------------------------------------------------------------

## 6) Model Architecture and Training Loss

### 6.1 Network architecture (svrap_solver.py)

1. Node embedding layer from coordinates
2. Two encoder streams:
   - routing encoder uses c_matrix bias
   - allocation encoder uses d_matrix bias
3. Cross attention from route stream to alloc stream
4. Binary classifier output per node: off_route / route

### 6.2 Loss composition (active mainline)

Training uses:
1. REINFORCE term from sampled categorical actions
2. Optional counterfactual node-guidance BCE term
3. Entropy bonus regularization term

Conceptual form:
- Loss = RL_LOSS_WEIGHT * reinforce_loss
         + NODE_LOSS_WEIGHT * node_loss
         - ENTROPY_BONUS_WEIGHT * entropy

Special mode:
- Some datasets can be configured as reinforce-only via EXTRA_REINFORCE_ONLY_DATASETS

-------------------------------------------------------------------------------

## 7) Quantified Scale and Workload

Current dataset counts:
1. pretraining_dataset candidate graphs: 30
2. main_dataset graphs: 13

Step count formula:
- Total optimization steps = N_p * E_p + N_m * E_f

Example at E_p = 200 and E_f = 100:
1. Pretraining steps: 30 * 200 = 6000
2. Finetuning steps: 13 * 100 = 1300
3. Total: 7300

-------------------------------------------------------------------------------

## 8) Result Governance (Cleaned Repo Policy)

Current policy:
1. Keep result files that are coupled to active scripts/docs
2. Remove uncoupled historical result variants from results/
3. Keep historical assets in archive/

Current top-level kept files under results/ include:
1. guided_vs_no_nn_summary.csv
2. guided_vs_no_nn_summary.md
3. main_dataset_no_berlin_parallel_raw.csv
4. main_dataset_no_berlin_parallel_raw_by_dataset/
5. main_dataset_p_route_summary.md
6. paper_baseline_large_10runs_raw.csv
7. paper_baseline_large_10runs_summary.csv
8. pr152_transfer_large_10runs.csv
9. p_route_heatmaps/
10. table_with_control_20260320_temp500_retrain.csv
11. table_with_control_20260320_temp500_retrain.md
12. table_with_control_20260320_temp500_retrain_no783.csv
13. table_with_control_20260320_temp500_retrain_no783.md
14. table_with_control_20260320_temp500_retrain_no783_paper_baseline.csv
15. table_with_control_20260320_temp500_retrain_no783_paper_baseline.md

Archive roots:
1. archive/p_route_divergence_20260419/
2. archive/cleanup_20260419_backup/

-------------------------------------------------------------------------------

## 9) Reproducibility Cookbook

### 9.1 Environment setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Build C++ executable:

```bash
make
```

or

```bash
g++ -std=c++17 -O2 -o svrap.exe main.cpp input.cpp greedy.cpp tabu_search.cpp
```

### 9.2 Minimal single-dataset run

Train/infer neural prior:

```bash
python svrap_solver.py --dataset formatted_dataset/berlin52.txt --train
```

Run C++ optimization:

```bash
./svrap.exe 7 formatted_dataset/berlin52.txt
```

### 9.3 Determinism guidance

1. Set Python seed through --seed
2. Keep dependency versions stable
3. Use the same hardware stack when comparing runs
4. Note that C++ side still has nontrivial randomness unless externally constrained

-------------------------------------------------------------------------------

## 10) Interface and File Protocols

Dataset format:
- each line: x,y

Probability format consumed by C++:
- attention_probs.csv with columns compatible with x,y,p_off,p_route

Backbone index format:
- backbone_indices.txt with one node index per line

Optional labels from C++:
- one integer per line in label_output_path
- 1 indicates on-tour, 0 indicates off-tour

-------------------------------------------------------------------------------

## 11) Complexity and Performance Notes

Let n be node count.

Approximate complexity highlights:
1. Distance matrix construction: O(n^2)
2. Full cost evaluation: worst-case O(n^2)
3. Tabu iteration: dominated by repeated candidate evaluations
4. Attention forward per layer: typical O(n^2 * d)
5. Counterfactual node loss in training: expensive on larger graphs

Likely bottlenecks:
1. C++ candidate cost evaluations in tabu loop
2. Python counterfactual per-node on/off evaluation

-------------------------------------------------------------------------------

## 12) Defensive Design and Failure Handling

Observed fallback logic:
1. Missing probability file -> fallback initialization path
2. Invalid rows in input files -> row-level skip instead of full crash
3. Per-instance try/catch in main.cpp -> one instance failure does not abort all
4. Empty neighborhood handling -> safe loop exit behavior

Operational cautions:
1. Keep working directory consistent so attention_probs.csv is discoverable
2. Ensure probability rows correspond to current dataset coordinates
3. Interpret final quality using C++ best cost as primary metric

-------------------------------------------------------------------------------

## 13) What Was Retired and Why

Retired from active mainline:
1. Gumbel-based training/baseline branch
2. Legacy variant scripts and temporary solvers

Rationale:
1. Final paper baseline and training protocol converged to top-k-based definition
2. Historical branches were kept only for traceability and forensic comparison
3. Mainline now prioritizes reproducibility and reduced branch ambiguity

Where old content lives:
- archive/p_route_divergence_20260419/

-------------------------------------------------------------------------------

## 14) Suggested Learning Order (for NotebookLM Q&A)

Recommended reading sequence:
1. README.md
2. PRETRAINING_PIPELINE_QUANTIFIED.md
3. README_TECHNICAL.md
4. svrap_solver.py
5. main.cpp
6. greedy.cpp + tabu_search.cpp
7. results/ key tables
8. archive/ for historical comparison

Suggested Q&A prompts after upload:
1. Explain exactly how baseline_cost is computed and used
2. Compare paper_baseline vs baseline strategy behavior in C++
3. Describe where topology prior enters tabu search decisions
4. Enumerate all places where fallback logic is applied
5. Summarize trade-offs between neural prior quality and tabu convergence

-------------------------------------------------------------------------------

## 15) Glossary

1. p_route: probability of node being on route
2. Backbone: top-k on-tour seed set from p_route ranking
3. on-tour / off-tour: whether node is in route set
4. Assignment / Isolation: per off-tour node cost-minimizing selection
5. Tabu list: short-term memory to block recent reverse moves
6. Diversification: mechanism to escape local minima
7. Path relinking: intensification between elite solutions

-------------------------------------------------------------------------------

## 16) Snapshot of Current Repository State (Semantic)

1. Mainline docs and code are aligned to final paper strategy
2. results/ is curated to coupled artifacts only
3. archive/ stores removed logs, temp outputs, and divergent historical branches
4. Top-k baseline is explicit in both docs and implementation
5. Gumbel branch remains only as archived history

-------------------------------------------------------------------------------

## 17) Closing Notes for Upload Use

This document is intentionally redundant and self-contained so an external LLM
workspace (NotebookLM or similar) can answer without traversing many files.

If you want an even stronger retrieval source, pair this file with:
1. README.md
2. README_TECHNICAL.md
3. PRETRAINING_PIPELINE_QUANTIFIED.md
4. svrap_solver.py
5. main.cpp

That 6-file bundle usually gives the best balance of conceptual and implementation context.
