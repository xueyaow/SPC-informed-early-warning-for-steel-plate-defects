# SPC‑Informed Early Warning for Steel Plate Defects

## Overview

This project demonstrates how **classical Statistical Process Control (SPC)** can be extended into a **data‑driven early warning system** for manufacturing quality.

Using the Steel Plates Faults dataset, the work combines:

* **I–MR control charts** for process stability monitoring
* **SPC‑derived features** capturing instability and change
* **Explainable predictive models** for defect risk (focus: *Bumps*)

The emphasis is not on replacing SPC, but on **operationalizing SPC signals** within a modern analytics workflow.

---

## Dataset

Source: UCI Steel Plates Faults dataset.

Raw files used:

* `Faults.NNA` — numeric measurements for 1941 steel plates
* `Faults27x7_var` — variable names (27 process variables + 7 fault indicators)

Key characteristics:

* 1941 samples
* 34 columns total

  * 27 continuous process / surface / geometry variables
  * 7 binary fault indicators (`Pastry`, `Z_Scratch`, `K_Scatch`, `Stains`, `Dirtiness`, `Bumps`, `Other_Faults`)

The raw text files were parsed and combined programmatically to form a clean analysis table.

---

## Methodology

### 1. SPC Monitoring (I–MR)

Selected Critical‑to‑Quality (CTQ) variables were monitored using **Individuals–Moving Range (I–MR) charts**, appropriate for individual, sequential observations.

For each CTQ:

* Process mean and sigma estimated from moving ranges
* Control limits computed using standard SPC constants
* Special‑cause variation identified via limit violations and MR spikes

This stage answers:

> *Is the process behaving abnormally?*

---

### 2. SPC Feature Engineering

SPC theory was translated into **model‑ready, interpretable features**, including:

* Control‑limit violation flags
* Moving range magnitude
* Distance to control limits
* Rolling MR mean and standard deviation (local instability)

These features capture **process dynamics**, not just static values.

---

### 3. Predictive Modeling

A logistic regression model was trained to predict the occurrence of **Bumps** defects.

Two models were compared:

**Baseline model**

* Raw CTQs only (size, luminosity, orientation)

**SPC‑informed model**

* Raw CTQs + SPC‑derived instability features

Both models used the same:

* Target variable
* Train/test split
* Class‑imbalance handling

---

## Results

### Defect Prediction Performance (Bumps)

* Both models achieved **high recall**, suitable for early warning
* The SPC‑informed model:

  * Reduced false alarms compared to the raw‑feature baseline
  * Improved overall decision quality
  * Provided clearer physical interpretation

**Key insight:**

> Defects correlate more strongly with **process instability** than with absolute measurement values alone.

SPC‑derived features improved discrimination by encoding *how the process changes*, not just *where it is*.

---

## Figures

The following SPC charts are generated automatically by the analysis script:

* Individuals (I) Chart — `Pixels_Areas`
* Moving Range (MR) Chart — `Pixels_Areas`

These figures illustrate:

* Special‑cause events
* Sudden variability increases
* Short windows of relative process stability

(See the `figures/` directory.)

---

## System‑Level Deployment Concept

Beyond offline analysis, the project proposes a **two‑stage, plant‑ready quality monitoring architecture**:

1. **SPC Early Warning Layer**

   * Conservative, process‑centric monitoring
   * High interpretability and auditability

2. **Predictive Risk Scoring Layer**

   * SPC‑derived features as inputs
   * Fault‑specific risk probabilities
   * Reduced false alarms and better prioritization

Human operators, quality technicians, and engineers remain **in the loop**.

## How to Run

From the project root:

```powershell
& "D:\Anaconda\envs\spc_quality\python.exe" src/spc_analysis.py
```

The script will:

* Parse raw dataset files
* Compute SPC limits
* Generate SPC charts (saved to `figures/`)
* Train and evaluate baseline and SPC‑informed models

---

## Environment

A reproducible Conda environment is provided:

```yaml
name: spc_quality
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy
  - pandas
  - matplotlib
  - scipy
  - pip
  - pip:
      - scikit-learn
```

---

## Engineering Takeaway

This project shows how **classical SPC** can serve as a foundation for **modern predictive quality systems**.

Rather than replacing SPC, data‑driven models can:

* Leverage SPC signals as early indicators
* Improve prioritization and diagnosis
* Preserve interpretability and trust

The result is a practical, production‑aligned approach to quality analytics.
