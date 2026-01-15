# System-Level Deployment Concept: SPC‑Informed Quality Monitoring

## 1. Purpose

This document describes how the SPC‑informed early‑warning model developed in this project would be deployed in a real steel manufacturing plant. The goal is to move from offline analysis to a **practical, auditable, and operator‑usable quality monitoring system**.

The design emphasizes:

* Early detection of abnormal process behavior
* Clear separation between **monitoring** and **diagnosis**
* Human‑in‑the‑loop decision making
* Robustness to process drift and operational constraints

---

## 2. High‑Level Architecture

```
Sensors / Vision System
        ↓
Edge / Inspection Computer
        ↓
Process Historian / MES
        ↓
SPC Monitoring Service  ──► Operator Alerts
        ↓
Fault‑Risk Scoring Model ─► Quality / Engineering Review
```

The system is intentionally **two‑stage**:

1. **SPC early warning** (process‑centric, conservative)
2. **Predictive fault risk scoring** (data‑driven, selective)

---

## 3. Data Sources and Inputs

### 3.1 Measurement Inputs

Typical inputs available in a steel plate production line:

* Vision‑based defect metrics (e.g., area proxies such as `Pixels_Areas`)
* Surface intensity / luminosity metrics
* Geometry‑related indices (orientation, shape)
* Optional: line speed, roll force, temperature (if available)

Each record is timestamped and associated with a **plate ID**.

### 3.2 Data Storage

* Online buffer at the inspection station
* Long‑term storage in a **process historian** or MES database
* Data retained for:

  * SPC limit estimation
  * Root cause analysis
  * Model retraining and audit

---

## 4. Stage 1: Online SPC Monitoring (Early Warning)

### 4.1 Purpose

Stage 1 answers a single question:

> *Is the process behaving abnormally right now?*

It is **not fault‑specific** by design.

### 4.2 SPC Logic

For each selected CTQ:

* Maintain rolling estimates of:

  * Process mean
  * Moving range (MR)
  * Estimated process sigma
* Generate:

  * Individuals (I) chart
  * Moving Range (MR) chart

### 4.3 Alert Conditions

Alerts are triggered by:

* Control limit violations (I‑chart)
* Large MR spikes (sudden variability)
* Sustained runs or shifts (Western Electric–style rules)

### 4.4 Output of Stage 1

Each alert event contains:

* Timestamp / plate index
* CTQ involved
* Type of SPC violation
* Severity score (distance to control limit, MR magnitude)

This information is **operator‑readable and auditable**.

---

## 5. Stage 2: Fault‑Risk Scoring (Predictive Layer)

### 5.1 Purpose

Stage 2 answers:

> *Given abnormal behavior, what is the likelihood of a specific defect (e.g., Bumps)?*

It is activated:

* When Stage 1 flags instability, or
* Continuously with lower priority

### 5.2 Model Inputs

Inputs are **SPC‑derived features**, not raw measurements alone:

* Control limit violation flags
* Moving range values
* Rolling MR mean and standard deviation
* Distance to control limits

These features encode **process dynamics**, not just state.

### 5.3 Model Outputs

* Probability of each defect type (or top‑N risks)
* Confidence score
* Explanation signals (which SPC features contributed most)

The model supports **decision‑making**, not automated rejection.

---

## 6. Decision Logic and Human‑in‑the‑Loop Actions

### 6.1 Operator Actions

When alerts occur:

* Verify sensor cleanliness and lighting
* Check recent setup changes
* Inspect flagged plates

### 6.2 Quality Technician Actions

* Confirm defect type
* Record defect codes and context
* Provide feedback for model evaluation

### 6.3 Engineer Actions

* Root cause analysis (5‑Why, fishbone)
* Process parameter adjustment
* Update control plans or maintenance schedules

The system **supports people**; it does not replace them.

---

## 7. Alert Governance and False‑Alarm Control

To prevent alert fatigue:

* Two thresholds:

  * *Warning* (monitor)
  * *Action* (intervene)
* Alert suppression rules:

  * No duplicate alerts within a defined time window
  * Require multiple confirming signals
* Thresholds tuned based on **cost of false alarms vs missed defects**

This governance layer is essential for real adoption.

---

## 8. Model Monitoring and Drift Management

### 8.1 Sources of Drift

* Steel grade changes
* Sensor replacement or recalibration
* Lighting or environment changes
* Process upgrades

### 8.2 Monitoring Strategy

* Track feature distributions over time
* Monitor alert rates and false‑alarm ratios
* Periodic validation against confirmed defects

### 8.3 Retraining Policy

* Scheduled retraining (e.g., quarterly)
* Versioned models with rollback capability
* Retraining only with **verified labels**

---

## 9. Why This Design Works in Practice

This system:

* Aligns with traditional SPC philosophy
* Introduces data‑driven prediction safely
* Is explainable and auditable
* Matches how operators, quality staff, and engineers actually work

It treats SPC not as an obsolete tool, but as a **foundation for modern predictive quality systems**.

---

## 10. Summary

The proposed deployment integrates classical SPC and machine learning into a coherent, production‑ready quality monitoring framework. By separating early warning from diagnosis and keeping humans in the loop, it achieves both **practical usability** and **modern analytical capability**.
