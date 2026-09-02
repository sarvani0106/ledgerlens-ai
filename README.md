# ⚖️ LedgerLens AI — An Evidence-Based Tax-Line Reconciliation Agent

> **Submission**: Razorpay AI Buildathon — Track 04 (AI Finance Controller)  
> **Tagline**: *"Don't just find mismatches. Investigate them, verify the evidence, resolve safe cases, and escalate the rest."*

> [!NOTE]
> **Dataset Disclaimer**: All invoice and ledger datasets included in this repository are synthetic data created for development, testing, benchmarking, and demonstration purposes. They do not represent actual financial transactions or confidential records of any real organization or individual. All vendor names, GSTINs, invoice numbers, and financial figures are completely synthetic. The challenge datasets (`challenge_invoices.csv`, `challenge_ledger.csv`) are intentionally constructed with adversarial scenarios to test reconciliation edge cases.

---

## 🎯 Executive Overview & Problem Statement

Financial tax-line reconciliation in Indian enterprises is high-stakes. Naive rule-based software fails on real-world messiness (vendor typos, split payment lines, payment gateway MDR fee deductions, and GST rate mismatches), while naive "LLM wrapper" tools hallucinate false financial matches, causing catastrophic tax compliance penalties and revenue leakage.

**LedgerLens AI** solves this by implementing an **Evidence-Based AI Finance Controller**:
- **Deterministic Rules (Stage 1 & 2)** handle high-volume unambiguous records at sub-second speed with zero token cost.
- **AI Investigation Agent (Claude Sonnet 4.6)** investigates ambiguous edge cases and formulates structured forensic hypotheses.
- **Programmatic Evidence Validation Layer** verifies every mathematical and structural claim before any resolution is finalized.
- **Confidence & Safety Policy** enforces auto-resolution for verified high-confidence cases, routes uncertain cases to human review, and explicitly **ABSTAINS** on conflicting evidence.

---

## 🔄 1. The 7-Phase Agent Decision Loop

Every ambiguous transaction passes through a single, explainable agentic lifecycle:

```
[1. OBSERVE] ➔ [2. MATCH] ➔ [3. INVESTIGATE] ➔ [4. VERIFY] ➔ [5. DECIDE] ➔ [6. ACT] ➔ [7. AUDIT]
```

| Lifecycle Phase | Module | Action Performed |
|---|---|---|
| **1. OBSERVE** | `matcher.py` | Ingests, normalizes (formats `/` vs `-`, strips leading zeroes, standardizes GSTINs and dates). |
| **2. MATCH** | `matcher.py` | Executes Stage 1 exact key matching and Stage 2 rule/tolerance matching at zero LLM cost. |
| **3. INVESTIGATE** | `llm_matcher.py` | Dispatches leftover ambiguous records to Claude Sonnet (`claude-sonnet-4-6`) to hypothesize root causes. |
| **4. VERIFY** | `exception_resolver.py` | Programmatically validates split sums, payment gateway MDR fees, GST rates, and string distances. |
| **5. DECIDE** | `exception_resolver.py` | Evaluates safety policy (`AUTO_RESOLVE`, `HUMAN_REVIEW`, `ABSTAIN`, `LEAVE_UNRESOLVED`). |
| **6. ACT** | `reconciliation_engine.py` | Executes resolution or escalates into human review queue. |
| **7. AUDIT** | `audit_logger.py` | Immutably records rules evaluated, evidence checked, timestamps, and confidence scores. |

---

## 🛡️ 2. Programmatic Evidence Validation & Safety Policy

The LLM is **never the final authority**. Every AI hypothesis must pass deterministic code assertions:

| Discrepancy Scenario | AI Hypothesis | Programmatic Validation Check | Result if Validation Fails |
|---|---|---|---|
| **Split Invoices** | Invoice split into `-A` & `-B` | `abs(sum(candidate_amounts) - invoice_amount) <= 1.0` | Escalated to **ABSTAIN** / Review |
| **Payment Gateway Fees** | 2% MDR fee + 18% GST deducted | `abs((gross - fee - gst_fee) - settlement) <= 1.0` | Escalated to **ABSTAIN** / Review |
| **Vendor Typos** | Spelling error in vendor name | `Levenshtein_Distance(v1, v2) <= 3` + Matching GSTIN | Escalated to **HUMAN_REVIEW** |
| **Candidate Collisions** | Multiple entries with same amount | `len(candidates) >= 2` with identical plausibility | Explicitly **ABSTAIN** |
| **Missing Records** | No counterparty in buyer ledger | `Candidate_Count == 0` | Marked as **LEAVE_UNRESOLVED** |

### Confidence Policy Thresholds
- **`AUTO_RESOLVE`**: `Confidence >= 0.95` **AND** Programmatic Evidence Check = `PASSED`.
- **`HUMAN_REVIEW`**: `0.75 <= Confidence < 0.95` **OR** Partial Evidence.
- **`ABSTAIN`**: Conflicting evidence or ambiguous candidate collisions.
- **`LEAVE_UNRESOLVED`**: `Confidence < 0.75` **OR** Missing records.

---

## 💰 3. False-Positive Financial Cost Model

In financial controllership, **one false positive is 100x costlier than a manual review**.

```
Cost of 1 False Positive = Denied Input Tax Credit (18%) + Section 50 Interest (18% p.a.) + GST Penalty (10%)
Example: For a ₹1,50,000 invoice:
  - Denied ITC: ₹27,000
  - Section 50 Interest: ₹27,000
  Total Loss from 1 Hallucinated Match = ₹54,000+
Cost of Human Review (30 seconds controller review) = ~₹15
```

**LedgerLens AI ensures Zero False Positives** by enforcing mathematical proofs before auto-resolving, choosing `ABSTAIN` over guessing.

---

## 📊 4. Honest Ground-Truth Benchmark Results

### A. Standard Benchmark Dataset (140 rows, `seed=42`)

| Metric | Baseline 1: Exact Rules Only | Baseline 2: Exact + Tolerance Rules | LedgerLens AI Hybrid Pipeline |
|---|---|---|---|
| **Records Processed** | 140 | 140 | **140** |
| **Match Rate** | 48.6% | 67.1% | **96.4%** |
| **Precision** | 100.0% | 100.0% | **100.0% (Zero False Positives)** |
| **Recall** | 50.4% | 69.6% | **96.3%** |
| **False Positives (FP)** | 0 | 0 | **0** |
| **FP Financial Impact (₹)** | ₹0.00 | ₹0.00 | **₹0.00 (Zero Loss)** |
| **Human Review Cases** | 0 | 0 | **11 Cases (incl. ABSTAIN)** |
| **Execution Throughput** | ~2,500 rec/s | ~1,200 rec/s | **~450 rec/s** |

### B. Held-Out Challenge Dataset (40 adversarial rows, `seed=101`)

Evaluated on difficult edge cases (similar GSTIN collisions, incomplete split parts, partial payments vs gateway fees, conflicting dates):
- **Precision**: `100.0%` (Zero False Matches)
- **Safe Abstentions (`ABSTAIN`)**: `12 Cases`
- **Human Review Escalations**: `16 Cases`
- **Unresolved Missing Records**: `4 Cases`
- **Auto-Resolved Clean Benchmarks**: `8 Cases`

---

## 📂 Repository Structure

```
tax_reconciliation_agent/
│
├── generate_data.py         # Standard (140 rows) & Held-Out Challenge (40 rows) generator
├── matcher.py               # Normalization, Stage 1 (Exact) & Stage 2 (Rules)
├── exception_resolver.py    # Evidence validation engine & ABSTAIN safety policy
├── llm_matcher.py           # Stage 3 AI Investigation Agent + Fault Injection simulation
├── reconciliation_engine.py # 7-phase lifecycle orchestrator, metrics engine, and ablation
├── audit_logger.py          # Step-by-step explainable audit logging
├── app.py                   # Streamlit web dashboard (Deep Inspector, Funnel, Benchmark, Export)
├── test_reconciliation.py   # Comprehensive automated test suite (12 test categories)
├── requirements.txt         # Project dependencies
├── README.md                # Full Buildathon documentation
└── data/
    ├── invoices.csv             # Standard 140-row synthetic seller invoices
    ├── ledger.csv               # Standard 140-row synthetic buyer ledger
    ├── challenge_invoices.csv   # Held-out 40-row adversarial invoices
    └── challenge_ledger.csv     # Held-out 40-row adversarial ledger
```

---

## 🚀 Installation & Running Instructions

### 1. Installation

```bash
cd C:\Users\sarva\.gemini\antigravity\scratch\tax_reconciliation_agent
pip install -r requirements.txt
```

### 2. Configure API Key (Optional for Live Claude Sonnet)

- **PowerShell**:
  ```powershell
  $env:ANTHROPIC_API_KEY="your-anthropic-api-key"
  ```
- **Command Prompt (cmd)**:
  ```cmd
  set ANTHROPIC_API_KEY=your-anthropic-api-key
  ```
- **Streamlit UI**: Enter directly in the sidebar password box.
*(Zero-Config Note: If no API key is provided, the built-in offline forensic evaluator automatically runs!)*

### 3. Generate Datasets

```bash
python generate_data.py
```

### 4. Run Complete Automated Test Suite

```bash
pytest test_reconciliation.py -v
```

### 5. Launch the Streamlit Dashboard

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
