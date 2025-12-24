# Next Steps: Designing a Robust Model for FVC Prediction with Irregular Time Series

## 1. Problem Restatement (Very Clear)

**Final objective**  
Predict **future FVC** (e.g. at 52 weeks) for each patient using:

- **CT scans** (static, high-dimensional, slice-based)
- **Tabular data** (demographics, handcrafted radiomics)
- **Longitudinal FVC measurements** (irregularly sampled, variable length)

**Core difficulty**  
Patients:
- do **not** share the same visit times  
- have **different numbers of FVC measurements**  

This is **not an implementation issue** — it is the *central modeling challenge*.

---

## 2. Key Insight (Guiding Principle)

> **Do not force temporal alignment across patients.**  
> Instead, **learn a representation of each patient’s early FVC trajectory** that naturally handles irregular sampling.

Trying to impose fixed timestamps will:
- introduce bias
- discard patients
- or inject artificial information via interpolation

A good model must **embrace irregularity**, not fight it.

---

## 3. Reformulate the Prediction Task (Critical Step)

### What *not* to predict first
- ❌ Raw `FVC@52w` directly from baseline data

This target is:
- noisy
- heterogeneous
- weakly identifiable from early information

### Recommended targets (in order of robustness)
1. **Future slope** (after an early observation window)
2. **ΔFVC relative to baseline**
3. **FVC@52w** (only once the above are stable)

This aligns with:
- your existing CNN → slope → corrector pipeline
- clinical reasoning (rate of decline matters more than absolute value)

---

## 4. Introduce a Temporal Encoder for FVC History

### Input (per patient)
An **irregular sequence** of visits:
[(week₁, FVC₁), (week₂, FVC₂), ..., (weekₙ, FVCₙ)]

Instead of aligning timepoints, represent each visit as:
(FVC_t, Δt_since_previous_visit)


This preserves:
- visit timing
- spacing information
- trajectory shape

---

### Temporal Encoder (Recommended Options)

Use a **small, data-efficient model**:

**Best choices (in order):**
1. **MLP over visit embeddings + pooling**
2. **GRU / LSTM with Δt as an explicit input**

Avoid (for now):
- full Transformers
- Neural ODEs
- large sequence models

Your dataset size (~170 patients) favors **low-variance models**.

---

### Output of Temporal Encoder
A **fixed-length embedding**:
z_fvc ← learned representation of early FVC trajectory

This replaces hand-crafted features like:
- early slope
- % change
- baseline-only summaries

---

## 5. Multimodal Fusion: Final Model Structure

### Conceptual Architecture

Early FVC visits (irregular)
│
▼
Temporal Encoder
│
▼
Trajectory embedding (z_fvc)
│
├───────────────┐
│ │
CT Encoder Tabular Encoder
(CNN) (MLP)
│ │
└───── concat ──┘
│
Prediction Head
(future slope / ΔFVC / FVC@52w)


### Important Design Choices
- CT scans are **static per patient** (correct assumption)
- Temporal modeling is applied **only** to FVC
- Fusion happens at the **patient level**
- Output is a **continuous value**, not a binary label

---

## 6. Why This Design Is the Right Next Step

- ✔ Handles irregular timestamps correctly
- ✔ Preserves temporal structure without alignment
- ✔ Scales to small clinical datasets
- ✔ Naturally extends your current pipeline
- ✔ Easy to ablate and interpret
- ✔ Defensible in a thesis or paper

This is how **serious clinical ML models** are built.

---

## 7. What Not to Do (Explicitly)

- ❌ Force interpolation to identical time grids as the main solution
- ❌ Pad sequences and rely on attention to “figure it out”
- ❌ Treat CT slices as time steps
- ❌ Jump directly to multimodal Transformers

These approaches look elegant but fail silently with small data.

---

## 8. Concrete Next Actions (Execution Plan)

1. **Choose an early observation window** (e.g. first 12 or 24 weeks)
2. **Implement a temporal encoder** for `(FVC, Δt)`
3. **Generate `z_fvc` for each patient**
4. **Fuse `z_fvc` with CT + tabular features**
5. **Predict future slope or ΔFVC**
6. **Evaluate with patient-level cross-validation**
7. **Run ablations**:
   - CT only  
   - FVC history only  
   - CT + FVC  
   - CT + FVC + tabular  

---

## 9. One-Sentence Takeaway

> **The right solution is not to align time — it is to learn the trajectory.**

This step moves your project from *engineering* to **proper longitudinal modeling**.
