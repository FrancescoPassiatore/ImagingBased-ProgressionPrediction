# Cox Proportional Hazards Survival Analysis

Analisi di sopravvivenza per predire la progressione dell'IPF usando il modello Cox Proportional Hazards.

## Descrizione

Questo modulo implementa l'analisi di sopravvivenza usando:
- **Features CNN**: Embeddings da CT scan estratti con ResNet50
- **Features Handcrafted**: 9 features radiologiche quantitative
- **Demographics**: Età, sesso, status fumatore

## Differenze rispetto alla classificazione binaria

| Aspetto | Classificazione Binaria | Survival Analysis |
|---------|------------------------|-------------------|
| **Output** | Probabilità di progressione | Risk score + tempo all'evento |
| **Tempo** | Non considerato | Esplicitamente modellato |
| **Censoring** | Non gestito | Pazienti senza evento inclusi |
| **Metrica** | AUC, Accuracy | C-index (concordance) |
| **Interpretazione** | Predizione sì/no | Modellazione temporale del rischio |

## Vantaggi del Cox Model

1. **Gestione del tempo**: Modella quando avviene la progressione, non solo se avviene
2. **Censoring**: Include pazienti senza evento osservato (informazione preziosa)
3. **Hazard ratios**: Interpretazione diretta dell'effetto di ogni feature
4. **Kaplan-Meier**: Visualizzazione delle curve di sopravvivenza stratificate per rischio

## File generati

Per ogni fold e configurazione:

```
results/
├── cnn1_hand1_demo1/              # Configurazione full
│   ├── fold_1/
│   │   ├── fold_1_kaplan_meier.png         # Curve K-M (alto vs basso rischio)
│   │   ├── fold_1_hazard_ratios.png        # Top 20 hazard ratios con CI
│   │   └── fold_1_risk_distribution.png    # Distribuzione risk score
│   ├── fold_2/
│   ├── ...
│   └── cross_validation_summary.csv        # C-index per ogni fold
```

## Come eseguire

### 1. Configurazione singola (full features)
```bash
python run_cox_analysis.py --mode single --cnn 1 --hand 1 --demo 1
```

### 2. Solo CNN
```bash
python run_cox_analysis.py --mode single --cnn 1 --hand 0 --demo 0
```

### 3. Solo handcrafted + demographics
```bash
python run_cox_analysis.py --mode single --cnn 0 --hand 1 --demo 1
```

### 4. Ablation study (tutte le configurazioni)
```bash
python run_cox_analysis.py --mode ablation
```

Questo eseguirà 7 configurazioni:
- Demographics only
- Handcrafted only
- Handcrafted + Demographics
- CNN only
- CNN + Demographics
- CNN + Handcrafted
- Full (CNN + Handcrafted + Demographics)

## Interpretazione risultati

### C-index (Concordance Index)
- Range: 0.5 (random) - 1.0 (perfetto)
- Misura la capacità del modello di ordinare correttamente i pazienti per rischio
- Analogo all'AUC ma per dati di sopravvivenza

### Hazard Ratios
- HR > 1: Feature aumenta il rischio di progressione
- HR < 1: Feature riduce il rischio di progressione
- HR = 1: Feature non ha effetto

### Kaplan-Meier Curves
- Stratificazione in alto/basso rischio (mediana)
- Separazione delle curve indica buona discriminazione
- Confidence intervals mostrano l'incertezza

## Dipendenze

```bash
pip install lifelines scikit-learn pandas numpy matplotlib seaborn torch torchvision
```

## Struttura del codice

### `cox_survival_analysis.py`
- **SurvivalDataLoader**: Carica dati di sopravvivenza e features
- **CoxSurvivalAnalyzer**: Fit del Cox model con cross-validation
- Genera tutti i plot e le metriche

### `run_cox_analysis.py`
- Script di esecuzione con CLI
- Supporta modalità singola o ablation study

## Note tecniche

### Preprocessing
- Features normalizzate con StandardScaler (fit su train set)
- CNN features estratte usando lo stesso metodo di `ablation_study.py`
- Regularization: L2 penalizer = 0.1 (configurabile)

### Cross-validation
- Usa gli stessi 5 fold stratificati del resto del progetto
- Ogni fold è completamente indipendente (no data leakage)

### Gestione casi edge
- Pazienti senza CT scan vengono esclusi
- Convergenza del Cox model verificata per ogni fold
- Errori gestiti gracefully con messaggi informativi

## Esempio output

```
================================================================================
COX SURVIVAL ANALYSIS - CROSS VALIDATION
================================================================================

FOLD 1
======================================================================
Train: 142 patients, Val: 35 patients
Train C-index: 0.7234
Val C-index: 0.6891
✓ Saved Kaplan-Meier plot
✓ Saved hazard ratios plot
✓ Saved risk distribution plot

...

======================================================================
CROSS-VALIDATION SUMMARY
======================================================================
  Fold  Train C-index  Val C-index  N_train  N_val
     1         0.7234       0.6891      142     35
     2         0.7156       0.7023      142     35
     3         0.7289       0.6754      142     35
     4         0.7201       0.7145      142     35
     5         0.7178       0.6998      142     35
  Mean         0.7212       0.6962      142     35
   Std         0.0051       0.0148        0      0

✓ Results saved to: D:\FrancescoP\...\3_Survival_Analysis\results\cnn1_hand1_demo1
```

## Riferimenti

- Cox, D. R. (1972). "Regression models and life-tables"
- Lifelines documentation: https://lifelines.readthedocs.io/
- C-index: Harrell, F. E., et al. (1982). "Evaluating the yield of medical tests"
