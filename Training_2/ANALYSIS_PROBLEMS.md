# Analisi Problemi Training Precedente

## 🔴 PROBLEMI CRITICI IDENTIFICATI

### 1. **CNN Training - Hyperparameters Subottimali**

**File**: `train_cnn_kfold.py`

```python
CONFIG = {
    'n_epochs': 50,
    'patience': 10,
    'batch_size': 4,  # TROPPO PICCOLO (solo 4 pazienti)
    'lr': 1e-4,       # Potenzialmente troppo basso
    'weight_decay': 1e-4,
}
```

**Problemi**:
- ✗ **Batch size troppo piccolo (4 pazienti)**: causa instabilità nel training
- ✗ **Learning rate potenzialmente basso**: convergenza lenta
- ✗ **No gradient accumulation**: con batch piccoli, dovrebbe essere usato
- ✗ **No data augmentation avanzata**: manca augmentation per CT scans
- ✗ **Scheduler troppo semplice**: ReduceLROnPlateau con patience=5 potrebbe ridurre LR troppo presto
- ✗ **Gradient clipping max_norm=1.0**: potrebbe essere troppo restrittivo

**Metriche attese ma mancanti**:
- Loss molto alto o instabile
- Overfitting sul training set
- Scarsa generalizzazione

---

### 2. **MLP Corrector - ASSENZA TOTALE DI NORMALIZZAZIONE FEATURES**

**File**: `utilities.py`, classe `CorrectorDataset`

```python
def __getitem__(self, idx):
    # ...
    feature_vector = np.concatenate([[slope_cnn], features])
    
    # Normalize
    if self.scaler is not None:
        feature_vector = self.scaler.transform(feature_vector.reshape(1, -1))[0]
```

**IL PROBLEMA**:
- ✅ Lo scaler ESISTE nel codice
- ✗ **MA** lo scaler viene fittato su **TUTTA** la feature vector (CNN slope + features)
- ✗ **Questo è SBAGLIATO** perché:
  - Il CNN slope è già normalizzato (tramite slope_scaler della CNN)
  - Le features handcrafted/demographic **NON sono normalizzate** prima
  - Lo StandardScaler viene fittato su valori con scale completamente diverse

**Scala delle features prima della normalizzazione**:
```
CNN slope:      [-2, 2]      (già normalizzato)
approx_vol:     [100000, 8000000]  ← ENORME
age:            [50, 90]
sex:            [0, 1]
smoking_status: [0, 1, 2]
```

**Conseguenza**: Il model impara principalmente da `approx_vol` che ha varianza dominante!

---

### 3. **Regolarizzazione Inadeguata**

**File**: `train_corrector_kfold.py`

```python
CONFIG = {
    'n_epochs': 200,
    'patience': 15,
    'batch_size': 32,
    'lr': 1e-3,
    'weight_decay': 1e-4,  # Solo weight decay L2
}
```

**Problemi**:
- ✗ **No dropout nei modelli corrector**
- ✗ **No early stopping efficace** (patience=15 è troppo alto per 200 epochs)
- ✗ **Weight decay unico**: non differenziato tra layers
- ✗ **No L1 regularization**: utile per feature selection

---

### 4. **Architettura Modelli Corrector**

Controllando i modelli in `utilities.py`:

```python
class SlopeCorrectorFull(nn.Module):
    def __init__(self, n_handcrafted=9, n_demographics=3):
        # 1 (cnn) + 9 (hand) + 3 (demo) = 13 input
        self.fc1 = nn.Linear(1 + n_handcrafted + n_demographics, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
```

**Problemi**:
- ✗ **No Dropout**: rischio overfitting alto
- ✗ **No BatchNorm**: training instabile
- ✗ **Architettura troppo semplice**: 3 layers potrebbero non bastare
- ✗ **No skip connections**: difficoltà nell'apprendimento

---

## 📊 RISULTATI OTTENUTI (Poor Performance)

```
CNN Only:                     Acc=0.418, Precision=0.000, AUC=0.516
CNN + Handcrafted:            Acc=0.465, Precision=0.657, AUC=0.533
CNN + Demographics:           Acc=0.465, Precision=0.648, AUC=0.528
CNN + Full:                   Acc=0.471, Precision=0.670, AUC=0.536
```

**Osservazioni**:
- CNN Only ha Precision=0 → predice sempre classe maggioritaria
- AUC ~0.5 → praticamente random
- Nessun miglioramento significativo con features aggiuntive → **LE FEATURES NON VENGONO USATE CORRETTAMENTE**

---

## ✅ PIANO DI RISOLUZIONE

### FASE 1: CNN Training (Training_2/CNN_Training)

1. **Hyperparameter Optimization con Optuna**:
   - Learning rate: [1e-5, 1e-3]
   - Batch size: [8, 16, 32]
   - Weight decay: [1e-5, 1e-3]
   - Dropout: [0.1, 0.5]
   - Optimizer: [AdamW, SGD with momentum]

2. **Miglioramenti architettura**:
   - Data augmentation avanzata (rotazioni, flip, brightness)
   - Gradient accumulation per batch size effettivi più grandi
   - Mixed precision training (AMP)
   - Scheduler cosine annealing con warm restarts

3. **Early stopping migliorato**:
   - Patience adattivo (5-10 epochs)
   - Monitor anche MAE oltre a MSE loss

---

### FASE 2: MLP Corrector (Training_2/MLP_Corrector)

1. **NORMALIZZAZIONE CORRETTA**:
   ```python
   # PRIMA normalizzare le features
   handcrafted_normalized = handcrafted_scaler.transform(handcrafted)
   demographics_normalized = demographic_scaler.transform(demographics)
   
   # POI concatenare con CNN slope (già normalizzato)
   feature_vector = np.concatenate([
       [slope_cnn],  # già [-2, 2]
       handcrafted_normalized,
       demographics_normalized
   ])
   ```

2. **Architettura migliorata**:
   ```python
   nn.Linear(input_dim, 128)
   nn.BatchNorm1d(128)
   nn.ReLU()
   nn.Dropout(0.3)
   nn.Linear(128, 64)
   nn.BatchNorm1d(64)
   nn.ReLU()
   nn.Dropout(0.2)
   nn.Linear(64, 32)
   nn.BatchNorm1d(32)
   nn.ReLU()
   nn.Dropout(0.1)
   nn.Linear(32, 1)
   ```

3. **Regolarizzazione**:
   - L1 + L2 regularization
   - Dropout per layer
   - Early stopping con patience=10
   - Learning rate scheduling

4. **Hyperparameter optimization con Optuna**:
   - Architecture depth: [3, 5] layers
   - Hidden dimensions: [32, 256]
   - Dropout rates per layer
   - L1/L2 penalties
   - Learning rate + scheduler

---

## 🎯 METRICHE ATTESE DOPO FIX

Con questi fix, ci aspettiamo:
- **AUC-ROC**: 0.65-0.75 (vs attuale 0.52-0.54)
- **Accuracy**: 0.60-0.70 (vs attuale 0.42-0.47)
- **Precision**: 0.60-0.75 (vs attuale 0.00-0.67)
- **Recall**: 0.50-0.70 (vs attuale 0.00-0.31)

---

## 📁 STRUTTURA PROPOSTA

```
Training_2/
├── CNN_Training/
│   ├── train_cnn_optuna.py          # Hyperparameter optimization
│   ├── train_cnn_best.py            # Training con best params
│   ├── config.yaml                  # Best hyperparameters
│   └── checkpoints/
│
├── MLP_Corrector/
│   ├── train_corrector_optuna.py    # Hyperparameter optimization
│   ├── train_corrector_best.py      # Training con best params
│   ├── config.yaml                  # Best hyperparameters
│   └── checkpoints/
│
└── utilities.py                     # Codice condiviso (CORRETTO)
```

---

## 🚀 PROSSIMI PASSI

1. ✅ Creata struttura Training_2/
2. [ ] Implementare utilities.py con normalizzazione corretta
3. [ ] Implementare train_cnn_optuna.py
4. [ ] Eseguire optimization CNN (100-200 trials)
5. [ ] Training CNN con best params
6. [ ] Implementare train_corrector_optuna.py
7. [ ] Eseguire optimization Corrector per ogni approach
8. [ ] Training finale e valutazione

Procediamo?
