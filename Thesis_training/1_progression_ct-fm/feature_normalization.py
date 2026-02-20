# Feature Normalization System for Ablation Study
# IMPORTANTE: Normalizzare SOLO sul training set, poi applicare ai val/test

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ablation_study import ABLATION_CONFIGS
from model_train import train_single_fold, aggregate_fold_results, create_ablation_comparison

class FeatureNormalizer:
    """
    Gestisce la normalizzazione delle features separatamente per train/val/test
    per evitare data leakage
    """
    
    def __init__(self, normalization_type: str = 'standard'):
        """
        Args:
            normalization_type: 'standard' (z-score), 'minmax' (0-1), o 'robust' (IQR-based)
        """
        self.normalization_type = normalization_type
        self.scalers = {}  # Dizionario di scalers per ogni feature type
        
    def fit(self, 
            features_df: pd.DataFrame,
            train_patient_ids: List[str],
            hand_feature_cols: List[str],
            demo_feature_cols: List[str]):
        """
        Fit normalizers SOLO sui dati di training
        
        Args:
            features_df: DataFrame completo con tutte le features
            train_patient_ids: Lista di patient IDs nel training set
            hand_feature_cols: Colonne delle hand-crafted features da normalizzare
            demo_feature_cols: Colonne delle features demografiche da normalizzare
        """
        # Filtra solo i pazienti di training
        train_df = features_df[features_df['patient_id'].isin(train_patient_ids)].copy()
        
        # Per le features a livello paziente, prendi un sample per paziente
        # (tutte le slices hanno gli stessi valori patient-level)
        train_patient_df = train_df.groupby('patient_id').first().reset_index()
        
        print("\n" + "="*70)
        print("FITTING NORMALIZERS (Training Set Only)")
        print("="*70)
        
        # Normalizza hand-crafted features (continue)
        if hand_feature_cols:
            available_hand = [col for col in hand_feature_cols if col in train_patient_df.columns]
            if available_hand:
                hand_scaler = self._create_scaler()
                hand_data = train_patient_df[available_hand].values
                
                # Fit scaler
                hand_scaler.fit(hand_data)
                self.scalers['hand'] = {
                    'scaler': hand_scaler,
                    'columns': available_hand
                }
                
                print(f"\nHand-crafted features ({len(available_hand)} features):")
                for col in available_hand:
                    orig_mean = train_patient_df[col].mean()
                    orig_std = train_patient_df[col].std()
                    print(f"  {col:30s}: mean={orig_mean:10.4f}, std={orig_std:10.4f}")
        
        # Normalizza demographics
        if demo_feature_cols:
            available_demo = [col for col in demo_feature_cols if col in train_patient_df.columns]
            
            # Separa features continue da categoriche
            continuous_demo = []
            categorical_demo = []
            
            for col in available_demo:
                # Age è continua, Sex e SmokingStatus sono categoriche
                if col == 'Age':
                    continuous_demo.append(col)
                else:
                    categorical_demo.append(col)
            
            # Normalizza solo le features continue
            if continuous_demo:
                demo_scaler = self._create_scaler()
                demo_data = train_patient_df[continuous_demo].values
                
                demo_scaler.fit(demo_data)
                self.scalers['demo'] = {
                    'scaler': demo_scaler,
                    'columns': continuous_demo,
                    'categorical': categorical_demo  # Non normalizzate
                }
                
                print(f"\nDemographic features (continuous: {len(continuous_demo)}, categorical: {len(categorical_demo)}):")
                for col in continuous_demo:
                    orig_mean = train_patient_df[col].mean()
                    orig_std = train_patient_df[col].std()
                    print(f"  {col:30s}: mean={orig_mean:10.4f}, std={orig_std:10.4f}")
                for col in categorical_demo:
                    unique_vals = train_patient_df[col].nunique()
                    print(f"  {col:30s}: categorical ({unique_vals} unique values) - NOT normalized")
    
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applica normalizzazione a un DataFrame usando i parametri dal training set
        
        Args:
            features_df: DataFrame da normalizzare
            
        Returns:
            DataFrame normalizzato
        """
        result_df = features_df.copy()
        
        # Normalizza hand-crafted features
        if 'hand' in self.scalers:
            scaler_info = self.scalers['hand']
            scaler = scaler_info['scaler']
            columns = scaler_info['columns']
            
            # Applica normalizzazione
            result_df[columns] = scaler.transform(result_df[columns])
        
        # Normalizza demographic features (solo quelle continue)
        if 'demo' in self.scalers:
            scaler_info = self.scalers['demo']
            scaler = scaler_info['scaler']
            continuous_cols = scaler_info['columns']
            
            # Applica normalizzazione solo alle continue
            if continuous_cols:
                result_df[continuous_cols] = scaler.transform(result_df[continuous_cols])
        
        return result_df
    
    def fit_transform(self, 
                      features_df: pd.DataFrame,
                      train_patient_ids: List[str],
                      hand_feature_cols: List[str],
                      demo_feature_cols: List[str]) -> pd.DataFrame:
        """
        Fit sul training set e transform sull'intero dataset
        """
        self.fit(features_df, train_patient_ids, hand_feature_cols, demo_feature_cols)
        return self.transform(features_df)
    
    def _create_scaler(self):
        """Crea lo scaler appropriato"""
        if self.normalization_type == 'standard':
            return StandardScaler()
        elif self.normalization_type == 'minmax':
            return MinMaxScaler()
        elif self.normalization_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown normalization type: {self.normalization_type}")
    
    def save(self, filepath: Path):
        """Salva i normalizers"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"Normalizers saved to: {filepath}")
    
    def load(self, filepath: Path):
        """Carica i normalizers"""
        with open(filepath, 'rb') as f:
            self.scalers = pickle.load(f)
        print(f"Normalizers loaded from: {filepath}")


def create_normalized_feature_dataframe(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    train_patient_ids: List[str],
    ablation_config: dict,
    normalization_type: str = 'standard'
) -> Tuple[pd.DataFrame, FeatureNormalizer]:
    """
    Crea DataFrame con features normalizzate correttamente
    
    Args:
        slice_features_df: DataFrame con CNN features per slice
        patient_features_df: DataFrame con hand-crafted e demographic features
        train_patient_ids: IDs dei pazienti nel training set (per fit normalizer)
        ablation_config: Configurazione ablation (quali features usare)
        normalization_type: Tipo di normalizzazione
    
    Returns:
        Tuple di (normalized_df, normalizer)
    """
    print("\n" + "="*70)
    print(f"CREATING NORMALIZED FEATURE SET: {ablation_config['description']}")
    print("="*70)
    
    # Start with CNN features (già normalizzate da ImageNet)
    result_df = slice_features_df.copy()
    
    # Define feature columns
    hand_features = [
        'ApproxVol_30_60',
        'Avg_NumTissuePixel_30_60',
        'Avg_Tissue_30_60',
        'Avg_Tissue_thickness_30_60',
        'Avg_TissueByTotal_30_60',
        'Avg_TissueByLung_30_60',
        'Mean_30_60',
        'Skew_30_60',
        'Kurtosis_30_60'
    ]
    
    demo_features = []
    if 'Age' in patient_features_df.columns:
        demo_features.append('Age')
    if 'Sex' in patient_features_df.columns:
        demo_features.append('Sex')
    if 'SmokingStatus' in patient_features_df.columns:
        demo_features.append('SmokingStatus')
    
    # Prepare patient-level features for merging
    patient_level_features = ['Patient']
    hand_to_add = []
    demo_to_add = []
    
    if ablation_config['use_hand_features']:
        available_hand = [f for f in hand_features if f in patient_features_df.columns]
        patient_level_features.extend(available_hand)
        hand_to_add = available_hand
        print(f"Adding {len(available_hand)} hand-crafted features")
    
    if ablation_config['use_demographics']:
        available_demo = [f for f in demo_features if f in patient_features_df.columns]
        patient_level_features.extend(available_demo)
        demo_to_add = available_demo
        print(f"Adding {len(available_demo)} demographic features")
    
    # Merge patient-level features
    if len(patient_level_features) > 1:
        result_df = result_df.merge(
            patient_features_df[patient_level_features],
            left_on='patient_id',
            right_on='Patient',
            how='left'
        )
        result_df.drop('Patient', axis=1, inplace=True)
        
        # Check for missing values
        missing = result_df[hand_to_add + demo_to_add].isnull().sum()
        if missing.any():
            print(f"\n⚠️ Missing values found:")
            print(missing[missing > 0])
            
            # Fill missing values BEFORE normalization
            for col in hand_to_add:
                if result_df[col].isnull().any():
                    # Per hand features, usa la mediana del training set
                    train_median = result_df[result_df['patient_id'].isin(train_patient_ids)][col].median()
                    result_df[col].fillna(train_median, inplace=True)
                    print(f"  Filled {col} with training median: {train_median:.4f}")
            
            for col in demo_to_add:
                if result_df[col].isnull().any():
                    if col == 'Age':
                        # Per Age, usa la mediana del training set
                        train_median = result_df[result_df['patient_id'].isin(train_patient_ids)][col].median()
                        result_df[col].fillna(train_median, inplace=True)
                        print(f"  Filled {col} with training median: {train_median:.4f}")
                    else:
                        # Per categoriche, usa 0 (unknown)
                        result_df[col].fillna(0, inplace=True)
                        print(f"  Filled {col} with 0 (unknown)")
    
    # NORMALIZZAZIONE
    normalizer = FeatureNormalizer(normalization_type=normalization_type)
    
    if hand_to_add or demo_to_add:
        print(f"\nNormalizing with: {normalization_type} scaler")
        result_df = normalizer.fit_transform(
            features_df=result_df,
            train_patient_ids=train_patient_ids,
            hand_feature_cols=hand_to_add,
            demo_feature_cols=demo_to_add
        )
        
        # Verifica normalizzazione (sul training set)
        train_sample = result_df[result_df['patient_id'].isin(train_patient_ids)].groupby('patient_id').first()
        
        print("\nPost-normalization statistics (Training Set):")
        if hand_to_add:
            print("\nHand-crafted features:")
            for col in hand_to_add:
                mean = train_sample[col].mean()
                std = train_sample[col].std()
                print(f"  {col:30s}: mean={mean:8.4f}, std={std:8.4f}")
        
        if demo_to_add:
            print("\nDemographic features:")
            for col in demo_to_add:
                if col == 'Age':  # Solo le continue
                    mean = train_sample[col].mean()
                    std = train_sample[col].std()
                    print(f"  {col:30s}: mean={mean:8.4f}, std={std:8.4f}")
                else:
                    unique = train_sample[col].nunique()
                    print(f"  {col:30s}: {unique} unique values (categorical, not normalized)")
    
    print(f"\nFinal feature set shape: {result_df.shape}")
    
    return result_df, normalizer


# AGGIORNAMENTO per ablation_study.py
# Sostituisci la funzione create_feature_enhanced_dataframe con questa versione:

def create_feature_enhanced_dataframe_normalized(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    fold_data: dict,  # NUOVO: per ottenere train IDs
    ablation_config: dict
) -> Tuple[pd.DataFrame, FeatureNormalizer]:
    """
    Versione che normalizza correttamente usando solo il training set
    """
    train_ids = fold_data['train']
    
    return create_normalized_feature_dataframe(
        slice_features_df=slice_features_df,
        patient_features_df=patient_features_df,
        train_patient_ids=train_ids,
        ablation_config=ablation_config,
        normalization_type='standard'  # o 'robust' per dati con outliers
    )


# ESEMPIO DI USO nel train_single_fold:

def train_single_fold_with_normalization(features_df, fold_data, fold_idx, config, results_dir):
    """
    Esempio di come usare la normalizzazione nel training
    """
    
    # Le features sono già normalizzate quando passi features_df
    # Ma salva il normalizer per reference
    if 'normalizer' in config:
        normalizer = config['normalizer']
        normalizer.save(results_dir / f"fold_{fold_idx}" / "feature_normalizer.pkl")
    
    # ... resto del training normale ...
    

# ESEMPIO COMPLETO di integrazione nell'ablation study:

def run_ablation_study_with_normalization(
    slice_features_df: pd.DataFrame,
    patient_features_df: pd.DataFrame,
    kfold_splits: dict,
    base_config: dict,
    results_base_dir: Path
):
    """
    Ablation study con normalizzazione corretta per fold
    """
    
    all_ablation_results = {}
    
    for config_name, ablation_config in ABLATION_CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"ABLATION: {config_name}")
        print(f"{'='*80}")
        
        ablation_results_dir = results_base_dir / f"ablation_{config_name}"
        ablation_results_dir.mkdir(parents=True, exist_ok=True)
        
        fold_results = []
        fold_keys = sorted(kfold_splits.keys())
        
        for fold_n in fold_keys:
            fold_idx = int(fold_n.split("_")[1])
            fold_data = kfold_splits[fold_n]
            
            # IMPORTANTE: Normalizza per ogni fold separatamente
            # usando solo i train IDs di quel fold
            features_df, normalizer = create_normalized_feature_dataframe(
                slice_features_df=slice_features_df,
                patient_features_df=patient_features_df,
                train_patient_ids=fold_data['train'],  # CRUCIAL!
                ablation_config=ablation_config,
                normalization_type='standard'
            )
            
            # Calcola feature dimension
            cnn_feature_cols = [c for c in features_df.columns if c.startswith('cnn_feature_')]
            hand_feature_cols = [c for c in features_df.columns if c in [
                'ApproxVol_30_60', 'Avg_NumTissuePixel_30_60', 'Avg_Tissue_30_60',
                'Avg_Tissue_thickness_30_60', 'Avg_TissueByTotal_30_60',
                'Avg_TissueByLung_30_60', 'Mean_30_60', 'Skew_30_60', 'Kurtosis_30_60'
            ]]
            demo_feature_cols = [c for c in features_df.columns if c in ['Age', 'Sex', 'SmokingStatus']]
            
            total_feature_dim = len(cnn_feature_cols) + len(hand_feature_cols) + len(demo_feature_cols)
            
            # Configura
            config = base_config.copy()
            config['feature_dim'] = total_feature_dim
            config['normalizer'] = normalizer  # Salva per reference
            
            # Train
            result = train_single_fold(
                features_df=features_df,
                fold_data=fold_data,
                fold_idx=fold_idx,
                config=config,
                results_dir=ablation_results_dir,
                resume_from_checkpoint=config['resume_from_checkpoint']
            )
            
            fold_results.append(result)
        
        # Aggregate
        summary_df, detailed_df = aggregate_fold_results(
            fold_results=fold_results,
            save_path=ablation_results_dir
        )
        
        all_ablation_results[config_name] = {
            'config': ablation_config,
            'summary': summary_df,
            'detailed': detailed_df,
            'fold_results': fold_results
        }
    
    create_ablation_comparison(all_ablation_results, results_base_dir)
    
    return all_ablation_results