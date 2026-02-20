"""
Generate Survival Analysis Ground Truth for 84 Patients

This script regenerates the ground truth with survival analysis metrics:
- event: 1 if FVC drops ≥10% from baseline, 0 otherwise
- time: time from baseline to event or censoring (in weeks)

Uses the same logic from event_extractor.py but only for the 84 patients
in the existing ground truth.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def get_baseline(df):
    """
    Get baseline FVC for each patient (week 0 or closest to 0)
    """
    baseline_rows = []

    for pid, group in df.groupby('Patient'):
        # If week 0 exists, use it
        if 0 in group['Weeks'].values:
            baseline_fvc = group.loc[group['Weeks'] == 0, 'FVC'].values[0]
            baseline_week = 0
        else:
            # Otherwise, choose the week closest to 0
            closest_idx = (group['Weeks'] - 0).abs().idxmin()
            baseline_fvc = group.loc[closest_idx, 'FVC']
            baseline_week = group.loc[closest_idx, 'Weeks']
        
        baseline_rows.append({
            'Patient': pid,
            'FVC_baseline': baseline_fvc,
            'baseline_week': baseline_week
        })

    baseline_df = pd.DataFrame(baseline_rows)
    return baseline_df


def compute_event_time(group):
    """
    Compute survival analysis metrics for each patient:
    - event: 1 if FVC drops ≥10% from baseline, 0 otherwise
    - time: time from baseline to event or last follow-up (in weeks)
    """
    group = group.copy()
    
    # Calculate percentage drop from baseline
    group['FVC_drop_pct'] = (group['FVC'] - group['FVC_baseline']) / group['FVC_baseline'] * 100
    
    # Find first occurrence of ≥10% FVC drop
    progression = group[group['FVC_drop_pct'] <= -10]
    
    if len(progression) > 0:
        # Event occurred
        event = 1
        event_week = progression.iloc[0]['Weeks']
        time = max(event_week - group['baseline_week'].iloc[0], 1)  # At least 1 week
    else:
        # Censored (no event during follow-up)
        event = 0
        last_week = group['Weeks'].iloc[-1]
        time = max(last_week - group['baseline_week'].iloc[0], 1)  # At least 1 week
        
    return pd.Series({
        'event': event,
        'time': time,
        'last_fvc': group['FVC'].iloc[-1],
        'last_week': group['Weeks'].iloc[-1],
        'max_fvc_drop_pct': group['FVC_drop_pct'].min()  # Most negative = largest drop
    })


def main():
    """Generate survival ground truth for 84 patients"""
    
    print("="*80)
    print("GENERATING SURVIVAL ANALYSIS GROUND TRUTH FOR 84 PATIENTS")
    print("="*80)
    
    # Paths
    existing_ground_truth_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Label_ground_truth\ground_truth.csv")
    train_csv_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Dataset\train.csv")
    output_path = Path(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\3_Survival_Analysis\84_patients\ground_truth_survival.csv")
    
    # Load existing ground truth to get the 84 patient IDs
    print(f"\n1. Loading existing ground truth from:")
    print(f"   {existing_ground_truth_path}")
    existing_gt = pd.read_csv(existing_ground_truth_path)
    patient_ids_84 = existing_gt['PatientID'].tolist()
    print(f"   ✓ Found {len(patient_ids_84)} patients")
    
    # Load train.csv with FVC measurements
    print(f"\n2. Loading FVC measurements from:")
    print(f"   {train_csv_path}")
    train_df = pd.read_csv(train_csv_path)
    print(f"   ✓ Loaded {len(train_df)} FVC measurements")
    print(f"   ✓ Patients in train.csv: {train_df['Patient'].nunique()}")
    
    # Filter to only the 84 patients
    train_df_84 = train_df[train_df['Patient'].isin(patient_ids_84)].copy()
    print(f"\n3. Filtered to 84 patients:")
    print(f"   ✓ {len(train_df_84)} FVC measurements")
    print(f"   ✓ {train_df_84['Patient'].nunique()} patients (should be 84)")
    
    if train_df_84['Patient'].nunique() != 84:
        print(f"\n   ⚠️ WARNING: Expected 84 patients but found {train_df_84['Patient'].nunique()}")
        missing = set(patient_ids_84) - set(train_df_84['Patient'].unique())
        if missing:
            print(f"   Missing patients: {sorted(missing)[:5]}... ({len(missing)} total)")
    
    # Get baseline FVC for each patient
    print(f"\n4. Computing baseline FVC (week 0 or closest)...")
    baseline_df = get_baseline(train_df_84)
    print(f"   ✓ Baselines computed for {len(baseline_df)} patients")
    print(f"\n   Baseline week distribution:")
    print(baseline_df['baseline_week'].value_counts().sort_index().head(10))
    
    # Merge baseline back to train_df_84
    train_df_84 = train_df_84.merge(
        baseline_df[['Patient', 'FVC_baseline', 'baseline_week']],
        on='Patient',
        how='left'
    )
    
    # Compute event and time for each patient
    print(f"\n5. Computing survival metrics (event & time)...")
    survival_df = train_df_84.groupby('Patient').apply(compute_event_time).reset_index()
    
    # Merge with baseline info
    survival_df = survival_df.merge(
        baseline_df[['Patient', 'FVC_baseline', 'baseline_week']],
        on='Patient',
        how='left'
    )
    
    # Rename Patient to PatientID for consistency
    survival_df.rename(columns={'Patient': 'PatientID'}, inplace=True)
    
    # Reorder columns
    survival_df = survival_df[[
        'PatientID',
        'FVC_baseline',
        'baseline_week',
        'event',
        'time',
        'last_fvc',
        'last_week',
        'max_fvc_drop_pct'
    ]]
    
    # Sort by PatientID
    survival_df = survival_df.sort_values('PatientID').reset_index(drop=True)
    
    # Print summary statistics
    print(f"\n6. Survival Analysis Summary:")
    print(f"   Total patients: {len(survival_df)}")
    print(f"   Events (progression): {survival_df['event'].sum()} ({survival_df['event'].mean()*100:.1f}%)")
    print(f"   Censored: {(survival_df['event']==0).sum()} ({(survival_df['event']==0).mean()*100:.1f}%)")
    print(f"\n   Time statistics (weeks):")
    print(f"     Mean: {survival_df['time'].mean():.1f}")
    print(f"     Median: {survival_df['time'].median():.1f}")
    print(f"     Min: {survival_df['time'].min():.1f}")
    print(f"     Max: {survival_df['time'].max():.1f}")
    print(f"\n   FVC drop statistics (%):")
    print(f"     Mean: {survival_df['max_fvc_drop_pct'].mean():.2f}")
    print(f"     Median: {survival_df['max_fvc_drop_pct'].median():.2f}")
    print(f"     Min: {survival_df['max_fvc_drop_pct'].min():.2f}")
    print(f"     Max: {survival_df['max_fvc_drop_pct'].max():.2f}")
    
    # Compare with existing ground truth
    print(f"\n7. Comparing with existing ground truth...")
    comparison = survival_df[['PatientID', 'event']].merge(
        existing_gt[['PatientID', 'has_progressed']],
        on='PatientID',
        how='inner'
    )
    
    agreement = (comparison['event'] == comparison['has_progressed'].astype(int)).mean()
    print(f"   ✓ Agreement: {agreement*100:.1f}%")
    
    if agreement < 1.0:
        print(f"\n   Disagreements:")
        disagreements = comparison[comparison['event'] != comparison['has_progressed'].astype(int)]
        for _, row in disagreements.head(5).iterrows():
            print(f"     {row['PatientID']}: event={row['event']}, has_progressed={row['has_progressed']}")
        if len(disagreements) > 5:
            print(f"     ... and {len(disagreements)-5} more")
    
    # Save to CSV
    print(f"\n8. Saving survival ground truth to:")
    print(f"   {output_path}")
    survival_df.to_csv(output_path, index=False)
    print(f"   ✓ Saved!")
    
    # Display first few rows
    print(f"\n9. Preview (first 5 rows):")
    print(survival_df.head().to_string(index=False))
    
    print(f"\n{'='*80}")
    print("✓ SURVIVAL GROUND TRUTH GENERATION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nColumns in output:")
    print(f"  - PatientID: Patient identifier")
    print(f"  - FVC_baseline: Baseline FVC value")
    print(f"  - baseline_week: Week of baseline measurement")
    print(f"  - event: 1 if ≥10% FVC drop occurred, 0 if censored")
    print(f"  - time: Time from baseline to event/censoring (weeks)")
    print(f"  - last_fvc: FVC at last follow-up")
    print(f"  - last_week: Week of last follow-up")
    print(f"  - max_fvc_drop_pct: Maximum FVC drop percentage (negative = decline)")
    
    return survival_df


if __name__ == "__main__":
    survival_df = main()
