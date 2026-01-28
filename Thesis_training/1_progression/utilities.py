

class IPFDataLoader:
    """Load patient data from CSV and NPY files"""
    
    def __init__(self, csv_path: str, features_path: str, npy_dir: str):
        self.csv_path = csv_path
        self.features_path = features_path
        self.npy_dir = Path(npy_dir)
        
    def get_patient_data(self) -> Tuple[Dict, Dict]:
        """
        Load patient data and features
        
        Returns:
            patient_data: Dict with {patient_id: {'weeks', 'fvc_values', 'intercept', 'slope'}}
            features_data: Dict with {patient_id: {feature: value}}
        """
        # Load main CSV
        df = pd.read_csv(self.csv_path)
        
        # Load features
        features_df = pd.read_csv(self.features_path)
        
        # Build patient_data dictionary
        patient_data = {}
        for patient_id in df['Patient'].unique():
            patient_df = df[df['Patient'] == patient_id].sort_values('Weeks')
            
            weeks = patient_df['Weeks'].values
            fvc_values = patient_df['FVC'].values
            
            patient_data[patient_id] = {
                'weeks': weeks.tolist(),
                'fvc_values': fvc_values.tolist(),
                'intercept': float(patient_df['fvc_intercept0'].iloc[0]),
                'slope': float(patient_df['fvc_slope'].iloc[0])
            }
        
        # Build features_data dictionary
        features_data = {}
        for _, row in features_df.iterrows():
            patient_id = row['Patient']
            
            # Map column names (CSV uses different naming convention)
            features_data[patient_id] = {
                'approx_vol': float(row['ApproxVol_30_60']),
                'avg_num_tissue_pixel': float(row['Avg_NumTissuePixel_30_60']),
                'avg_tissue': float(row['Avg_Tissue_30_60']),
                'avg_tissue_thickness': float(row['Avg_Tissue_thickness_30_60']),
                'avg_tissue_by_total': float(row['Avg_TissueByTotal_30_60']),
                'avg_tissue_by_lung': float(row['Avg_TissueByLung_30_60']),
                'mean': float(row['Mean_30_60']),
                'skew': float(row['Skew_30_60']),
                'kurtosis': float(row['Kurtosis_30_60']),
                'age': float(row['Age']) if 'Age' in row else 65.0,  # Default if missing
                'sex': int(row['Sex']) if 'Sex' in row else 0,
                'smoking_status': int(row['SmokingStatus']) if 'SmokingStatus' in row else 0
            }
        
        return patient_data, features_data