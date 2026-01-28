import pandas as pd


#Read csv train file
def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df




def main():
    file_path = r'C:\Users\frank\OneDrive\Desktop\ImagingBased-ProgressionPrediction\Dataset\train.csv'
    df = read_csv(file_path)

    #For each patient calculate the decay percentage between baseline and 52 weeks 
    #For the value at week 52 take only in the range of [-8,+8] weeks from week 52
    #For the value at week 0 take the closest value to week 0
    #Save also the rwo dates taken in consideration
    decay_data = []
    patient_ids = df['Patient'].unique()
    for patient_id in patient_ids:
        patient_data = df[df['Patient'] == patient_id]
        
        # Get baseline value (closest to week 0)
        baseline_data = patient_data.iloc[(patient_data['Weeks'] - 0).abs().argsort()[:1]]
        baseline_value = baseline_data['FVC'].values[0]
        baseline_week = baseline_data['Weeks'].values[0]
        
        # Get week 52 value (within [-8, +8] weeks from week 52)
        week_52_data = patient_data[(patient_data['Weeks'] >= 44) & (patient_data['Weeks'] <= 60)]
        if not week_52_data.empty:
            week_52_closest = week_52_data.iloc[(week_52_data['Weeks'] - 52).abs().argsort()[:1]]
            week_52_value = week_52_closest['FVC'].values[0]
            week_52_week = week_52_closest['Weeks'].values[0]
        else:
            #Place the closest value to week 52 even if out of range
            week_52_closest = patient_data.iloc[(patient_data['Weeks'] - 52).abs().argsort()[:1]]
            week_52_value = week_52_closest['FVC'].values[0]
            week_52_week = week_52_closest['Weeks'].values[0]
            
        # Calculate decay percentage
        decay_percentage = ((baseline_value - week_52_value) / baseline_value) * 100
        decay_data.append({
            'PatientID': patient_id,
            'BaselineFVC': baseline_value,
            'BaselineWeek': baseline_week,
            'Week52FVC': week_52_value,
            'Week52Week': week_52_week,
            'DecayPercentage': decay_percentage
        })

    decay_df = pd.DataFrame(decay_data)
    # Save to csv
    #decay_df.to_csv('decay_percentage_percent.csv', index=False)


    #From dataframe filter the pateints outside the range of [-5,10] for baselineweek and [42,62] for week52week
    filtered_decay_df = decay_df[
        (decay_df['BaselineWeek'] >= -5) & (decay_df['BaselineWeek'] <= 10) &
        (decay_df['Week52Week'] >= 42) & (decay_df['Week52Week'] <= 62)
    ]
    # Save filtered to csv
    #filtered_decay_df.to_csv('decay_percentage_percent_filtered.csv', index=False)

    #Count the patients that have both baselineweek and week52week in the specified ranges
    count_filtered_patients = filtered_decay_df.shape[0]
    print(f"Number of patients with both baseline week and week 52 week in the specified ranges: {count_filtered_patients}")

    #Create dataset based on filtered patients
    #Add progressed label or not based on decay percentage > 10%
    filtered_decay_df['has_progressed'] = filtered_decay_df['DecayPercentage'] > 10
    
    #Save to csv
    filtered_decay_df.to_csv('decay_percentage_progressed_label.csv', index=False)
    
    #Add from tran.csv, age,sex,smoking status
    train_df = read_csv(file_path)
    merged_df = pd.merge(filtered_decay_df, train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates(), left_on='PatientID', right_on='Patient', how='left')
    merged_df = merged_df.drop(columns=['Patient'])
    #Save to csv
    merged_df.to_csv('decay_percentage_progressed_label_with_demographics.csv', index=False)

if __name__ == "__main__":
    main()