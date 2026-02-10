#Carica i checkpoint di entrambi gli esperimenti
import torch


checkpoint_ablation = torch.load(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression\ablation_study_results\ablation_cnn_only\fold_0\best_model.pth", weights_only=False)
checkpoint_progression = torch.load(r"D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\1_progression\results_resnet50_config_1\fold_0\best_model.pth", weights_only=False)

# Confronta le configurazioni
print("Ablation config:")
print(checkpoint_ablation['config'])

print("\nProgression config:")
print(checkpoint_progression['config'])

# Confronta gli state_dict (dimensioni dei layer)
ablation_state = checkpoint_ablation['model_state_dict']
progression_state = checkpoint_progression['model_state_dict']

print("\nFirst linear layer input dimension:")
print(f"Ablation: {ablation_state['classifier.0.weight'].shape}")
print(f"Progression: {progression_state['classifier.0.weight'].shape}")