import pickle

# Load the splits
with open(r'D:\FrancescoP\ImagingBased-ProgressionPrediction\Thesis_training\Dataset\kfold_splits_stratified.pkl', 'rb') as f:
    splits = pickle.load(f)

print("Type:", type(splits))
print("Length:", len(splits))

if isinstance(splits, dict):
    print("\nKeys:", list(splits.keys()))
    
    # Check first fold
    first_fold = splits[0]
    print(f"\nFirst fold keys: {list(first_fold.keys())}")
    
    for key in first_fold.keys():
        val = first_fold[key]
        print(f"\n{key}:")
        print(f"  Type: {type(val)}")
        if isinstance(val, (list, tuple)):
            print(f"  Length: {len(val)}")
            if len(val) > 0:
                print(f"  First few items: {val[:3]}")
        elif hasattr(val, 'shape'):
            print(f"  Shape: {val.shape}")
            print(f"  First few items: {val[:3]}")
elif isinstance(splits, list):
    print(f"\nList with {len(splits)} elements")
    if len(splits) > 0:
        print(f"First element type: {type(splits[0])}")
        if isinstance(splits[0], dict):
            print(f"First element keys: {list(splits[0].keys())}")
        elif isinstance(splits[0], (list, tuple)) and len(splits[0]) >= 2:
            print(f"First fold: train={len(splits[0][0])}, test={len(splits[0][1])}")
