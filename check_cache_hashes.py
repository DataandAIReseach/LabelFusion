"""Check which percentages correspond to cached model hashes."""
import hashlib

# Available hashes in cache/old_models
available_hashes = [
    '108fe7f5', '348ebc33', '474cb2af', '5479c6c9', '989c531e', 
    'd53d1708', 'e1585558', 'e2825cfb', 'e5a80f22', 'fcc291fd'
]

# Full training size
full_train_size = 207389
num_labels = 28

print("="*80)
print("CACHE HASH LOOKUP")
print("="*80)

# Check percentages from 10% to 100%
print("\nChecking hashes for different percentages:")
print(f"{'Percentage':<12} {'Train Size':<12} {'Hash':<12} {'Cached?'}")
print("-"*60)

for pct in range(10, 101, 10):
    percentage = pct / 100.0
    train_size = int(full_train_size * percentage)
    
    # Calculate hash using same logic as eval_goemotions.py
    size_str = f"goemotions_train_size_{train_size}_labels_{num_labels}_seed_42"
    hash_obj = hashlib.sha256(size_str.encode('utf-8'))
    calculated_hash = hash_obj.hexdigest()[:8]
    
    is_cached = "✅ YES" if calculated_hash in available_hashes else "❌ NO"
    
    print(f"{pct:>3}% ({percentage:.1f})  {train_size:<12} {calculated_hash}  {is_cached}")

print("\n" + "="*80)
print("\nAvailable cached models:")
for h in available_hashes:
    print(f"  - fusion_roberta_model_{h}")
