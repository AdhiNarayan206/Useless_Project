import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 1000
flavors = ["Masala", "Hot and Sweet Chilli", "Classic", "Sour Cream and Onion", "Barbecue", "Tangy Tomato"]
flavor_probs = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]  # Masala and Hot and Sweet Chilli more common
base_ratio = 0.623  # Chips per gram, based on real data

# Generate flavors
flavor_choices = np.random.choice(flavors, n_samples, p=flavor_probs)

# Initialize arrays
weights = np.zeros(n_samples)
chip_counts = np.zeros(n_samples)
crinkle_factors = np.random.randint(1, 11, n_samples)

# Generate weights and chip counts based on flavor
for i, flavor in enumerate(flavor_choices):
    if flavor == "Masala":
        weights[i] = np.random.uniform(10, 15)  # Small packets (10-15g)
        expected_chips = weights[i] * base_ratio
    elif flavor == "Hot and Sweet Chilli":
        weights[i] = np.random.uniform(20, 30)  # Medium packets (20-30g)
        expected_chips = weights[i] * base_ratio
    elif flavor == "Classic":
        weights[i] = np.random.uniform(10, 15)  # Small packets (10-15g)
        expected_chips = weights[i] * base_ratio
    elif flavor == "Sour Cream and Onion":
        weights[i] = np.random.uniform(20, 30)  # Medium packets (20-30g)
        expected_chips = weights[i] * base_ratio
    elif flavor == "Barbecue":
        weights[i] = np.random.uniform(40, 60)  # Large packets (40-60g)
        expected_chips = weights[i] * base_ratio
    elif flavor == "Tangy Tomato":
        weights[i] = np.random.uniform(40, 60)  # Large packets (40-60g)
        expected_chips = weights[i] * base_ratio

    # Add random noise (±10% of expected chips)
    noise = np.random.uniform(-0.1, 0.1) * expected_chips
    chip_counts[i] = np.clip(int(expected_chips + noise), 6, 40)  # Ensure realistic range

# Create DataFrame
synthetic_data = pd.DataFrame({
    "flavor": flavor_choices,
    "weight_g": weights,
    "chip_count": chip_counts
})

# Save synthetic data
synthetic_data.to_csv("lays_synthetic_data-updated.csv", index=False)
print("Generated 1000 fake Lay's packets for CrispProphet with maintained ratio!")