import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Real data from user
data = [
    {"flavor": "Masala", "weight_g": 11.0, "chip_count": 7},
    {"flavor": "Masala", "weight_g": 11.2, "chip_count": 7},
    {"flavor": "Masala", "weight_g": 11.34, "chip_count": 7},
    {"flavor": "Hot and Sweet Chilli", "weight_g": 25.0, "chip_count": 15},
    {"flavor": "Hot and Sweet Chilli", "weight_g": 25.0, "chip_count": 16},
]



# Convert to DataFrame and save
df = pd.DataFrame(data)
df.to_csv("lays_chip_data.csv", index=False)
print("Real data saved to lays_chip_data.csv")