import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('sigiriya_dataset.csv')

print(f"Current dataset size: {len(df)} rows")
print(f"Mirror Wall samples before update: {len(df[df['location_name'] == 'Mirror Wall'])}")

# Get current Mirror Wall data
mirror_wall_old = df[df['location_name'] == 'Mirror Wall'].copy()
if len(mirror_wall_old) > 0:
    print(f"Old Mirror Wall coordinates (sample): lat={mirror_wall_old.iloc[0]['latitude']:.6f}, lon={mirror_wall_old.iloc[0]['longitude']:.6f}")

# Remove old Mirror Wall entries
df = df[df['location_name'] != 'Mirror Wall'].copy()

# Correct Mirror Wall coordinates
CORRECT_LAT = 7.95757
CORRECT_LON = 80.76056

# Description
description = "Ancient wall covered with inscriptions and graffiti dating back to the 8th century. Contains verses written by visitors over centuries, offering insights into the historical and cultural significance of Sigiriya."

# Generate 250 new samples with correct coordinates
np.random.seed(42)
num_samples = 250
noise_std = 0.0003

mirror_wall_samples = []
for _ in range(num_samples):
    lat = CORRECT_LAT + np.random.normal(0, noise_std)
    lon = CORRECT_LON + np.random.normal(0, noise_std)
    mirror_wall_samples.append({
        'latitude': lat,
        'longitude': lon,
        'location_name': 'Mirror Wall',
        'description': description
    })

mirror_wall_df = pd.DataFrame(mirror_wall_samples)

# Add corrected Mirror Wall data
updated_df = pd.concat([df, mirror_wall_df], ignore_index=True)

# Save
updated_df.to_csv('sigiriya_dataset.csv', index=False)

print(f"\n✅ Updated dataset size: {len(updated_df)} rows")
print(f"✅ Mirror Wall samples with correct coordinates: {len(mirror_wall_df)}")
print(f"✅ New Mirror Wall coordinates: lat={CORRECT_LAT}, lon={CORRECT_LON}")
print(f"\nAll locations: {sorted(updated_df['location_name'].unique())}")
