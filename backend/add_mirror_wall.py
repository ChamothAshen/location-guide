import pandas as pd
import numpy as np

# Load existing dataset
df = pd.read_csv('sigiriya_dataset.csv')

print(f"Current dataset size: {len(df)} rows")
print(f"Current locations: {sorted(df['location_name'].unique())}")

# Mirror Wall coordinates
MIRROR_WALL_LAT = 7.9570
MIRROR_WALL_LON = 80.7432

# Description for Mirror Wall
description = "Ancient wall covered with inscriptions and graffiti dating back to the 8th century. Contains verses written by visitors over centuries, offering insights into the historical and cultural significance of Sigiriya."

# Generate 250 samples with GPS noise
np.random.seed(42)
num_samples = 250
noise_std = 0.0003  # Same noise level as other locations

mirror_wall_samples = []
for _ in range(num_samples):
    lat = MIRROR_WALL_LAT + np.random.normal(0, noise_std)
    lon = MIRROR_WALL_LON + np.random.normal(0, noise_std)
    mirror_wall_samples.append({
        'latitude': lat,
        'longitude': lon,
        'location_name': 'Mirror Wall',
        'description': description
    })

mirror_wall_df = pd.DataFrame(mirror_wall_samples)

# Add to existing dataset
updated_df = pd.concat([df, mirror_wall_df], ignore_index=True)

# Save
updated_df.to_csv('sigiriya_dataset.csv', index=False)

print(f"\n✅ Updated dataset size: {len(updated_df)} rows")
print(f"Mirror Wall samples added: {len(mirror_wall_df)}")
print(f"\nNew locations ({len(updated_df['location_name'].unique())}): {sorted(updated_df['location_name'].unique())}")
print(f"\nMirror Wall average coords: ({MIRROR_WALL_LAT}, {MIRROR_WALL_LON})")
