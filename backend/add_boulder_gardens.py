import pandas as pd
import numpy as np

# Load existing dataset
df = pd.read_csv('sigiriya_dataset.csv')
print(f'Current dataset size: {len(df)} rows')
print(f'Current locations: {list(df["location_name"].unique())}')

# Boulder Gardens data
boulder_lat = 7.9556  # 7°57'20.2" N
boulder_lon = 80.7583  # 80°45'30.02" E
description = 'Ancient rock-shelter monasteries and intricate landscaping. Featuring massive boulders with inscriptions and cave dwellings used by Buddhist monks.'

# Generate 250 samples with GPS noise
np.random.seed(42)
num_samples = 250
noise_std = 0.0003  # Similar noise level to other locations

boulder_data = []
for _ in range(num_samples):
    lat = boulder_lat + np.random.normal(0, noise_std)
    lon = boulder_lon + np.random.normal(0, noise_std)
    boulder_data.append({
        'latitude': lat,
        'longitude': lon,
        'location_name': 'Boulder Gardens',
        'description': description
    })

boulder_df = pd.DataFrame(boulder_data)

# Append to existing dataset
df_updated = pd.concat([df, boulder_df], ignore_index=True)

# Save
df_updated.to_csv('sigiriya_dataset.csv', index=False)

print(f'\nUpdated dataset size: {len(df_updated)} rows')
print(f'Boulder Gardens samples added: {len(boulder_df)}')
print(f'\nUpdated locations ({len(df_updated["location_name"].unique())}):')
print(list(df_updated["location_name"].unique()))
print(f'\nLocation counts:')
print(df_updated["location_name"].value_counts())
