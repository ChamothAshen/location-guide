import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('sigiriya_dataset.csv')
print(f'Current dataset size: {len(df)} rows')
print(f'Current locations: {df["location_name"].unique()}')

# Sigiriya Museum coordinates
museum_lat = 7.9570  # 7.9570° N
museum_lon = 80.7520  # 80.7520° E
description = 'Archaeological museum displaying artifacts, inscriptions, and historical exhibits from Sigiriya. Features detailed information about the site\'s history and King Kashyapa\'s reign.'

# Generate 250 samples with GPS noise
np.random.seed(42)
num_samples = 250
noise_std = 0.0003  # Same noise level as other locations

museum_data = []
for _ in range(num_samples):
    lat = museum_lat + np.random.normal(0, noise_std)
    lon = museum_lon + np.random.normal(0, noise_std)
    museum_data.append({
        'latitude': lat,
        'longitude': lon,
        'location_name': 'Sigiriya Museum',
        'description': description
    })

museum_df = pd.DataFrame(museum_data)

# Append to existing dataset
df_updated = pd.concat([df, museum_df], ignore_index=True)

# Save
df_updated.to_csv('sigiriya_dataset.csv', index=False)

print(f'\n✅ Updated dataset size: {len(df_updated)} rows')
print(f'Sigiriya Museum samples added: {len(museum_df)}')
print(f'\nUpdated locations ({len(df_updated["location_name"].unique())}):')
print(sorted(df_updated["location_name"].unique()))
print(f'\nLocation counts:')
print(df_updated["location_name"].value_counts().sort_index())
