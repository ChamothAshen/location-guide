import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('sigiriya_dataset.csv')

# Correct Water Fountains coordinates
correct_lat = 7.957264470805178
correct_lon = 80.75561410058413

# Get Water Fountains entries
wf_mask = df['location_name'] == 'Water Fountains'
num_wf = wf_mask.sum()

print(f'Found {num_wf} Water Fountains entries')
print(f'Updating to correct coordinates:')
print(f'  Latitude:  {correct_lat}')
print(f'  Longitude: {correct_lon}')

# Generate new coordinates with GPS noise
np.random.seed(42)
noise_std = 0.0003  # Same noise level as other locations

new_lats = []
new_lons = []
for _ in range(num_wf):
    lat = correct_lat + np.random.normal(0, noise_std)
    lon = correct_lon + np.random.normal(0, noise_std)
    new_lats.append(lat)
    new_lons.append(lon)

# Update the dataframe
wf_indices = df[wf_mask].index
df.loc[wf_indices, 'latitude'] = new_lats
df.loc[wf_indices, 'longitude'] = new_lons

# Save updated dataset
df.to_csv('sigiriya_dataset.csv', index=False)

print(f'\n✅ Updated {num_wf} Water Fountains entries')
print(f'New average coordinates:')
wf_updated = df[df['location_name'] == 'Water Fountains']
print(f'  Latitude:  {wf_updated["latitude"].mean():.6f}')
print(f'  Longitude: {wf_updated["longitude"].mean():.6f}')
