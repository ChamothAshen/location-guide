import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('sigiriya_dataset.csv')

print(f"Current dataset size: {len(df)} rows")
print(f"Current locations: {sorted(df['location_name'].unique())}")

# Show old coordinates
mirror_wall_old = df[df['location_name'] == 'Mirror Wall'].head(1)
boulder_old = df[df['location_name'] == 'Boulder Gardens'].head(1)

if len(mirror_wall_old) > 0:
    print(f"\nOld Mirror Wall coords: lat={mirror_wall_old.iloc[0]['latitude']:.6f}, lon={mirror_wall_old.iloc[0]['longitude']:.6f}")
if len(boulder_old) > 0:
    print(f"Old Boulder Gardens coords: lat={boulder_old.iloc[0]['latitude']:.6f}, lon={boulder_old.iloc[0]['longitude']:.6f}")

# Remove old entries for both locations
df = df[~df['location_name'].isin(['Mirror Wall', 'Boulder Gardens'])].copy()

print(f"\nDataset after removing old entries: {len(df)} rows")

# Correct coordinates
MIRROR_WALL_LAT = 7.95751
MIRROR_WALL_LON = 80.75908

BOULDER_GARDENS_LAT = 7.95462
BOULDER_GARDENS_LON = 80.75469

# Descriptions
mirror_wall_desc = "Ancient wall covered with inscriptions and graffiti dating back to the 8th century. Contains verses written by visitors over centuries, offering insights into the historical and cultural significance of Sigiriya."
boulder_gardens_desc = "Ancient rock-shelter monasteries and intricate landscaping. Features massive boulders with meditation caves used by Buddhist monks."

# Generate samples
np.random.seed(42)
num_samples = 250
noise_std = 0.0003

# Mirror Wall samples
mirror_wall_samples = []
for _ in range(num_samples):
    lat = MIRROR_WALL_LAT + np.random.normal(0, noise_std)
    lon = MIRROR_WALL_LON + np.random.normal(0, noise_std)
    mirror_wall_samples.append({
        'latitude': lat,
        'longitude': lon,
        'location_name': 'Mirror Wall',
        'description': mirror_wall_desc
    })

# Boulder Gardens samples
boulder_gardens_samples = []
for _ in range(num_samples):
    lat = BOULDER_GARDENS_LAT + np.random.normal(0, noise_std)
    lon = BOULDER_GARDENS_LON + np.random.normal(0, noise_std)
    boulder_gardens_samples.append({
        'latitude': lat,
        'longitude': lon,
        'location_name': 'Boulder Gardens',
        'description': boulder_gardens_desc
    })

# Combine all
mirror_wall_df = pd.DataFrame(mirror_wall_samples)
boulder_gardens_df = pd.DataFrame(boulder_gardens_samples)

updated_df = pd.concat([df, mirror_wall_df, boulder_gardens_df], ignore_index=True)

# Save
updated_df.to_csv('sigiriya_dataset.csv', index=False)

print(f"\n✅ Updated dataset size: {len(updated_df)} rows")
print(f"✅ Mirror Wall samples: {len(mirror_wall_df)} (NEW coords: {MIRROR_WALL_LAT}, {MIRROR_WALL_LON})")
print(f"✅ Boulder Gardens samples: {len(boulder_gardens_df)} (NEW coords: {BOULDER_GARDENS_LAT}, {BOULDER_GARDENS_LON})")
print(f"\nAll locations ({len(updated_df['location_name'].unique())}): {sorted(updated_df['location_name'].unique())}")
print(f"\nLocation counts:\n{updated_df['location_name'].value_counts().sort_index()}")
