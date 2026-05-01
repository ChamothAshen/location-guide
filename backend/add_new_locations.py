import pandas as pd
import numpy as np

# Load existing dataset
df = pd.read_csv('sigiriya_dataset.csv')
print(f'Current dataset size: {len(df)} rows')
print(f'Current locations ({df["location_name"].nunique()}): {list(df["location_name"].unique())}')

# ===== Three New Locations =====

new_locations = [
    {
        "name": "Pahangala",
        "lat": 7.95898,
        "lon": 80.75776,
        "description": "A rocky outcrop located near the Sigiriya fortress. Pahangala offers panoramic views of the surrounding landscape and is believed to have been used as a lookout point during King Kashyapa's reign."
    },
    {
        "name": "Aligala Caves",
        "lat": 7.95781,
        "lon": 80.76061,
        "description": "Prehistoric cave shelters located near Sigiriya. Aligala Caves contain evidence of ancient human habitation dating back thousands of years, with stone tools and pottery fragments discovered at the site."
    },
    {
        "name": "Rock Shelter",
        "lat": 7.95636,
        "lon": 80.75941,
        "description": "A natural rock shelter near the Sigiriya fortress complex. Used by ancient inhabitants for protection from the elements, these shelters contain traces of early human settlement and monastic activity."
    }
]

# Generate 250 samples per location with GPS noise (matching existing pattern)
np.random.seed(123)
num_samples = 250
noise_std = 0.0003  # Same noise level as other locations

all_new_data = []
for loc in new_locations:
    for _ in range(num_samples):
        lat = loc["lat"] + np.random.normal(0, noise_std)
        lon = loc["lon"] + np.random.normal(0, noise_std)
        all_new_data.append({
            'latitude': lat,
            'longitude': lon,
            'location_name': loc["name"],
            'description': loc["description"]
        })

new_df = pd.DataFrame(all_new_data)

# Append to existing dataset
df_updated = pd.concat([df, new_df], ignore_index=True)

# Save
df_updated.to_csv('sigiriya_dataset.csv', index=False)

print(f'\n[OK] Updated dataset size: {len(df_updated)} rows')
print(f'[OK] New samples added: {len(new_df)} (250 per location x 3 locations)')
print(f'\n[LOCATIONS] Updated locations ({df_updated["location_name"].nunique()}):')
print(list(df_updated["location_name"].unique()))
print(f'\n[COUNTS] Location counts:')
print(df_updated["location_name"].value_counts().to_string())

# Show sample data for each new location
print(f'\n--- Sample Data for New Locations ---')
for loc in new_locations:
    samples = df_updated[df_updated["location_name"] == loc["name"]].head(3)
    print(f'\n>> {loc["name"]} (center: {loc["lat"]}, {loc["lon"]}):')
    for _, row in samples.iterrows():
        print(f'   lat={row["latitude"]:.6f}, lon={row["longitude"]:.6f}')
