import pandas as pd

# Load dataset
df = pd.read_csv('sigiriya_dataset.csv')

# Calculate average coordinates for each location
locations = df.groupby('location_name')[['latitude', 'longitude']].mean().round(6).sort_index()

# Display table
print("\n" + "="*70)
print("ALL SIGIRIYA LOCATIONS - COORDINATES TABLE")
print("="*70)
print(f"\n{'Location Name':<30} {'Latitude':<12} {'Longitude':<12}")
print("-" * 70)

for loc, row in locations.iterrows():
    print(f"{loc:<30} {row['latitude']:<12.6f} {row['longitude']:<12.6f}")

print("="*70)
print(f"Total Locations: {len(locations)}")
print("="*70)

# Postman test format
print("\n\n" + "="*70)
print("POSTMAN TEST FORMAT (JSON Body)")
print("="*70)

for i, (loc, row) in enumerate(locations.iterrows(), 1):
    print(f"\n{i}. {loc}")
    print(f'   {{"lat": {row["latitude"]}, "lon": {row["longitude"]}}}')

print("\n" + "="*70)
