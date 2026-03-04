import pandas as pd

# Load dataset
df = pd.read_csv('sigiriya_dataset.csv')

# Get all unique locations with their average coordinates
locations = []
for location in df['location_name'].unique():
    loc_data = df[df['location_name'] == location]
    avg_lat = loc_data['latitude'].mean()
    avg_lon = loc_data['longitude'].mean()
    count = len(loc_data)
    
    locations.append({
        'location_name': location,
        'latitude': round(avg_lat, 6),
        'longitude': round(avg_lon, 6),
        'samples': count
    })

# Create DataFrame and sort by location name
locations_df = pd.DataFrame(locations).sort_values('location_name')

print("=" * 80)
print("ALL LOCATIONS IN DATASET - FOR FRONTEND TESTING")
print("=" * 80)
print()
print(locations_df.to_string(index=False))
print()
print("=" * 80)
print(f"Total Locations: {len(locations_df)}")
print(f"Total Samples: {df.shape[0]}")
print("=" * 80)
print()

# Print Postman/Frontend test format
print("\n📋 POSTMAN/FRONTEND TEST FORMAT:")
print("=" * 80)
for _, row in locations_df.iterrows():
    print(f"\n{row['location_name']}:")
    print(f"  {{")
    print(f'    "lat": {row["latitude"]},')
    print(f'    "lon": {row["longitude"]}')
    print(f"  }}")

# Print Markdown table
print("\n\n📊 MARKDOWN TABLE:")
print("=" * 80)
print("| Location | Latitude | Longitude | Samples |")
print("|----------|----------|-----------|---------|")
for _, row in locations_df.iterrows():
    print(f"| {row['location_name']} | {row['latitude']} | {row['longitude']} | {row['samples']} |")
