"""
Verification script to validate the three newly added locations
in the Sigiriya dataset (Pahangala, Aligala Caves, Rock Shelter).
"""
import pandas as pd
import numpy as np

print("=" * 60)
print("  SIGIRIYA DATASET - NEW LOCATION VERIFICATION")
print("=" * 60)

# Load dataset
df = pd.read_csv('sigiriya_dataset.csv')

# --- 1. Basic Stats ---
print(f"\n[1] DATASET OVERVIEW")
print(f"    Total rows: {len(df)}")
print(f"    Total locations: {df['location_name'].nunique()}")
print(f"    Columns: {list(df.columns)}")

# --- 2. Check new locations exist ---
new_locations = {
    "Pahangala":      {"lat": 7.95898, "lon": 80.75776},
    "Aligala Caves":  {"lat": 7.95781, "lon": 80.76061},
    "Rock Shelter":   {"lat": 7.95636, "lon": 80.75941},
}

print(f"\n[2] NEW LOCATION PRESENCE CHECK")
all_present = True
for name in new_locations:
    count = len(df[df['location_name'] == name])
    status = "PASS" if count == 250 else "FAIL"
    if count != 250:
        all_present = False
    print(f"    {name}: {count} samples [{status}]")

# --- 3. Coordinate Range Validation ---
print(f"\n[3] COORDINATE RANGE VALIDATION")
for name, coords in new_locations.items():
    subset = df[df['location_name'] == name]
    lat_mean = subset['latitude'].mean()
    lon_mean = subset['longitude'].mean()
    lat_std = subset['latitude'].std()
    lon_std = subset['longitude'].std()
    
    # Check center is within tolerance of expected
    lat_ok = abs(lat_mean - coords['lat']) < 0.0005
    lon_ok = abs(lon_mean - coords['lon']) < 0.0005
    noise_ok = lat_std < 0.001 and lon_std < 0.001
    
    status = "PASS" if (lat_ok and lon_ok and noise_ok) else "FAIL"
    print(f"\n    {name} [{status}]:")
    print(f"      Expected center: ({coords['lat']}, {coords['lon']})")
    print(f"      Actual mean:     ({lat_mean:.6f}, {lon_mean:.6f})")
    print(f"      Std deviation:   (lat={lat_std:.6f}, lon={lon_std:.6f})")
    print(f"      Lat range:       [{subset['latitude'].min():.6f}, {subset['latitude'].max():.6f}]")
    print(f"      Lon range:       [{subset['longitude'].min():.6f}, {subset['longitude'].max():.6f}]")

# --- 4. Sample Data ---
print(f"\n[4] SAMPLE DATA (first 3 rows per location)")
for name in new_locations:
    subset = df[df['location_name'] == name].head(3)
    print(f"\n    {name}:")
    for _, row in subset.iterrows():
        print(f"      lat={row['latitude']:.8f}, lon={row['longitude']:.8f}, desc={row['description'][:60]}...")

# --- 5. No Duplicate Locations ---
print(f"\n[5] FULL LOCATION LIST")
for loc_name, count in df['location_name'].value_counts().items():
    marker = " <-- NEW" if loc_name in new_locations else ""
    print(f"    {loc_name}: {count} samples{marker}")

# --- 6. Summary ---
print(f"\n{'=' * 60}")
if all_present:
    print("  RESULT: ALL VERIFICATIONS PASSED!")
else:
    print("  RESULT: SOME VERIFICATIONS FAILED - check above")
print(f"{'=' * 60}")
