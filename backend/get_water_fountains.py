import pandas as pd

df = pd.read_csv('sigiriya_dataset.csv')
wf = df[df['location_name']=='Water Fountains']

avg_lat = wf['latitude'].mean()
avg_lon = wf['longitude'].mean()

print(f'Water Fountains coordinates:')
print(f'Latitude: {avg_lat:.4f}')
print(f'Longitude: {avg_lon:.4f}')
print(f'\nTotal samples: {len(wf)}')
