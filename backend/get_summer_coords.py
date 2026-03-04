import pandas as pd

df = pd.read_csv('sigiriya_dataset.csv')
summer = df[df['location_name']=='Summer Palace'][['latitude','longitude']]

avg_lat = summer['latitude'].mean()
avg_lon = summer['longitude'].mean()

print(f'Summer Palace coordinates:')
print(f'Latitude: {avg_lat:.4f}')
print(f'Longitude: {avg_lon:.4f}')
print(f'\nSample coordinates:')
print(summer.head())
