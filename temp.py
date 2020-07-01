# Find a location with a decent label split...


import pandas as pd

k = pd.read_csv('rawdata/cleanAUSData.csv')
print(k)

k = k[['Location', 'RainNext4days']]

k['averageRain'] = k.groupby(['Location'])['RainNext4days'].transform('mean')


print(k)

k = k.drop_duplicates(subset=['Location', 'averageRain'])

print(k)
