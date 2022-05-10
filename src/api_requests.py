import requests
import pandas as pd

params = {'query': 'Hanover'}
headers = {'Content-Type': 'application/json'}
result_ls = requests.get(
    'https://www.metaweather.com/api/location/search',
    params=params,
    headers=headers
)

json_ls = result_ls.json()

# params = {'query': 'Hanover'}
headers = {'Content-Type': 'application/json'}
result_loc = requests.get(
    'https://www.metaweather.com/api/location/' + str(json_ls[0]['woeid']),
#    params=params,
    headers=headers
)

json_loc = result_loc.json()

# params = {'query': 'Hanover'}
date = '2018/03/08'
headers = {'Content-Type': 'application/json'}
result_hist = requests.get(
    'https://www.metaweather.com/api/location/' + str(json_ls[0]['woeid']) + '/' + date + '/',
#    params=params,
    headers=headers
)

json_hist = result_hist.json()

# print('JSON Response: ', json_ls)
print("\n----------------------------\n")
print('City name: ', json_ls[0]['title'])
print('woeid: ', json_ls[0]['woeid'])
print('latt_long: ', json_ls[0]['latt_long'])
print(f"\n----------------------------\n\n5 day forecast for {json_ls[0]['title']}:")

for i in range(0,5):
    forecast = json_loc['consolidated_weather'][i]
    print("\n----------------------------\n")
    for k in forecast.keys():
        print(k, ":\t", forecast[k])

print(f"\n----------------------------\n\nHistoric forecast for {json_ls[0]['title']} on {date}:")
# print('JSON Response: ', json_hist[0])
print("\n----------------------------\n")
forecast_hist = json_hist[0]
for k in forecast_hist.keys():
    print(k, ":\t", forecast_hist[k])
