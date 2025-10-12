import os
import requests
from aisetup import print_llm_response
from dotenv import load_dotenv

# Get the Weather API key from the .env file
# https://openweathermap.org/price
load_dotenv('.env', override=True)
api_key = os.getenv('WEATHER_API_KEY')

# Store the latitude & longitude value 
# Complete the code below to get the "feels_like" temperature at your current location
lat = 39.0469
lon = 77.4903
url = f"https://api.openweathermap.org/data/2.5/forecast?units=metric&cnt=1&lat={lat}&lon={lon}&appid={api_key}"
response = requests.get(url)

data = response.json()
print(data)
feels_like = data['list'][0]['main']['feels_like']
city = data['city']['name']
print(f"The temperature currently feels like {feels_like}°C in {city}.")
#The temperature currently feels like 17.24°C in Dong Ostang.