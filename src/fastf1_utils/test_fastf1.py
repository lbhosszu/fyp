import fastf1_utils

fastf1_utils.Cache.enable_cache("fastf1_cache")

session = fastf1_utils.get_session(2019, "Abu Dhabi", "R")
session.load(weather=True, laps=True, telemetry=False)

print(session.event)
print(session.results[["Abbreviation","Position","Status"]].head())
print(session.laps.head())
print(session.weather_data.head())
