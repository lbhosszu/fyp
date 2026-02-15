import fastf1
fastf1.Cache.enable_cache("fastf1_cache")

print(fastf1.__version__)
print(fastf1.get_event_schedule(2024).head())
