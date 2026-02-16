import fastf1
fastf1.Cache.enable_cache("fastf1_cache")
print("FastF1 version:", fastf1.__version__)
print(fastf1.get_event_schedule(2025).head())
