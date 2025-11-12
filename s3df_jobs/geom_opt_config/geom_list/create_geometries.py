import json

# Base structure
data = {
    "detector_type": "cylinder",
    "geometry_definitions": {
        "radius": 16.9,
        "height": 36.2,
        "n_sensors": None,
        "sensor_radius": 0.25
    }
}

# Generate files with n_sensors = 1000 and 20000
for n in range(1000, 20001, 1000):
    data["geometry_definitions"]["n_sensors"] = n
    filename = f"detector_{n}_sensors.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {filename}")
