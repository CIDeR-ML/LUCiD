import json

data = {
  "basic_config": {
    "default_json_filename": "",
    "data_dir": "/sdf/data/neutrino/cjesus/photonsim_output/water/monoenergetic/event_by_event/mu-/1050MeV/",
    "temperature": 0.1,
    "n_events": 150,
    "k": 7,
    "nphot": 75000,
    "c_medium": 0.22540751879699247,
    "physics_config": "config/SK_physics_config.json"
  },
  "optimization_weights": {"vertex_weight_scale": 1.0, "counts_weight_scale": 1.0},
  "learning_rates": {"position_learning_rate": 0.05, "direction_learning_rate": 0.005, "t0_learning_rate": 0.1, "energy_learning_rate": 10.0},
  "position_grid_search": {"pos_n_div": 5, "pos_levels": 6, "pos_fraction": 1.0, "pos_min_L": 0.001, "t0_n_div": 30, "t0_min": -15.0, "t0_max": 15.0},
  "cone_direction_search": {"cone_levels": 2, "cone_initial_div": 5, "cone_max_angle_deg": 30, "cone_reduction": 0.25},
  "energy_optimization": {"energy_delta": 400, "energy_scan_steps": 25},
  "gradient_descent": {"max_iterations": 300},
  "verbosity": {"level": 1},
  "storage": {"store_true_data": False},
  "detector_params": {"scatter_length": 50.0, "reflection_rate": 0.2, "absorption_length": 50.0, "qe": 0.065},
  "optimization_params": {"damping_factor": 0.998},
  "fixed_vertex": {"enabled": False, "position": [0.0, 0.0, 0.0]}
}

sensor_counts = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]

for i, n_sensors in enumerate(sensor_counts):
    data["basic_config"]["default_json_filename"] = f"/sdf/home/c/cjesus/REFACTORED/LUCiD/s3df_jobs/geom_config/geom_list/detector_{n_sensors}_sensors.json"
    with open(f"opt_config_{i}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Config {i}: n_sensors={n_sensors}, nphot=75000, n_events=150")
