import json

data = {
  "basic_config": {
    "default_json_filename": "/sdf/home/c/cjesus/REFACTORED/LUCiD/config/SK_geom_config.json",
    "data_dir": "",
    "temperature": 0.1,
    "n_events": 100,
    "k": 7,
    "nphot": 150000,
    "c_medium": 0.22540751879699247
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

energies = [400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800]

for i, E in enumerate(energies):
    if E < 1000:
        nphot = 50_000
    elif E < 1300:
        nphot = 100_000
    elif E < 1500:
        nphot = 150_000
    else:
        nphot = 200_000

    data["basic_config"]["nphot"] = nphot
    data["basic_config"]["data_dir"] = f"/sdf/data/neutrino/cjesus/photonsim_output/water/monoenergetic/event_by_event/mu-/{E}MeV/"
    with open(f"opt_config_{i}.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Config {i}: E={E}MeV, nphot={data['basic_config']['nphot']}")
