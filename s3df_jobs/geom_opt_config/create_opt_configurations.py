import json

# Base structure
data = {
  "basic_config": {
    "default_json_filename": "/sdf/home/c/cjesus/Dev//LUCiD/s3df_jobs/geom_opt_config/geom_list/file.json",
    "data_file": "../data/water/muon/muon_gun_1050_MeV_100_events_fixed_energy.root",
    "data_dir": "/sdf/data/neutrino/cjesus/photonsim_output/water/monoenergetic/event_by_event/mu-/1050MeV/",
    "temperature": 0.1,
    "n_events": 100,
    "k": 7,
    "nphot": 100000,
    "c_medium": 0.22540751879699247
  },
  "optimization_weights": {
    "vertex_weight_scale": 1.0,
    "counts_weight_scale": 1.0
  },
  "learning_rates": {
    "position_learning_rate": 0.05,
    "direction_learning_rate": 0.005,
    "t0_learning_rate": 0.1,
    "energy_learning_rate": 10.0
  },
  "position_grid_search": {
    "pos_n_div": 5,
    "pos_levels": 6,
    "pos_fraction": 1.0,
    "pos_min_L": 0.001,
    "t0_n_div": 30,
    "t0_min": -15.0,
    "t0_max": 15.0
  },
  "cone_direction_search": {
    "cone_levels": 2,
    "cone_initial_div": 5,
    "cone_max_angle_deg": 30,
    "cone_reduction": 0.25
  },
  "energy_optimization": {
    "energy_delta": 400,
    "energy_scan_steps": 25
  },
  "gradient_descent": {
    "max_iterations": 300
  },
  "verbosity": {
    "level": 1
  },
  "storage": {
    "store_true_data": True
  },
  "detector_params": {
    "scatter_length": 50.0,
    "reflection_rate": 0.2,
    "absorption_length": 50.0,
    "qe": 0.065
  },
  "optimization_params": {
    "damping_factor": 0.998
  }
}

# Generate files with n_sensors = 1000 and 20000
for i,n in enumerate(range(1000, 20001, 1000)):
    data["basic_config"]["default_json_filename"] = f"/sdf/home/c/cjesus/Dev//LUCiD/s3df_jobs/geom_opt_config/geom_list/detector_{n}_sensors.json"
    filename = f"opt_config_{i}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {filename}")
