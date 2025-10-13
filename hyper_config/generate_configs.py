#!/usr/bin/env python3
"""
Generate hyperparameter tuning configuration files

Creates a grid of configuration files varying:
- scale_w: [500, 1000, 2500]
- inv_scale_w: [0.05, 0.5, 10.0]
- angle_scale_deg: [1.0, 3.0]

Each configuration uses 50 events for faster experimentation.
"""

import json
import itertools
from pathlib import Path

# Base config file
BASE_CONFIG = Path(__file__).parent.parent / 'config' / 'single_ring_optimization_config.json'
OUTPUT_DIR = Path(__file__).parent

# Parameter grid
# SCALE_W_VALUES = [1000, 2500]
# INV_SCALE_W_VALUES = [0.05, 0.5, 10.0]
# ANGLE_SCALE_DEG_VALUES = [1.0, 3.0]

SCALE_W_VALUES = [1]
INV_SCALE_W_VALUES = [0.05, 0.1, 0.5, 100.]
ANGLE_SCALE_DEG_VALUES = [2.0]

# Fixed n_events for hyperparameter tuning
N_EVENTS = 100

def main():
    """Generate all configuration files"""

    # Load base configuration
    with open(BASE_CONFIG, 'r') as f:
        base_config = json.load(f)

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate all combinations
    config_count = 0
    config_list = []

    for scale_w, inv_scale_w, angle_scale_deg in itertools.product(
        SCALE_W_VALUES, INV_SCALE_W_VALUES, ANGLE_SCALE_DEG_VALUES
    ):
        # Create modified config
        config = json.loads(json.dumps(base_config))  # Deep copy

        # Set n_events to 50
        config['basic_config']['n_events'] = N_EVENTS

        # Update optimization parameters
        config['optimization_params']['scale_w'] = scale_w
        config['optimization_params']['inv_scale_w'] = inv_scale_w
        config['optimization_params']['angle_scale_deg'] = angle_scale_deg

        # Create descriptive filename
        filename = f'config_sw{scale_w}_iw{inv_scale_w}_as{angle_scale_deg}.json'
        filepath = OUTPUT_DIR / filename

        # Save configuration
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        config_list.append({
            'filename': filename,
            'scale_w': scale_w,
            'inv_scale_w': inv_scale_w,
            'angle_scale_deg': angle_scale_deg
        })

        config_count += 1
        print(f"Created: {filename}")

    # Save config list for easy reference
    config_list_file = OUTPUT_DIR / 'config_list.json'
    with open(config_list_file, 'w') as f:
        json.dump(config_list, f, indent=2)

    print(f"\n✓ Generated {config_count} configuration files")
    print(f"✓ Config list saved to: {config_list_file}")
    print(f"\nParameter grid:")
    print(f"  scale_w:         {SCALE_W_VALUES}")
    print(f"  inv_scale_w:     {INV_SCALE_W_VALUES}")
    print(f"  angle_scale_deg: {ANGLE_SCALE_DEG_VALUES}")
    print(f"  Total combinations: {config_count}")


if __name__ == '__main__':
    main()
