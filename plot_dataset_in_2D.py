import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils import load_single_event
from tools.visualization import create_detector_display
from tools.geometry import generate_detector

def main():
    # Set default values
    default_filename = 'events/single_event_data.h5'
    default_json_filename = 'config/cyl_geom_config.json'

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Display detector event visualization')
    parser.add_argument('--filename', type=str, default=default_filename,
                      help='The input h5 file containing event data')
    parser.add_argument('--json_filename', type=str, default=default_json_filename,
                      help='The JSON file containing detector geometry')
    parser.add_argument('--plot_time', action='store_true',
                      help='Plot hit times instead of charges')
    parser.add_argument('--output', type=str, default=None,
                      help='Output filename for the plot')

    args = parser.parse_args()

    # Generate detector to get number of PMTs
    detector = generate_detector(args.json_filename)
    NUM_DETECTORS = len(detector.all_points)

    # Load event data
    try:
        params, indices, charges, times = load_single_event(
            args.filename,
            NUM_DETECTORS,
            sparse=True
        )
    except Exception as e:
        print(f"Error loading event data: {e}")
        return

    # Print event parameters
    print("\nEvent Parameters:")
    print("─" * 20)
    print(f"Opening Angle: {params[0]:.2f} degrees")
    print(f"Initial Position: ({params[1][0]:.2f}, {params[1][1]:.2f}, {params[1][2]:.2f})")
    print(f"Initial Direction: ({params[2][0]:.2f}, {params[2][1]:.2f}, {params[2][2]:.2f})")
    print(f"Initial Intensity: {params[3]:.2f}")
    print("─" * 20)

    # Create detector display
    detector_display = create_detector_display(args.json_filename)

    # Generate output filename if not provided
    if args.output is None:
        output_base = os.path.splitext(args.filename)[0]
        output_suffix = '_time' if args.plot_time else '_charge'
        args.output = f"{output_base}{output_suffix}.png"

    # Create visualization
    detector_display(
        indices,
        charges,
        times,
        file_name=args.output,
        plot_time=args.plot_time
    )

    print(f"\nPlot saved to: {args.output}")

if __name__ == "__main__":
    main()