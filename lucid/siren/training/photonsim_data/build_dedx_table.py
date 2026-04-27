#!/usr/bin/env python3
"""
Build the dE/dx vs distance lookup table (dE/dx variant of the SIREN input).

Reads per-energy PhotonSim ROOT files containing `OpticalPhotons` (event count)
and `dEdxHist_Distance` (2D dE/dx vs distance histogram), and writes
`dedx_lookup_table.h5` with average dE/dx entries per event over the
(energy, dE/dx, distance) grid.

Console-script entry point: `lucid-build-dedx-table`.

For the full ROOT → h5 → SIREN pipeline see `docs/SIREN_TRAINING_INPUTS.md`.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import uproot
import h5py
from pathlib import Path
import argparse
from tqdm import tqdm


class AverageDedxTable3D:
    """3D table for normalized dE/dx average distributions."""

    def __init__(self, energy_min=10, energy_max=2000, energy_step=10):
        """
        Initialize the average 3D table.

        Parameters:
        -----------
        energy_min : int
            Minimum energy in MeV
        energy_max : int
            Maximum energy in MeV
        energy_step : int
            Energy step size in MeV
        """
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.energy_step = energy_step

        # Use histogram bins (500x500)
        self.dedx_bins = 500
        self.distance_bins = 500

        # Energy values - use full range
        self.energy_values = list(range(energy_min, energy_max + 1, energy_step))

        # No need to skip energies
        self.skip_energies = []

        # Define bin ranges from histogram (matches PhotonSim definition)
        self.dedx_range = (0, 1000)  # 0 to 1000 keV/mm
        self.distance_range = (0, 10000)  # 0 to 10000 mm

        # 3D histogram data
        self.dedx_table = None
        self.average_table = None
        self.events_per_file = {}
        self.bin_edges = None
        self.bin_centers = None

    def process_single_file(self, root_file_path, energy):
        """Process a single ROOT file and extract 2D histogram data."""
        try:
            with uproot.open(root_file_path) as file:
                # Get number of events from the tree
                n_events = 10000  # Default
                if "OpticalPhotons" in file:
                    tree = file["OpticalPhotons"]
                    n_events = tree.num_entries
                    print(f"  {energy} MeV: {n_events} events in file")

                self.events_per_file[energy] = n_events

                # Get the 2D histogram for dE/dx
                if "dEdxHist_Distance" in file:
                    hist = file["dEdxHist_Distance"]
                    counts = hist.values()
                    edges_x = hist.axes[0].edges()  # dE/dx edges (keV/mm)
                    edges_y = hist.axes[1].edges()  # Distance edges (mm)

                    # Store raw counts
                    total_entries = counts.sum()
                    print(f"  Total dE/dx entries: {total_entries:,.0f} ({total_entries/n_events:.1f} per event)")

                    return counts, (edges_x, edges_y)
                else:
                    print(f"  WARNING: No dEdxHist_Distance in {energy} MeV file")
                    return None, None

        except Exception as e:
            print(f"  ERROR processing {energy} MeV: {e}")
            return None, None

    def create_3d_table(self, data_dir):
        """Create the average-normalized 3D table from ROOT files."""
        print(f"Creating average-normalized 3D dE/dx lookup table for {self.energy_min}-{self.energy_max} MeV...")

        data_path = Path(data_dir)

        # Initialize 3D arrays
        n_energies = len(self.energy_values)
        self.dedx_table = np.zeros((n_energies, self.dedx_bins, self.distance_bins))
        self.average_table = np.zeros((n_energies, self.dedx_bins, self.distance_bins))

        # Process each file
        print("\nProcessing files:")
        for idx, energy in enumerate(tqdm(self.energy_values)):
            energy_dir = data_path / f"{energy}MeV"

            # Try different file naming conventions
            file_path = energy_dir / "output.root"
            if not file_path.exists():
                # Try output_job_*.root pattern
                root_files = list(energy_dir.glob("output_job_*.root"))
                if root_files:
                    file_path = root_files[0]  # Use first match
                else:
                    # Try any .root file
                    root_files = list(energy_dir.glob("*.root"))
                    if root_files:
                        file_path = root_files[0]

            if not file_path.exists():
                print(f"\n  WARNING: File not found for {energy} MeV")
                continue

            raw_counts, edges = self.process_single_file(file_path, energy)

            if raw_counts is None:
                continue

            # Store raw counts
            self.dedx_table[idx] = raw_counts

            # Store bin edges from first file
            if idx == 0:
                self.bin_edges = (
                    np.array(self.energy_values),
                    edges[0],  # dE/dx edges
                    edges[1]   # distance edges
                )

                # Calculate bin centers
                self.bin_centers = (
                    np.array(self.energy_values),
                    (self.bin_edges[1][:-1] + self.bin_edges[1][1:]) / 2,
                    (self.bin_edges[2][:-1] + self.bin_edges[2][1:]) / 2
                )

            # Calculate average (entries per event)
            n_events = self.events_per_file[energy]
            self.average_table[idx] = raw_counts / n_events

        print(f"\n3D average dE/dx table created successfully!")
        print(f"Energy range: {self.energy_values[0]}-{self.energy_values[-1]} MeV ({len(self.energy_values)} energies)")
        print(f"Shape: {self.average_table.shape}")
        print(f"Total entries: {self.dedx_table.sum():,.0f}")
        print(f"Average entries per event: {self.dedx_table.sum() / sum(self.events_per_file.values()):.1f}")

    def save_table(self, output_path):
        """Save the average 3D table to HDF5 format."""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

        # Save to HDF5 file
        h5_file = output_path / "dedx_lookup_table.h5"

        with h5py.File(h5_file, 'w') as f:
            # Create main data group
            data_group = f.create_group('data')

            # Save table versions
            data_group.create_dataset('dedx_table_raw', data=self.dedx_table,
                                    compression='gzip', compression_opts=9)
            data_group.create_dataset('dedx_table_average', data=self.average_table,
                                    compression='gzip', compression_opts=9)

            # Create coordinate group
            coords_group = f.create_group('coordinates')

            # Save coordinate arrays
            coords_group.create_dataset('energy_values', data=self.energy_values)
            coords_group.create_dataset('energy_edges', data=self.bin_edges[0])
            coords_group.create_dataset('dedx_edges', data=self.bin_edges[1])
            coords_group.create_dataset('distance_edges', data=self.bin_edges[2])
            coords_group.create_dataset('energy_centers', data=self.bin_centers[0])
            coords_group.create_dataset('dedx_centers', data=self.bin_centers[1])
            coords_group.create_dataset('distance_centers', data=self.bin_centers[2])

            # Create metadata group
            meta_group = f.create_group('metadata')

            # Save scalar metadata
            meta_group.attrs['energy_min'] = self.energy_min
            meta_group.attrs['energy_max'] = self.energy_max
            meta_group.attrs['energy_step'] = self.energy_step
            meta_group.attrs['dedx_bins'] = self.dedx_bins
            meta_group.attrs['distance_bins'] = self.distance_bins
            meta_group.attrs['dedx_range_min'] = self.dedx_range[0]
            meta_group.attrs['dedx_range_max'] = self.dedx_range[1]
            meta_group.attrs['distance_range_min'] = self.distance_range[0]
            meta_group.attrs['distance_range_max'] = self.distance_range[1]
            meta_group.attrs['table_shape'] = self.average_table.shape
            meta_group.attrs['total_entries'] = self.dedx_table.sum()
            meta_group.attrs['normalization'] = 'average'
            meta_group.attrs['average_units'] = 'entries/event'
            meta_group.attrs['data_type'] = 'dedx'
            meta_group.attrs['photonsim_version'] = '1.0'
            meta_group.attrs['format_version'] = '1.1'

            # Save events per file as dataset (since it can be large)
            if self.events_per_file:
                events_data = np.array([(k, v) for k, v in self.events_per_file.items()],
                                     dtype=[('energy', 'i4'), ('events', 'i4')])
                meta_group.create_dataset('events_per_file', data=events_data)

        print(f"\n3D dE/dx lookup table saved to HDF5 format: {h5_file}")
        print(f"  - dedx_table_raw: Raw dE/dx counts")
        print(f"  - dedx_table_average: Average dE/dx entries per event")
        print(f"  - coordinates: Energy, dE/dx, distance grids")
        print(f"  - metadata: Table parameters and info")

    def visualize_table(self, output_path=None):
        """Create visualizations of the average 3D table."""
        if self.average_table is None:
            print("No data to visualize. Run create_3d_table first.")
            return

        print("\nCreating visualizations...")

        # Figure 1: Average visualizations
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # 1. Energy scaling (entries per event)
        ax = axes[0, 0]
        entries_per_event = []
        for i, energy in enumerate(self.energy_values):
            if energy in self.events_per_file:
                total = self.dedx_table[i].sum()
                n_events = self.events_per_file[energy]
                entries_per_event.append(total / n_events)
            else:
                entries_per_event.append(0)

        ax.plot(self.energy_values, entries_per_event, 'b-', linewidth=2, marker='o', markersize=2)
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('dE/dx Entries per Event')
        ax.set_title('dE/dx Entries per Event vs Energy')
        ax.grid(True, alpha=0.3)

        # 2-4. Sample average 2D slices
        sample_energies = [300, 1000, 1700]
        for i, energy in enumerate(sample_energies):
            ax = axes[0, i+1] if i < 2 else axes[1, 0]

            idx = self.energy_values.index(energy) if energy in self.energy_values else 0
            slice_2d = self.average_table[idx]

            # Use log scale for better visualization with minimum threshold
            with np.errstate(divide='ignore'):
                log_slice = np.log10(np.maximum(slice_2d, 1e-6))

            im = ax.imshow(log_slice.T, origin='lower', aspect='auto', cmap='viridis',
                          extent=[0, self.bin_edges[1][-1],
                                 0, self.bin_edges[2][-1]],
                          vmin=-6, vmax=np.log10(np.maximum(slice_2d.max(), 1e-6)),
                          interpolation='nearest')

            ax.set_xlabel('dE/dx (keV/mm)')
            ax.set_ylabel('Distance (mm)')
            ax.set_title(f'{energy} MeV (log scale)')
            ax.set_xlim(0, 500)  # Focus on lower dE/dx range
            ax.set_ylim(0, 10000)

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('log (entries/event)', rotation=270, labelpad=20)

        # 5. Integrated dE/dx distribution
        ax = axes[1, 1]
        for energy in [200, 1000, 1800]:
            if energy in self.energy_values:
                idx = self.energy_values.index(energy)
                # Sum over distance to get total entries at each dE/dx
                dedx_distribution = self.average_table[idx].sum(axis=1)
                dedx_values = self.bin_centers[1]
                ax.plot(dedx_values, dedx_distribution, linewidth=2, label=f'{energy} MeV')

        ax.set_xlabel('dE/dx (keV/mm)')
        ax.set_ylabel('Entries per Event')
        ax.set_title('dE/dx Distribution (summed over distance)')
        ax.set_xlim(0, 500)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # 6. Statistics
        ax = axes[1, 2]
        ax.axis('off')
        stats_text = f"""Average 3D dE/dx Table Statistics:

Energy range: {self.energy_min}-{self.energy_max} MeV
Valid energies: {len(self.energy_values)}
Skipped: {self.skip_energies}

Table dimensions: {self.average_table.shape}
dE/dx bins: {self.dedx_bins} (0-1000 keV/mm)
Distance bins: {self.distance_bins} (0-10m)

Total events: {sum(self.events_per_file.values()):,}
Total entries: {self.dedx_table.sum():,.0f}
Avg entries/event: {self.dedx_table.sum()/sum(self.events_per_file.values()):.1f}

Normalization: event average
Units: entries/event"""

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
                fontsize=10, family='monospace', verticalalignment='top')

        plt.suptitle('Average-Normalized 3D dE/dx Lookup Table', fontsize=16)
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path / "average_3d_dedx_table_visualization.png",
                       dpi=150, bbox_inches='tight')
            print(f"Visualizations saved to {output_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Build the dE/dx vs distance lookup table (dedx_lookup_table.h5) consumed by SIREN training (dE/dx variant).')
    parser.add_argument('--data-dir', '-d',
                       default='data/mu-',
                       help='Directory containing energy subdirectories with ROOT files (<E>MeV/output.root)')
    parser.add_argument('--output', '-o',
                       default='output/3d_dedx_lookup_table',
                       help='Output directory for table and visualizations')
    parser.add_argument('--energy-min', type=int, default=10,
                       help='Minimum energy in MeV')
    parser.add_argument('--energy-max', type=int, default=2000,
                       help='Maximum energy in MeV')
    parser.add_argument('--energy-step', type=int, default=10,
                       help='Energy step size in MeV')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations after building table')

    args = parser.parse_args()

    # Create table builder
    table_builder = AverageDedxTable3D(
        energy_min=args.energy_min,
        energy_max=args.energy_max,
        energy_step=args.energy_step
    )

    # Build the 3D table
    table_builder.create_3d_table(args.data_dir)

    # Save the table
    table_builder.save_table(args.output)

    # Create visualizations if requested
    if args.visualize:
        table_builder.visualize_table(args.output)

    print("\nDone! Average-normalized 3D dE/dx lookup table created successfully.")


if __name__ == "__main__":
    main()
