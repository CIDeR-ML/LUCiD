# fiTQun Cherenkov-profile scan

Cluster fan-out (SLURM or HTCondor — see
[`../../../../docs/guides/production/cluster-abstraction.md`](../../../../docs/guides/production/cluster-abstraction.md))
for the one fiTQun tuning table that needs a momentum scan of PhotonSim jobs.

Each cell runs PhotonSim on a macro generated at submit time, reduces the
photon list to `cell.npz`, and deletes the ROOT file. The reduction is a few
tens of kB against hundreds of MB of raw photons, so nothing else is kept.

The other four tuning stages don't fan out this way: see
[`../../fitqun/README.md`](../../fitqun/README.md).

## Quick start

```bash
# 1. Configure your paths (shared across all job stages)
cp ../user_paths.lxplus.sh.template ../user_paths.sh   # on LXPLUS, or
cp ../user_paths.s3df.sh.template   ../user_paths.sh   # on S3DF
vim ../user_paths.sh

# 2. Smoke test: one cell, 20 events
python3 generate_jobs.py -c configs/water_mu_test.json -t -s

# 3. Full scan, one particle at a time
python3 generate_jobs.py -c configs/water_mu.json -s
python3 generate_jobs.py -c configs/water_el.json -s
python3 generate_jobs.py -c configs/water_pi.json -s

# 4. Build the table once the cells are in
python -m lucid.production.fitqun cprofile build \
    $OUTPUT_BASE_PATH/fitqun_cprofile/13/*/cell.npz \
    --pdg 13 -o CProf_13_WCSim.root
```

Run `generate_jobs.py` with the **host** python3: it submits, and it imports
only the numpy-free parts of the package so a bare submit host is enough.

Re-running skips cells that already have `cell.npz`; `--no-skip-existing`
forces them.

## Output layout

```
<OUTPUT_BASE>/fitqun_cprofile/<pdg>/<p>MeV/
    photonsim.mac     generated at submit time
    cell.npz          the reduced profile — the only thing that persists
    cprofile.sub      the submit description
```

## Config schema

```json
{
  "name": "fitqun_cprofile_mu",
  "pdgs": [13],
  "seed_base": 1000,
  "events_schedule": {"500": 400, "1500": 200, "4000": 100, "1e9": 50},
  "request_disk_mb": 32768
}
```

`events_schedule` maps an upper momentum bound to an event count: photon yield
grows with track length, so a flat count would over-sample exactly the cells
that are slow and disk-hungry. `momentum_list_MeV` overrides the reference
grid (587 points, from `CprofileMomRepList.dat`) with an explicit list —
that's what the test config does.

`request_disk_mb` matters: PhotonSim writes the whole raw photon list before
it is reduced.
