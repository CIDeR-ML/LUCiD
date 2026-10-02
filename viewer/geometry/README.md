# JUNO Geometry Viewer

This browser viewer renders the effective JUNO geometry exported from the
LUCiD configuration. It currently includes:

- LS--Acrylic, Acrylic--Water, and PMT-equator boundaries;
- PMT front heads grouped by PMT type;
- effective support rods grouped by family;
- the hollow transparent-Acrylic and opaque-steel top-chimney volumes;
- closed Acrylic-bowl volumes with the same sampled 776 mm circular-arc
  profile and flat end caps used by photon transport;
- the bottom-chimney Acrylic tube and flange, steel neck and flange, and
  lower-half ellipsoidal steel shell;
- source-attached PTFE diffuser ball, PEEK tube, and PE fibre using the same
  effective dimensions as photon transport.

The viewer is a geometry inspection tool. It does not run photon transport
or reproduce the full JUNOSW CAD geometry.

## Start on the BAT host

From the repository root, run:

```sh
/opt/gentoo/startprefix
source run/env.sh
JAX_PLATFORMS=cpu "$LUCID_PY" scripts/view_juno_geometry.py \
  --config config/JUNO_laser_geom_config.json \
  --source 0,0,17.1 \
  --source-axis 0,0,1 \
  --host 127.0.0.1 \
  --port 8766 \
  --no-open
```

If the browser runs on another computer, create an SSH tunnel there:

```sh
ssh -N -L 8766:127.0.0.1:8766 USER@BAT_HOST
```

Then open `http://127.0.0.1:8766/` in the local browser. The browser needs
network access to `unpkg.com` to load Three.js.

## Presentation controls

- `RESET VIEW` restores the whole-detector view.
- `TOP DETAIL` focuses on the chimney and nearby top-region structures.
- `BOTTOM DETAIL` focuses on the south-pole opening and bottom chimney.
- `SOURCE DETAIL` focuses on the millimetre-scale diffuser hardware at the
  source position selected on the command line.
- `AUTO ROTATE` starts or stops automatic rotation.
- Checkboxes independently show or hide every boundary and component group.
- The opacity sliders reveal structures hidden behind PMT front heads.
- The z section plane provides a clipped detector view.
- Left drag orbits, the wheel zooms, and right drag pans.

For a clear top- or bottom-region demonstration, select the corresponding
detail view, reduce PMT opacity to about 0.2--0.4, and raise structure opacity
to about 0.8--1.0.

The viewer displays the finite Acrylic geometry used by transport, but it is
not an optical-event viewer: Snell/Fresnel/TIR decisions, material state, and
bulk absorption diagnostics remain outputs of the forward simulation.

The diffuser hardware follows `--source` and `--source-axis`. For example,
use `--source 0,0,0`, `--source 0,0,16.1`, or `--source 0,0,17.1` to match a
simulation source position. The default is the detector centre with a `+z`
hardware axis.
