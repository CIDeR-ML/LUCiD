# JUNO Geometry Viewer

This browser viewer renders the effective JUNO geometry exported from the
LUCiD configuration. It currently includes:

- LS--Acrylic, Acrylic--Water, and PMT-equator boundaries;
- PMT front heads grouped by PMT type;
- effective support rods grouped by family;
- the hollow Acrylic and stainless-steel chimney segments;
- Acrylic bowls with the same sampled 776 mm circular-arc profile used by
  photon transport.

The viewer is a geometry inspection tool. It does not run photon transport
or reproduce the full JUNOSW CAD geometry.

## Start on the BAT host

From the repository root, run:

```sh
/opt/gentoo/startprefix
/opt/gentoo/opt/PyMC/bin/python scripts/view_juno_geometry.py \
  --config config/JUNO_laser_geom_config.json \
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
- `AUTO ROTATE` starts or stops automatic rotation.
- Checkboxes independently show or hide every boundary and component group.
- The opacity sliders reveal structures hidden behind PMT front heads.
- The z section plane provides a clipped detector view.
- Left drag orbits, the wheel zooms, and right drag pans.

For a clear top-region demonstration, select `TOP DETAIL`, reduce PMT opacity
to about 0.2--0.4, and raise structure opacity to about 0.8--1.0.
