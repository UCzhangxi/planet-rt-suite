# Planet RT User Guide

This guide explains how to use `run_planet_rt.py` and the provided YAML templates.

## 1) Files you can start from

- `planet_rt_local_dry_h2s_ch4.yaml`
- `planet_rt_local_moist_h2s_ch4.yaml`
- `planet_rt_cubesphere_dry_h2s_ch4.yaml`
- `planet_rt_cubesphere_moist_h2s_ch4.yaml`

Run with:

```bash
python run_planet_rt.py -c <yaml-file> --output-dir <output-dir>
```

## 2) Quick mode switches in YAML

### Opacity `data` field note
- In these templates, `opacities.*.data` is a compatibility placeholder for the in-script `GreyOpacity` path.
- No `.pt` opacity file is read by `run_planet_rt.py` for this grey JIT setup.
- Placeholder names like `placeholder_not_used_grey_sw.pt` are intentional.

### Geometry + domain decomposition
- `geometry.type`
  - `cartesian` for local box
  - `gnomonic-equiangle` for cubed sphere
- `distribute.layout`
  - `slab` (2D decomposition over x2/x3)
  - `cubed` (3D decomposition over x1/x2/x3)
  - `cubed-sphere` (6-face decomposition)
- `distribute.nb1`, `distribute.nb2`, `distribute.nb3`
  - For `slab`: world size = `nb2 * nb3`
  - For `cubed`: world size = `nb1 * nb2 * nb3`
  - For `cubed-sphere`: world size = `6 * nb2 * nb3` and `nb1` is effectively 1

### Dry vs moist chemistry
- `chemistry.enable_kinetics: false` => dry/non-reactive tracer run (faster)
- `chemistry.enable_kinetics: true` => evolve kinetics each step
- If `reactions:` is present and `enable_kinetics` is omitted, kinetics is enabled automatically.

### Radiative forcing geometry
- `radiative-transfer.zenith_mode`
  - `nadir`: direct overhead beam everywhere (local box use)
  - `orbital`: cubed-sphere orbital/day-night forcing
  - `auto`: orbital for cubed-sphere, nadir otherwise

### RT diagnostics output
- Keep `uov` in `outputs.variables` to write custom RT diagnostics.
- Control what is produced with `radiative-transfer-output`:
  - `enable_tau`
  - `enable_cell_flux`
  - `enable_toa_flux`

## 3) New drag + sponge controls

Use `drag:` block (all optional):

```yaml
drag:
  enabled: true
  apply_to: [v1, v2, v3]
  reference_wind: { v1: 0.0, v2: 0.0, v3: 0.0 }
  uniform:
    enabled: false
    tau: 0.0
  sponge_top:
    enabled: true
    thickness: 1.5e5
    n_layers: 6
    tau: 3.0e4
  sponge_bottom:
    enabled: true
    thickness: 5.0e4
    n_layers: 6
    tau: 3.0e4
  coeff_max: 0.0
```

Notes:
- The sponge ramp uses a smooth `sin^2` profile.
- If `thickness > 0`, sponge is altitude-based over that many meters from top/bottom boundary.
- If `thickness <= 0` (or omitted), sponge uses layer-count mode with `n_layers`.
- `n_layers` defaults to `6` if omitted.
- `tau` is damping timescale in seconds (`1/tau` is local drag coefficient).
- `uniform` and sponge coefficients add together.
- `coeff_max > 0` caps total drag coefficient.
- Set `drag.enabled: false` to turn everything off.

Layer-based example:

```yaml
drag:
  enabled: true
  sponge_top:
    enabled: true
    thickness: 0.0
    n_layers: 6
    tau: 3.0e4
  sponge_bottom:
    enabled: true
    thickness: 0.0
    n_layers: 6
    tau: 3.0e4
```

## 4) How many GPU cards should I use?

You must use a process count equal to world size implied by your layout and `nb*` values.

Examples:
- Local slab with `nb2: 4`, `nb3: 2` => world size = 8 (use 8 GPUs/processes)
- Cubed sphere with `nb2: 2`, `nb3: 2` => world size = `6*2*2 = 24`

Typical launch (single node):

```bash
torchrun --standalone --nproc_per_node=8 run_planet_rt.py -c planet_rt_local_dry_h2s_ch4.yaml
```

If `nproc_per_node` does not match world size, initialization will fail.

## 5) CPU runs

Use `distribute.backend: gloo` in YAML and launch with `torchrun` process count = world size.

Example (8 CPU processes):

```bash
torchrun --standalone --nproc_per_node=8 run_planet_rt.py -c planet_rt_local_dry_h2s_ch4.yaml
```

How many CPU cores to request from scheduler:
- Usually at least one core per process.
- If you launch 8 processes, request >=8 cores (often more for I/O and system overhead).
- You typically control CPU allocation in your cluster job script, not in this YAML.

## 6) Common checklist when switching cases

1. Set `geometry.type` and `distribute.layout`
2. Set `nb*` so world size matches your launch process count
3. Set `radiative-transfer.zenith_mode` (`nadir` or `auto`/`orbital`)
4. Choose dry vs moist (`chemistry.enable_kinetics` and presence of `reactions`)
5. Set drag/sponge (`drag.enabled`, sponge pressure ranges, `tau`)
6. Ensure `outputs` includes `uov` if RT diagnostics are needed
7. Check `opacities.*.species` matches your `species` list

## 7) Performance tips

- Dry case: disable kinetics and remove `reactions`/condensate species if not needed.
- Increase `radiative-transfer.update_dt` for cheaper RT updates.
- Disable `enable_tau` and `enable_cell_flux` if you only need TOA diagnostics.
- Keep grid and process decomposition balanced to avoid communication bottlenecks.
