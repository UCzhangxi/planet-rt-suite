# AGENT_CONTEXT

Purpose: bootstrap future coding assistants quickly for this repository.

## Project Goal

`planet-rt-suite` provides a YAML-driven planetary atmosphere workflow that supports:
- Cubed-sphere global setups and local Cartesian box setups
- Dry-like runs (non-reactive/non-condensing tracers) and moist/reactive runs
- Grey shortwave/longwave RT via in-script `GreyOpacity` + pyharp Toon solver
- Optional damping controls (uniform linear drag + top/bottom sponge layers)

Core runner:
- `core/run_planet_rt.py`

Primary user guide:
- `docs/PLANET_RT_USER_GUIDE.md`

## Key Design Decisions

1. One generalized runner
- Geometry, chemistry mode, RT zenith mode, and damping are controlled by YAML.
- No separate code paths needed for local vs cubed-sphere users.

2. Chemistry toggle logic
- `chemistry.enable_kinetics: true/false` is the explicit switch.
- If omitted, runner infers from presence of `reactions`.

3. RT zenith logic
- `radiative-transfer.zenith_mode`:
  - `nadir` for local direct overhead beam
  - `orbital` for global day-night/orbital forcing
  - `auto` picks orbital for cubed-sphere, nadir otherwise

4. Opacity `data` field
- In current grey JIT path, `opacities.*.data` is **compatibility placeholder only**.
- No `.pt` file is loaded by `run_planet_rt.py` for this mode.
- Templates intentionally use placeholder names:
  - `placeholder_not_used_grey_sw.pt`
  - `placeholder_not_used_grey_lw.pt`

5. Damping/drag implementation
- Pressure-based sponge ramps use smooth `sin^2` profile (inspired by canoe sponge logic).
- Supports additive combination of:
  - uniform Rayleigh drag (`uniform.tau`)
  - top sponge (`sponge_top`)
  - bottom sponge (`sponge_bottom`)
- Drag applies to selected velocity components with optional nonzero reference wind.

## Template Set (current)

- `templates/planet_rt_local_dry_h2s_ch4.yaml`
- `templates/planet_rt_local_moist_h2s_ch4.yaml`
- `templates/planet_rt_cubesphere_dry_h2s_ch4.yaml`
- `templates/planet_rt_cubesphere_moist_h2s_ch4.yaml`

Intent:
- Local vs cubed-sphere
- Dry vs moist
- Same high-level knob structure for easier switching

## Important Conventions

1. Output custom diagnostics
- Keep `uov` in outputs if RT custom diagnostics are required.

2. TOA diagnostics key
- Use `radiative-transfer-output.enable_toa_flux` (not older interface-flux key names).

3. Decomposition/world-size constraints
- `slab`: world size = `nb2 * nb3`
- `cubed`: world size = `nb1 * nb2 * nb3`
- `cubed-sphere`: world size = `6 * nb2 * nb3`

4. Dry speed optimization
- For dry/non-reactive runs:
  - set `chemistry.enable_kinetics: false`
  - avoid `reactions` if not needed
  - keep only needed species/outputs

## Known Practical Pitfalls

1. GitHub auth mismatch
- `gh auth status` may show a different account than expected; verify before repo operations.

2. Git push may fail after token switch
- Run `gh auth setup-git` when GitHub token/account changes.

3. Placeholder opacity confusion
- `.pt` filenames in YAML are placeholders for current grey mode.

## Recommended New-Chat Bootstrap Prompt

In a future chat, paste:

"Use `core/run_planet_rt.py` and `docs/PLANET_RT_USER_GUIDE.md` + `docs/AGENT_CONTEXT.md` as source of truth. Keep YAML-driven design. Preserve current dry/moist + local/cubed-sphere workflow and drag/sponge behavior."

## If extending the model later

Potential next upgrades:
- Add optional explicit loading path for real precomputed opacity tables
- Add named preset blocks for common planets (Uranus/Jupiter/Neptune)
- Add minimal smoke-test script for validating output variables and run initialization
