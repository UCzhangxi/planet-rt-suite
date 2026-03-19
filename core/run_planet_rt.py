#!/usr/bin/env python3
"""Run a planetary GCM with pyharp Toon radiative transfer."""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import yaml
import snapy
from snapy import MeshBlock, MeshBlockOptions, kIDN, kIV1, kICY, kIPR, kConserved
from kintera import Kinetics, KineticsOptions, ThermoX
from paddle import evolve_kinetics, setup_profile

import pyharp

SECONDS_PER_DAY = 86400.0
torch.set_default_dtype(torch.float64)

@dataclass
class RadiativeTransferConfig:
    update_dt: float
    sw_surface_albedo: float
    lw_surface_albedo: float
    stellar_flux_nadir: float
    zenith_mode: str
    obliquity_rad: float
    true_anomaly0_rad: float
    orbital_period: float
    rotation_period: float
    subsolar_lon0_rad: float


@dataclass
class BottomRelaxationConfig:
    enabled: bool
    depth_levels: int
    tau_bottom: float
    tau_top: float
    target_temp_levels: torch.Tensor  # (depth_levels,)


@dataclass
class RadiativeTransferOutputConfig:
    enable_tau: bool
    enable_cell_flux: bool
    enable_toa_flux: bool


@dataclass
class DragConfig:
    enabled: bool
    apply_v1: bool
    apply_v2: bool
    apply_v3: bool
    vref1: float
    vref2: float
    vref3: float
    uniform_enabled: bool
    uniform_tau: float
    top_enabled: bool
    top_tau: float
    top_thickness: float
    top_n_layers: int
    bottom_enabled: bool
    bottom_tau: float
    bottom_thickness: float
    bottom_n_layers: int
    z_levels: Optional[torch.Tensor]  # (nlyr,)
    coeff_max: float


@dataclass
class RadiativeTransferState:
    cfg: RadiativeTransferConfig

    grey_sw: torch.nn.Module
    grey_lw: torch.nn.Module
    toon_sw: torch.nn.Module
    toon_lw: torch.nn.Module

    lon: Optional[torch.Tensor]  # (ny, nx)
    lat: Optional[torch.Tensor]  # (ny, nx)
    dz: torch.Tensor  # (nlyr,)
    area1: torch.Tensor  # (ncol, nlyr + 1)
    vol: torch.Tensor  # (ncol, nlyr)
    il: int
    iu: int
    last_heating: torch.Tensor  # (ny, nx, nlyr), W/m^3 == Pa/s
    bottom_relax: BottomRelaxationConfig
    drag: DragConfig
    output_cfg: RadiativeTransferOutputConfig
    next_update_time: float
    diagnostic_time: float


class GreyOpacity(torch.nn.Module):
    def __init__(
        self,
        species_weights: list[float],
        kappa_a: float,
        kappa_b: float,
        kappa_cut: float,
        w0: float = 0.0,
        g: float = 0.0,
        nwave: int = 1,
        nmom: int = 1,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "species_weights",
            torch.tensor(species_weights),
            persistent=True,
        )
        self.kappa_a = float(kappa_a)
        self.kappa_b = float(kappa_b)
        self.kappa_cut = float(kappa_cut)
        self.nwave = int(nwave)
        self.w0 = float(w0)
        self.g = float(g)
        self.nprop = 2 + int(nmom)  # (extinction, w0, g)

    def forward(self, conc: torch.Tensor, pres: torch.Tensor, temp: torch.Tensor) -> torch.Tensor:
        '''
        Compute grey opacity properties for each layer and column.

        Args:
            conc: (ncol, nlyr, nspecies) [mol/m^3]
            pres: (ncol, nlyr) [pa]
            temp: (ncol, nlyr) [K]

        Returns:
            prop: (nwave, ncol, nlyr, nprop) where nprop includes extinction [1/m],
            single scattering albedo, and asymmetry factor.
        '''

        ncol = conc.shape[0]
        nlyr = conc.shape[1]

        # extinction = rho * kappa(pres)
        rho = (conc * self.species_weights.view(1, 1, -1)).sum(dim=-1)

        kappa = self.kappa_a * torch.pow(pres, self.kappa_b)
        kappa = torch.clamp(kappa, min=self.kappa_cut)
        extinction = rho * kappa  # [1/m]

        out = torch.zeros(
            (self.nwave, ncol, nlyr, self.nprop),
            dtype=conc.dtype,
            device=conc.device,
        )

        out[..., 0] = extinction.unsqueeze(0)
        out[..., 1] = self.w0
        out[..., 2] = self.g

        return out


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_device(block: MeshBlock) -> torch.device:
    backend = str(block.options.layout().backend()).lower()
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA, but no GPU is available.")

        ngpu = torch.cuda.device_count()
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
        else:
            local_rank = int(snapy.distributed.get_local_rank())

        if ngpu <= 0:
            raise RuntimeError("NCCL backend requested but torch reports zero CUDA devices.")

        if local_rank < 0 or local_rank >= ngpu:
            local_rank = local_rank % ngpu

        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    return torch.device("cpu")


def create_models_general(config_file: str, config: dict[str, Any], output_dir: str | None = None):
    op = MeshBlockOptions.from_yaml(config_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        op.output_dir(output_dir)

    block = MeshBlock(op)
    device = select_device(block)
    block.to(device)

    thermo_y = block.module("hydro.eos.thermo")
    thermo_x = ThermoX(thermo_y.options)
    thermo_x.to(device)

    chemistry_cfg = config.get("chemistry", {})
    enable_kinetics_override = chemistry_cfg.get("enable_kinetics", None)
    if enable_kinetics_override is None:
        enable_kinetics = bool(config.get("reactions", []))
    else:
        enable_kinetics = bool(enable_kinetics_override)

    kinet: Optional[Kinetics] = None
    if enable_kinetics:
        op_kin = KineticsOptions.from_yaml(config_file)
        kinet = Kinetics(op_kin)
        kinet.to(device)

    eos = block.module("hydro.eos")
    return block, eos, thermo_y, thermo_x, kinet, device, enable_kinetics


def initialize_atm(block: MeshBlock, config: dict[str, Any]) -> tuple[dict[str, torch.Tensor], float]:
    grav = -float(config["forcing"]["const-gravity"]["grav1"])
    problem = config["problem"]

    param: dict[str, float] = {
        "Ts": float(problem["Ts"]),
        "Ps": float(problem["Ps"]),
        "Tmin": float(problem.get("Tmin", problem["Ts"])),
        "grav": grav,
    }

    thermo_y = block.module("hydro.eos.thermo")
    for name in thermo_y.options.species():
        param[f"x{name}"] = float(problem.get(f"x{name}", 0.0))

    hydro_w = setup_profile(block, param, method="pseudo-adiabat")

    # add random noise to IV1
    hydro_w[kIV1] += 1e-6 * torch.randn_like(hydro_w[kIV1])

    return block.initialize({"hydro_w": hydro_w})


def _resolve_local_face_name(block: MeshBlock) -> str:
    if hasattr(block, "get_layout"):
        layout = block.get_layout()
    else:
        layout = snapy.distributed.get_layout(block)

    rank = int(layout.options.rank())
    loc = layout.loc_of(rank)
    face_id = int(loc[2])
    return snapy.coord.get_cs_face_name(face_id)


def _build_local_lonlat(block: MeshBlock) -> tuple[torch.Tensor, torch.Tensor]:
    coord = block.module("coord")
    x2v = coord.buffer("x2v")
    x3v = coord.buffer("x3v")

    beta, alpha = torch.meshgrid(x3v, x2v, indexing="ij")
    face_name = _resolve_local_face_name(block)
    lon, lat = snapy.coord.cs_ab_to_lonlat(face_name, alpha, beta)
    return lon, lat


def _compute_cos_zenith_dayside(
    lon: torch.Tensor,
    lat: torch.Tensor,
    rt_cfg: RadiativeTransferConfig,
    current_time: float,
) -> torch.Tensor:
    true_anomaly = rt_cfg.true_anomaly0_rad + 2.0 * math.pi * current_time / rt_cfg.orbital_period
    subsolar_lon = rt_cfg.subsolar_lon0_rad + 2.0 * math.pi * current_time / rt_cfg.rotation_period
    declination = math.asin(math.sin(rt_cfg.obliquity_rad) * math.sin(true_anomaly))

    cos_zenith = (
        torch.sin(lat) * math.sin(declination)
        + torch.cos(lat) * math.cos(declination) * torch.cos(lon - subsolar_lon)
    )
    return torch.clamp(cos_zenith, min=0.0)


def _extract_species_weights_from_config(config: dict[str, Any]) -> list[float]:
    # Atomic masses [kg/mol] for elements used in current RT species definitions.
    atomic_mass = {
        "H": 1.00784e-3,
        "He": 4.002602e-3,
        "C": 12.0107e-3,
        "N": 14.0067e-3,
        "O": 15.999e-3,
        "S": 32.065e-3,
    }
    out: list[float] = []
    for sp in config["species"]:
        mw = 0.0
        for el, stoich in sp["composition"].items():
            if el not in atomic_mass:
                raise KeyError(f"Unsupported element '{el}' in species '{sp['name']}' for JIT opacity.")
            mw += float(stoich) * atomic_mass[el]
        out.append(mw)
    return out

def _parse_band_range(config: dict[str, Any], key: str) -> Tuple[float, float]:
    for band in config["bands"]:
        if band["name"] == key:
            return band["range"]
    raise ValueError(f"Band '{key}' not found in config 'bands' section.")


def _build_drag_config(config: dict[str, Any]) -> DragConfig:
    drag_raw = config.get("drag", {})
    apply_to = drag_raw.get("apply_to", ["v1", "v2", "v3"])
    apply_set = {str(item).lower() for item in apply_to}

    ref_raw = drag_raw.get("reference_wind", {})
    uniform_raw = drag_raw.get("uniform", {})
    top_raw = drag_raw.get("sponge_top", {})
    bot_raw = drag_raw.get("sponge_bottom", {})

    return DragConfig(
        enabled=bool(drag_raw.get("enabled", False)),
        apply_v1=("v1" in apply_set),
        apply_v2=("v2" in apply_set),
        apply_v3=("v3" in apply_set),
        vref1=float(ref_raw.get("v1", 0.0)),
        vref2=float(ref_raw.get("v2", 0.0)),
        vref3=float(ref_raw.get("v3", 0.0)),
        uniform_enabled=bool(uniform_raw.get("enabled", False)),
        uniform_tau=max(float(uniform_raw.get("tau", 0.0)), 0.0),
        top_enabled=bool(top_raw.get("enabled", False)),
        top_tau=max(float(top_raw.get("tau", 0.0)), 0.0),
        top_thickness=max(float(top_raw.get("thickness", 0.0)), 0.0),
        top_n_layers=max(int(top_raw.get("n_layers", 6)), 0),
        bottom_enabled=bool(bot_raw.get("enabled", False)),
        bottom_tau=max(float(bot_raw.get("tau", 0.0)), 0.0),
        bottom_thickness=max(float(bot_raw.get("thickness", 0.0)), 0.0),
        bottom_n_layers=max(int(bot_raw.get("n_layers", 6)), 0),
        z_levels=None,
        coeff_max=max(float(drag_raw.get("coeff_max", 0.0)), 0.0),
    )


def create_grey_opacities(config: dict[str, Any]) -> Tuple[torch.nn.Module,
                                                           torch.nn.Module]:
    opacities = config.get("opacities", {})
    species_weights = _extract_species_weights_from_config(config)

    opacity_models = []
    for name, spec in opacities.items():
        # For the in-script GreyOpacity model, `data` is only a compatibility
        # placeholder in YAML and is not loaded.
        _ = spec.get("data", [])

        nmom = int(spec.get("nmom", 1))
        params = spec.get("parameters", {})
        model = GreyOpacity(
            species_weights=species_weights,
            kappa_a=float(params["kappa_a"]),
            kappa_b=float(params["kappa_b"]),
            kappa_cut=float(params["kappa_cut"]),
            w0=float(params.get("w0", 0.0)),
            g=float(params.get("g", 0.0)),
            nwave=1,
            nmom=nmom,
        )
        #scripted = torch.jit.script(model)
        #scripted.save(str(out_file))
        #print(f"Saved opacity for '{name}' -> {out_file}")
        #opacity_models.append(torch.compile(model))
        opacity_models.append(model)
    return opacity_models

def create_toon_solvers(config: dict[str, Any]) -> Tuple[torch.nn.Module,
                                                         torch.nn.Module]:
    # shortwave solver
    op_sw = pyharp.ToonMcKay89Options()
    toon_sw = pyharp.ToonMcKay89(op_sw)

    # longwave solver
    op_lw = pyharp.ToonMcKay89Options()
    wave_lo, wave_hi = _parse_band_range(config, "lw")
    op_lw.wave_lower([wave_lo])
    op_lw.wave_upper([wave_hi])
    toon_lw = pyharp.ToonMcKay89(op_lw)

    return toon_sw, toon_lw

def build_rt_state(
    block: MeshBlock,
    eos,
    block_vars: dict[str, torch.Tensor],
    config: dict[str, Any],
    config_file: str,
) -> RadiativeTransferState:
    rt_cfg_raw = config.get("radiative-transfer", {})
    problem = config.get("problem", {})

    geom_type = str(config.get("geometry", {}).get("type", ""))
    layout_type = str(config.get("distribute", {}).get("layout", "slab"))
    is_cubed_sphere = (geom_type == "gnomonic-equiangle") or (layout_type == "cubed-sphere")

    zenith_mode_raw = str(rt_cfg_raw.get("zenith_mode", "auto")).lower()
    if zenith_mode_raw == "auto":
        zenith_mode = "orbital" if is_cubed_sphere else "nadir"
    elif zenith_mode_raw in ("orbital", "nadir"):
        zenith_mode = zenith_mode_raw
    else:
        raise ValueError("radiative-transfer.zenith_mode must be one of: auto, orbital, nadir")

    cfg = RadiativeTransferConfig(
        update_dt=float(rt_cfg_raw.get("update_dt", 3600.0)),
        sw_surface_albedo=float(rt_cfg_raw.get("sw_surface_albedo", 0.0)),
        lw_surface_albedo=float(rt_cfg_raw.get("lw_surface_albedo", 0.0)),
        stellar_flux_nadir=float(rt_cfg_raw.get("stellar_flux_nadir", 3.71)),
        zenith_mode=zenith_mode,
        obliquity_rad=math.radians(float(problem.get("obliquity_deg", 97.77))),
        true_anomaly0_rad=math.radians(float(problem.get("true_anomaly_deg", 0.0))),
        orbital_period=float(problem.get("orbital_period", 2.651e9)),
        rotation_period=float(problem.get("rotation_period", 62064.0)),
        subsolar_lon0_rad=math.radians(float(problem.get("subsolar_lon0_deg", 0.0))),
    )

    coord = block.module("coord")
    il, iu = coord.il(), coord.iu()
    nlyr = iu - il + 1
    ny = int(coord.buffer("x3v").numel())
    nx = int(coord.buffer("x2v").numel())
    ncol = ny * nx

    dz = coord.buffer("dx1f")[il : iu + 1]

    area1 = coord.face_area1()[..., il : iu + 2].view(ncol, nlyr + 1)
    vol = coord.cell_volume()[..., il : iu + 1].view(ncol, nlyr)
    area1 = area1.view(ncol, nlyr + 1)
    vol = vol.view(ncol, nlyr)

    device = select_device(block)

    lon: Optional[torch.Tensor] = None
    lat: Optional[torch.Tensor] = None
    if cfg.zenith_mode == "orbital":
        lon_t, lat_t = _build_local_lonlat(block)
        lon = lon_t.to(device)
        lat = lat_t.to(device)

    last_heating = torch.zeros((ny, nx, nlyr), device=device)

    relax_cfg_raw = config.get("bottom-relaxation", {})
    relax_enabled = bool(relax_cfg_raw.get("enabled", False))
    relax_depth = max(1, int(relax_cfg_raw.get("depth_levels", 1)))

    hydro_w = block_vars["hydro_w"]
    temp = eos.compute("W->T", (hydro_w,))
    il = int(il)
    relax_depth = min(relax_depth, nlyr)

    if "target_temp" in relax_cfg_raw:
        target_scalar = float(relax_cfg_raw["target_temp"])
        target_levels = torch.full(
            (relax_depth,),
            target_scalar,
            dtype=temp.dtype,
            device=temp.device,
        )
    else:
        target_levels = temp[..., il : il + relax_depth].mean(dim=(0, 1)).detach().clone()

    bottom_relax = BottomRelaxationConfig(
        enabled=relax_enabled,
        depth_levels=relax_depth,
        tau_bottom=max(float(relax_cfg_raw.get("tau_bottom", 1.0e6)), 1.0e-12),
        tau_top=max(float(relax_cfg_raw.get("tau_top", relax_cfg_raw.get("tau_bottom", 1.0e6))), 1.0e-12),
        target_temp_levels=target_levels,
    )
    drag_cfg = _build_drag_config(config)
    drag_cfg.z_levels = coord.buffer("x1v")[il : iu + 1].to(device)

    output_cfg_raw = config.get("radiative-transfer-output", {})
    output_cfg = RadiativeTransferOutputConfig(
        enable_tau=bool(output_cfg_raw.get("enable_tau", True)),
        enable_cell_flux=bool(output_cfg_raw.get("enable_cell_flux", True)),
        enable_toa_flux=bool(output_cfg_raw.get("enable_toa_flux", True)),
    )

    grey_sw, grey_lw = create_grey_opacities(config)
    toon_sw, toon_lw = create_toon_solvers(config)

    grey_sw.to(device)
    grey_lw.to(device)

    toon_sw.to(device)
    toon_lw.to(device)

    return RadiativeTransferState(
        cfg=cfg,
        grey_sw=grey_sw,
        grey_lw=grey_lw,
        toon_sw=toon_sw,
        toon_lw=toon_lw,
        lon=lon,
        lat=lat,
        dz=dz,
        area1=area1,
        vol=vol,
        il=il,
        iu=iu,
        last_heating=last_heating,
        bottom_relax=bottom_relax,
        drag=drag_cfg,
        output_cfg=output_cfg,
        next_update_time=0.0,
        diagnostic_time=0.0,
    )


def _compute_cos_zenith_field(rt_state: RadiativeTransferState, current_time: float) -> torch.Tensor:
    if rt_state.cfg.zenith_mode == "nadir":
        ny, nx, _ = rt_state.last_heating.shape
        return torch.ones((ny, nx), dtype=rt_state.last_heating.dtype, device=rt_state.last_heating.device)

    if rt_state.lon is None or rt_state.lat is None:
        raise RuntimeError("Orbital zenith mode requires lon/lat fields")

    return _compute_cos_zenith_dayside(
        lon=rt_state.lon,
        lat=rt_state.lat,
        rt_cfg=rt_state.cfg,
        current_time=current_time,
    )

def _compute_shortwave_rt(rt_state: RadiativeTransferState, 
                          cos_zenith_dayside: torch.Tensor,
                          conc_i: torch.Tensor,
                          pres_i: torch.Tensor,
                          temp_i: torch.Tensor) -> torch.Tensor:
    ncol = conc_i.shape[0]

    bc: dict[str, torch.Tensor] = {
        f"fbeam": (
            rt_state.cfg.stellar_flux_nadir
            * torch.ones((1, ncol), device=conc_i.device) # (nwave, ncol)
        ),
        f"umu0": cos_zenith_dayside.view(ncol), # (ncol,)
        f"albedo": (
            rt_state.cfg.sw_surface_albedo
            * torch.ones((1, ncol), device=conc_i.device) # (nwave, ncol)
        ),
    }

    # (ncol, nlyr, nspecies) -> (nwave, ncol, nlyr, nprop)
    prop = rt_state.grey_sw(conc_i, pres_i, temp_i)

    # extinction [1/m] -> optical thickness [unitless]
    prop *= rt_state.dz.view(1, 1, -1, 1)

    result = rt_state.toon_sw(prop, **bc).sum(0)  # (ncol, nlyr+1, 2)

    # set net flux to zero in layers with zero cos(zenith) to avoid numerical issues
    # with Toon solver output
    zero_cosz_mask = (bc["umu0"] == 0.0).view(ncol, 1, 1).expand_as(result)
    result[zero_cosz_mask] = 0.0

    # net flux = upward - downward
    return result[...,0] - result[...,1]

def _compute_longwave_rt(rt_state: RadiativeTransferState,
                         conc_i: torch.Tensor,
                         pres_i: torch.Tensor,
                         temp_i: torch.Tensor) -> None:
    ncol, nlyr, _ = conc_i.shape

    bc: dict[str, torch.Tensor] = {
        f"albedo": (
            rt_state.cfg.lw_surface_albedo
            * torch.ones((1, ncol), device=conc_i.device) # (nwave, ncol)
        ),
    }

    # (ncol, nlyr, nspecies) -> (nwave, ncol, nlyr, nprop)
    prop = rt_state.grey_lw(conc_i, pres_i, temp_i)

    # extinction [1/m] -> optical thickness [unitless]
    prop *= rt_state.dz.view(1, 1, -1, 1)

    # temperature at layer interfaces for thermal emission
    temf = torch.zeros((ncol, nlyr + 1), device=conc_i.device)

    temf[:, 0] = 2 * temp_i[:, 0] - temp_i[:, 1]  # extrapolate bottom interface
    temf[:, 1:-1] = 0.5 * (temp_i[:, :-1] + temp_i[:, 1:]) # average for interior
    temf[:, -1] = 2 * temp_i[:, -1] - temp_i[:, -2]  # extrapolate top interface

    result = rt_state.toon_lw(prop, temf=temf, **bc).sum(0)  # (ncol, nlyr+1, 2)

    # net flux = upward - downward
    return result[...,0] - result[...,1]


def _compute_rt_output_fields(
    eos,
    thermo_y,
    thermo_x: ThermoX,
    block_vars: dict[str, torch.Tensor],
    rt_state: RadiativeTransferState,
) -> dict[str, torch.Tensor]:
    hydro_w = block_vars["hydro_w"]

    temp = eos.compute("W->T", (hydro_w,))
    pres = hydro_w[kIPR]
    xfrac = thermo_y.compute("Y->X", (hydro_w[kICY:],))
    conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))

    il, iu = rt_state.il, rt_state.iu
    nlyr = iu - il + 1
    ny, nx = hydro_w[kIPR].shape[:2]
    ncol = ny * nx

    temp_i = temp[..., il : iu + 1].view(ncol, nlyr)
    pres_i = pres[..., il : iu + 1].view(ncol, nlyr)
    conc_i = conc[..., il : iu + 1, :].view(ncol, nlyr, conc.shape[-1])

    cos_zenith_dayside = _compute_cos_zenith_field(rt_state, rt_state.diagnostic_time)

    ncol = conc_i.shape[0]

    bc_sw: dict[str, torch.Tensor] = {
        "fbeam": (
            rt_state.cfg.stellar_flux_nadir
            * torch.ones((1, ncol), device=conc_i.device)
        ),
        "umu0": cos_zenith_dayside.view(ncol),
        "albedo": (
            rt_state.cfg.sw_surface_albedo
            * torch.ones((1, ncol), device=conc_i.device)
        ),
    }
    prop_sw = rt_state.grey_sw(conc_i, pres_i, temp_i)
    prop_sw *= rt_state.dz.view(1, 1, -1, 1)
    tau_sw_layer = prop_sw[0, :, :, 0]
    result_sw = rt_state.toon_sw(prop_sw, **bc_sw).sum(0)
    zero_cosz_mask = (bc_sw["umu0"] == 0.0).view(ncol, 1, 1).expand_as(result_sw)
    result_sw[zero_cosz_mask] = 0.0
    sw_up = result_sw[..., 0]
    sw_dn = result_sw[..., 1]

    bc_lw: dict[str, torch.Tensor] = {
        "albedo": (
            rt_state.cfg.lw_surface_albedo
            * torch.ones((1, ncol), device=conc_i.device)
        ),
    }
    prop_lw = rt_state.grey_lw(conc_i, pres_i, temp_i)
    prop_lw *= rt_state.dz.view(1, 1, -1, 1)
    tau_lw_layer = prop_lw[0, :, :, 0]

    temf = torch.zeros((ncol, nlyr + 1), device=conc_i.device)
    temf[:, 0] = 2 * temp_i[:, 0] - temp_i[:, 1]
    temf[:, 1:-1] = 0.5 * (temp_i[:, :-1] + temp_i[:, 1:])
    temf[:, -1] = 2 * temp_i[:, -1] - temp_i[:, -2]

    result_lw = rt_state.toon_lw(prop_lw, temf=temf, **bc_lw).sum(0)
    lw_up = result_lw[..., 0]
    lw_dn = result_lw[..., 1]

    sw_up_cell = 0.5 * (sw_up[:, :-1] + sw_up[:, 1:])
    sw_dn_cell = 0.5 * (sw_dn[:, :-1] + sw_dn[:, 1:])
    lw_up_cell = 0.5 * (lw_up[:, :-1] + lw_up[:, 1:])
    lw_dn_cell = 0.5 * (lw_dn[:, :-1] + lw_dn[:, 1:])

    sw_up_toa = sw_up[:, -1]
    sw_dn_toa = sw_dn[:, -1]
    lw_up_toa = lw_up[:, -1]
    lw_dn_toa = lw_dn[:, -1]

    shape_full = hydro_w[kIPR].shape
    out: dict[str, torch.Tensor] = {}
    if rt_state.output_cfg.enable_tau:
        tau_sw_layer_full = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)
        tau_lw_layer_full = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)

        tau_sw_cum_top = torch.flip(
            torch.cumsum(torch.flip(tau_sw_layer, dims=[1]), dim=1), dims=[1]
        )
        tau_lw_cum_top = torch.flip(
            torch.cumsum(torch.flip(tau_lw_layer, dims=[1]), dim=1), dims=[1]
        )

        tau_sw_full = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)
        tau_lw_full = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)

        tau_sw_layer_full[..., il : iu + 1] = tau_sw_layer.view(ny, nx, nlyr)
        tau_lw_layer_full[..., il : iu + 1] = tau_lw_layer.view(ny, nx, nlyr)
        tau_sw_full[..., il : iu + 1] = tau_sw_cum_top.view(ny, nx, nlyr)
        tau_lw_full[..., il : iu + 1] = tau_lw_cum_top.view(ny, nx, nlyr)

        out["rt_tau_sw"] = tau_sw_full
        out["rt_tau_lw"] = tau_lw_full
        out["rt_tau_sw_layer"] = tau_sw_layer_full
        out["rt_tau_lw_layer"] = tau_lw_layer_full

    if rt_state.output_cfg.enable_cell_flux:
        flx_sw_up = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)
        flx_sw_dn = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)
        flx_lw_up = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)
        flx_lw_dn = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)
        flx_sw_up[..., il : iu + 1] = sw_up_cell.view(ny, nx, nlyr)
        flx_sw_dn[..., il : iu + 1] = sw_dn_cell.view(ny, nx, nlyr)
        flx_lw_up[..., il : iu + 1] = lw_up_cell.view(ny, nx, nlyr)
        flx_lw_dn[..., il : iu + 1] = lw_dn_cell.view(ny, nx, nlyr)
        out["rt_flux_sw_up"] = flx_sw_up
        out["rt_flux_sw_dn"] = flx_sw_dn
        out["rt_flux_lw_up"] = flx_lw_up
        out["rt_flux_lw_dn"] = flx_lw_dn

    if rt_state.output_cfg.enable_toa_flux:
        flx_sw_up_toa = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)
        flx_sw_dn_toa = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)
        flx_lw_up_toa = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)
        flx_lw_dn_toa = torch.zeros(shape_full, dtype=hydro_w.dtype, device=hydro_w.device)

        flx_sw_up_toa[..., iu] = sw_up_toa.view(ny, nx)
        flx_sw_dn_toa[..., iu] = sw_dn_toa.view(ny, nx)
        flx_lw_up_toa[..., iu] = lw_up_toa.view(ny, nx)
        flx_lw_dn_toa[..., iu] = lw_dn_toa.view(ny, nx)

        out["rt_flux_sw_up_toa"] = flx_sw_up_toa
        out["rt_flux_sw_dn_toa"] = flx_sw_dn_toa
        out["rt_flux_lw_up_toa"] = flx_lw_up_toa
        out["rt_flux_lw_dn_toa"] = flx_lw_dn_toa

    return out

def _compute_rt_heating(
    block: MeshBlock,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    block_vars: dict[str, torch.Tensor],
    current_time: float,
    rt_state: RadiativeTransferState,
) -> torch.Tensor:
    hydro_w = block_vars["hydro_w"]

    temp = eos.compute("W->T", (hydro_w,))
    pres = hydro_w[kIPR]
    xfrac = thermo_y.compute("Y->X", (hydro_w[kICY:],))
    conc = thermo_x.compute("TPX->V", (temp, pres, xfrac))

    il, iu = rt_state.il, rt_state.iu
    nlyr = iu - il + 1
    ny, nx = hydro_w[kIPR].shape[:2]
    ncol = ny * nx

    temp_i = temp[..., il : iu + 1].view(ncol, nlyr)
    pres_i = pres[..., il : iu + 1].view(ncol, nlyr)
    conc_i = conc[..., il : iu + 1, :].view(ncol, nlyr, conc.shape[-1])
    
    cos_zenith_dayside = _compute_cos_zenith_field(rt_state, current_time)

    net_flux_sw = _compute_shortwave_rt(rt_state, cos_zenith_dayside, conc_i, pres_i, temp_i)
    #print('net_flux_sw shape:', net_flux_sw.shape, 'max:', net_flux_sw.max().item(),
    #      'min:', net_flux_sw.min().item())

    net_flux_lw = _compute_longwave_rt(rt_state, conc_i, pres_i, temp_i)
    #print('net_flux_lw shape:', net_flux_lw.shape, 'max:', net_flux_lw.max().item(),
    #      'min:', net_flux_lw.min().item())

    net_flux = net_flux_sw + net_flux_lw # (ncol, nlyr+1)

    # Volumetric heating [W/m^3] = -div(F) on a spherical shell using face areas.
    div_f = (
        rt_state.area1[:, 1:] * net_flux[:, 1:] - rt_state.area1[:, :-1] * net_flux[:, :-1]
    ) / rt_state.vol
    heating = -div_f

    # (ncol, nlyr) -> (ny, nx, nlyr)
    return heating.view(ny, nx, nlyr)


def update_rt_tendency_if_needed(
    block: MeshBlock,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    block_vars: dict[str, torch.Tensor],
    current_time: float,
    rt_state: RadiativeTransferState,
) -> None:
    if current_time + 1.0e-12 < rt_state.next_update_time:
        return

    rt_state.last_heating = _compute_rt_heating(
        block,
        eos,
        thermo_y,
        thermo_x,
        block_vars,
        current_time,
        rt_state,
    )
    #print('rt_heating shape:', rt_state.last_heating.shape, 'max:',
    #      rt_state.last_heating.max().item(), 'min:',
    #      rt_state.last_heating.min().item())
    rt_state.next_update_time = current_time + max(rt_state.cfg.update_dt, 0.0)

    print(
        "RT update:",
        f"t={current_time / SECONDS_PER_DAY:.3f} d,",
        f"heating[min,max]=({rt_state.last_heating.min().item():.3e}, {rt_state.last_heating.max().item():.3e}) Pa/s",
    )


def _apply_bottom_temp_relaxation(
    eos,
    block_vars: dict[str, torch.Tensor],
    rt_state: RadiativeTransferState,
    dt: float,
) -> None:
    relax = rt_state.bottom_relax
    if not relax.enabled:
        return

    hydro_w = block_vars["hydro_w"]
    hydro_u = block_vars["hydro_u"]
    temp = eos.compute("W->T", (hydro_w,))
    pres = hydro_w[kIPR]

    il = rt_state.il
    depth = relax.depth_levels

    if depth <= 1:
        tau_levels = torch.tensor([relax.tau_bottom], dtype=temp.dtype, device=temp.device)
    else:
        tau_levels = torch.linspace(relax.tau_bottom, relax.tau_top, depth, dtype=temp.dtype, device=temp.device)

    target = relax.target_temp_levels.to(temp.device)
    local_temp = temp[..., il : il + depth]
    local_pres = pres[..., il : il + depth]

    dTdt = (target.view(1, 1, depth) - local_temp) / tau_levels.view(1, 1, depth)
    dpdt = (local_pres / torch.clamp(local_temp, min=1.0)) * dTdt
    hydro_u[kIPR, ..., il : il + depth] += dpdt * dt


def _sine2_ramp(coord: torch.Tensor, start: float, end: float) -> torch.Tensor:
    span = float(end - start)
    if abs(span) < 1.0e-20:
        return torch.zeros_like(coord)

    eta = (coord - start) / span
    eta = torch.clamp(eta, min=0.0, max=1.0)
    return torch.sin(0.5 * math.pi * eta).pow(2)


def _build_layer_ramp(
    nlyr: int,
    n_layers: int,
    is_top: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    ramp = torch.zeros((nlyr,), dtype=dtype, device=device)
    n = max(min(int(n_layers), nlyr), 0)
    if n <= 0:
        return ramp

    eta = torch.linspace(0.0, 1.0, n, dtype=dtype, device=device)
    core = torch.sin(0.5 * math.pi * eta).pow(2)
    if is_top:
        ramp[nlyr - n :] = core
    else:
        ramp[:n] = torch.flip(core, dims=(0,))
    return ramp


def _build_sponge_ramp(
    local_pres: torch.Tensor,
    local_z: Optional[torch.Tensor],
    thickness: float,
    n_layers: int,
    is_top: bool,
) -> torch.Tensor:
    nlyr = int(local_pres.shape[-1])

    if (thickness > 0.0) and (local_z is not None):
        zmin = float(local_z[0].item())
        zmax = float(local_z[-1].item())
        if is_top:
            z_start = zmax - thickness
            z_end = zmax
        else:
            z_start = zmin + thickness
            z_end = zmin

        ramp_1d = _sine2_ramp(local_z, z_start, z_end)
        return ramp_1d.view(1, 1, nlyr).expand_as(local_pres)

    ramp_1d = _build_layer_ramp(
        nlyr=nlyr,
        n_layers=n_layers,
        is_top=is_top,
        dtype=local_pres.dtype,
        device=local_pres.device,
    )
    return ramp_1d.view(1, 1, nlyr).expand_as(local_pres)


def _apply_velocity_drag(
    block_vars: dict[str, torch.Tensor],
    rt_state: RadiativeTransferState,
    dt: float,
) -> None:
    cfg = rt_state.drag
    if not cfg.enabled:
        return

    hydro_w = block_vars["hydro_w"]
    hydro_u = block_vars["hydro_u"]
    pres = hydro_w[kIPR]
    rho = hydro_w[kIDN]

    il, iu = rt_state.il, rt_state.iu
    coeff = torch.zeros_like(pres)

    if cfg.uniform_enabled and cfg.uniform_tau > 0.0:
        coeff[..., il : iu + 1] += 1.0 / cfg.uniform_tau

    local_pres = pres[..., il : iu + 1]
    local_z = cfg.z_levels
    if local_z is not None:
        local_z = local_z.to(device=local_pres.device, dtype=local_pres.dtype)

    if cfg.top_enabled and cfg.top_tau > 0.0:
        ramp_top = _build_sponge_ramp(
            local_pres=local_pres,
            local_z=local_z,
            thickness=cfg.top_thickness,
            n_layers=cfg.top_n_layers,
            is_top=True,
        )
        coeff[..., il : iu + 1] += ramp_top / cfg.top_tau

    if cfg.bottom_enabled and cfg.bottom_tau > 0.0:
        ramp_bot = _build_sponge_ramp(
            local_pres=local_pres,
            local_z=local_z,
            thickness=cfg.bottom_thickness,
            n_layers=cfg.bottom_n_layers,
            is_top=False,
        )
        coeff[..., il : iu + 1] += ramp_bot / cfg.bottom_tau

    if cfg.coeff_max > 0.0:
        coeff = torch.clamp(coeff, max=cfg.coeff_max)

    vel_idx = [kIV1, kIV1 + 1, kIV1 + 2]
    apply_mask = [cfg.apply_v1, cfg.apply_v2, cfg.apply_v3]
    vref = [cfg.vref1, cfg.vref2, cfg.vref3]
    nvar = hydro_u.shape[0]

    for idx, do_apply, ref in zip(vel_idx, apply_mask, vref):
        if (not do_apply) or (idx >= nvar):
            continue

        vel = hydro_w[idx]
        tendency = -rho * (vel - ref) * coeff
        hydro_u[idx] += tendency * dt


def apply_rt_forcing(eos, block: MeshBlock, block_vars: dict[str, torch.Tensor], rt_state: RadiativeTransferState, dt: float) -> None:
    block_vars["hydro_u"][kIPR, ..., rt_state.il : rt_state.iu + 1] += rt_state.last_heating * dt
    block.apply_hydro_bc(block_vars["hydro_u"],type=kConserved)
    _apply_bottom_temp_relaxation(eos, block_vars, rt_state, dt)
    _apply_velocity_drag(block_vars, rt_state, dt)


def write_restart_manifest(
    checkpoint_dir: Path,
    checkpoint_day: int,
    current_time: float,
    config_file: str,
    output_dir: str,
    basename: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    restart_candidates = sorted(glob.glob(str(Path(output_dir) / f"{basename}.*.restart")))
    restart_file = restart_candidates[-1] if restart_candidates else None

    payload = {
        "checkpoint_day": checkpoint_day,
        "simulation_time_seconds": float(current_time),
        "simulation_time_days": float(current_time / SECONDS_PER_DAY),
        "config_file": str(Path(config_file).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "latest_restart_archive": restart_file,
        "resume_hint": {
            "command": (
                f"python {Path(__file__).name} "
                f"-c {config_file} --output-dir {output_dir} --restart-name "
                + (Path(restart_file).name if restart_file else "<restart-file-name>")
            )
        },
    }

    manifest_file = checkpoint_dir / f"checkpoint_day_{checkpoint_day:04d}.yaml"
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def run_simulation(
    block: MeshBlock,
    eos,
    thermo_y,
    thermo_x: ThermoX,
    kinet: Optional[Kinetics],
    block_vars: dict[str, torch.Tensor],
    current_time: float,
    tlim: float,
    rt_state: RadiativeTransferState,
    config_file: str,
    output_dir: str,
    basename: str,
) -> tuple[dict[str, torch.Tensor], float]:
    block.options.intg().tlim(tlim)

    next_checkpoint_day = int(current_time // (10.0 * SECONDS_PER_DAY)) * 10 + 10
    checkpoint_dir = Path(output_dir) / "restart_checkpoints"

    rt_state.diagnostic_time = current_time
    update_rt_tendency_if_needed(block, eos, thermo_y, thermo_x, block_vars, current_time, rt_state)
    block.make_outputs(block_vars, current_time)

    while not block.intg.stop(block.inc_cycle(), current_time):
        dt = block.max_time_step(block_vars)
        block.print_cycle_info(block_vars, current_time, dt)

        for stage in range(len(block.intg.stages)):
            block.forward(block_vars, dt, stage)
            update_rt_tendency_if_needed(block, eos, thermo_y, thermo_x, block_vars, current_time, rt_state)
            apply_rt_forcing(eos, block, block_vars, rt_state, dt)

        err = block.check_redo(block_vars)
        if err > 0:
            continue
        if err < 0:
            break

        if kinet is not None:
            del_rho = evolve_kinetics(block_vars["hydro_w"], eos, thermo_x, thermo_y, kinet, dt)
            block_vars["hydro_u"][kICY:] += del_rho

        current_time += dt
        rt_state.diagnostic_time = current_time
        block.make_outputs(block_vars, current_time)

        while current_time >= next_checkpoint_day * SECONDS_PER_DAY:
            write_restart_manifest(
                checkpoint_dir=checkpoint_dir,
                checkpoint_day=next_checkpoint_day,
                current_time=current_time,
                config_file=config_file,
                output_dir=output_dir,
                basename=basename,
            )
            next_checkpoint_day += 10

    return block_vars, current_time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run planetary simulation with Toon radiative transfer.")
    p.add_argument("-c", "--config", required=True, help="YAML configuration file")
    p.add_argument("--output-dir", default="output", help="Output directory")
    p.add_argument(
        "--restart-name",
        default="",
        help=(
            "Restart archive filename inside output dir (e.g. "
            "uranus_rt_h2o.00005.restart or uranus_rt_h2o.final.restart)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    block, eos, thermo_y, thermo_x, kinet, device, enable_kinetics = create_models_general(
        args.config,
        config,
        args.output_dir,
    )

    if args.restart_name:
        block_vars, current_time = block.initialize_from_restart(args.restart_name)
    else:
        block_vars, current_time = initialize_atm(block, config)

    for key, data in block_vars.items():
        if isinstance(data, torch.Tensor):
            print(f"{key}: shape={tuple(data.shape)} dtype={data.dtype} device={data.device}")

    rt_state = build_rt_state(
        block=block,
        eos=eos,
        block_vars=block_vars,
        config=config,
        config_file=args.config,
    )

    block.set_user_output_func(
        lambda vars: _compute_rt_output_fields(eos, thermo_y, thermo_x, vars, rt_state)
    )

    print(
        "RT forcing summary:",
        f"stellar_flux_nadir={rt_state.cfg.stellar_flux_nadir:.3f} W/m^2,",
        f"rt_update_dt={rt_state.cfg.update_dt:.1f} s,",
        f"zenith_mode={rt_state.cfg.zenith_mode},",
        f"obliquity={math.degrees(rt_state.cfg.obliquity_rad):.2f} deg,",
        f"bottom_relax_enabled={rt_state.bottom_relax.enabled},",
        f"kinetics_enabled={enable_kinetics}",
    )

    tlim = float(config["integration"]["tlim"])
    basename = Path(args.config).stem
    block_vars, current_time = run_simulation(
        block=block,
        eos=eos,
        thermo_y=thermo_y,
        thermo_x=thermo_x,
        kinet=kinet,
        block_vars=block_vars,
        current_time=current_time,
        tlim=tlim,
        rt_state=rt_state,
        config_file=args.config,
        output_dir=args.output_dir,
        basename=basename,
    )

    block.finalize(block_vars, current_time)


if __name__ == "__main__":
    main()
