# Throughput Data Provenance

All wavelengths are in nm unless noted. Files with wavelengths in Angstrom
are converted to nm at load time in `throughput.py` or `spectral_sky.py`.

## Detector

| File | Description | Units | Source |
|------|-------------|-------|--------|
| `imx455_qe.csv` | Sony IMX455 quantum efficiency | nm, fraction 0–1 | Digitized from Sony IMX455 datasheet |

## Atmosphere

| File | Description | Units | Source |
|------|-------------|-------|--------|
| `pachon_modtran_atm_aerosols.txt` | Atmospheric transmission at AM=1.0 | nm, fraction 0–1 | Gemini Observatory MODTRAN model for Cerro Pachon (zenith) |

## Optics

The telescope optics enter as one band-averaged efficiency per filter, `OPTICS_EFFICIENCY` in `throughput.py`
(no file). Each value is the optics transmission averaged over that filter's photon-rate response (filter x
detector QE x atmosphere at airmass 1.15, the typical survey airmass), which is the weighting
`spectral_sky.build_band_params` uses for the signal.

## Filters

Chroma Technology bandpass filters. Transmission values are in percent
(divided by 100 at load time).

| File | Description | Units | Source |
|------|-------------|-------|--------|
| `argus_g.csv` | Argus g-band filter | nm, percent 0–100 | Chroma Technology transmission data |
| `argus_r.csv` | Argus r-band filter | nm, percent 0–100 | Chroma Technology transmission data |
| `argus_i.csv` | Argus i-band filter | nm, percent 0–100 | Chroma Technology transmission data |
| `argus_A.csv` | Argus broadband (695nm short-pass, OD4 blocking to 1100nm) | nm, percent 0–100 | Chroma Technology transmission data |
| `argus_rho_ct699.csv` | Argus rho filter (CT699-259bp), shown in the depth tutorial | nm, fraction 0–1 | Filter vendor transmission curve for the CT699-259bp rho filter, averaged over the beam |

## Idealised Bandpasses

Idealised curves with short linear edges. The survey's `rho` band uses `bandpass_560_825nm.csv`. Transmission is a
fraction 0–1.

| File | Band | Description |
|------|------|-------------|
| `bandpass_560_825nm.csv` | `rho` | 560–825 nm band pass |
| `bandpass_560_850nm.csv` | `r+i` | 560–850 nm band pass |
| `longpass_550nm.csv` | `r+i+z` | 550 nm long pass |
| `longpass_700nm.csv` | `i+z` | 700 nm long pass |

## Reference Filters

| File | Description | Units | Source |
|------|-------------|-------|--------|
| `GCPD_Johnson.V.dat` | Johnson V bandpass | Angstroms (÷10 at load), fraction 0–1 | General Catalogue of Photometric Data |
| `Cameras_SQM.CM500.dat` | Sky Quality Meter spectral response | Angstroms (÷10 at load), fraction 0–1 | Unihedron SQM CM500 photocell response |

## Sky Templates

### Per-Component Sky Templates (ESO SkyCalc)

Generated 2026-02-25 with the
`skycalc_ipy` Python interface to the ESO SkyCalc service (Noll+ 2012,
A&A 543, A92; Jones+ 2013, A&A 560, A91).

Common query parameters: wavelength range 300–1100 nm (output in
Angstroms 3000–11000), fixed step 0.1 nm, vacuum wavelengths, airmass 1.2,
no thermal emission.

Only the spectral shape of each template matters. `SpectralSky` normalises
each template in the V band to give e⁻/s/arcsec² per S_10 unit, so the
absolute flux level cancels.

| File | Description | Units | Component flags | Extra parameters |
|------|-------------|-------|-----------------|------------------|
| `airglow_spectrum.csv` | Airglow emission (lower atmosphere + upper atmosphere + airglow continuum) | Angstroms, ph/s/um/arcsec²/m² | `incl_loweratm=Y`, `incl_upperatm=Y`, `incl_airglow=Y` | |
| `zodiacal_spectrum.csv` | Zodiacal light at ecliptic pole | Angstroms, ph/s/um/arcsec²/m² | `incl_zodiacal=Y` | `ecl_lon=135`, `ecl_lat=90` |
| `starlight_spectrum.csv` | Integrated unresolved starlight | Angstroms, ph/s/um/arcsec²/m² | `incl_starlight=Y` | |
| `moonlight_spectrum.csv` | Scattered moonlight (Rayleigh + Mie) | Angstroms, ph/s/um/arcsec²/m² | `incl_moon=Y` | `moon_sun_sep=90`, `moon_target_sep=45`, `moon_alt=45`, `moon_earth_dist=1.0` |

The moonlight template uses a reference geometry (quarter moon, 45° separation).
Its spectral shape depends only weakly on the geometry: the Rayleigh/Mie ratio
shifts slightly, a second-order colour change. The brightness comes from the
S_10 value of `Observatory.sky_components_at`.
