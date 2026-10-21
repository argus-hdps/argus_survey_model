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
`spectral_sky.build_band_params` uses for the signal. The beta and rho values (0.785, 0.835) were recomputed for the
beam-averaged curves below with the same optics transmission curve as before.

## Filters

The two Argus survey bands, Chroma Technology bandpass filters. Both start from the vendor's theoretical (design)
curve at normal incidence. Each is then averaged over the converging telescope beam: per-angle edge shifts from the
vendor's coating tables, averaged over the angles of incidence at the filter and over the sensor, unpolarised. Filter
only: optics, QE and atmosphere are applied separately.

| File | Band | Filter | Units | Source |
|------|------|--------|-------|--------|
| `argus_beta_ct446.csv` | `beta` | Chroma CT446-162bp | nm, fraction 0–1 | Vendor theoretical curve, averaged over the beam |
| `argus_rho_ct699.csv` | `rho` | Chroma CT699-259bp | nm, fraction 0–1 | Vendor theoretical curve, averaged over the beam |

Older versions called beta `g` (the collimated CT446-162 curve, `argus_g.csv`) and modelled rho as an idealised
560–825 nm box; they also carried the unused `A`, `r`, `i`, `r+i`, `i+z` and `r+i+z` filters. Those bands and files
are removed. Survey runs and visit stores written by older versions still say `g`; readers translate it to `beta`
(`throughput.canonical_band`).

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
