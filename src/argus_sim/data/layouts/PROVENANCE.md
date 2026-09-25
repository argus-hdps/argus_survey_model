# Layout Data Provenance

The default ring layout is `A170r_1200` (`A170r_1200_telescopes.csv`, band table
`A170r_1200_bands_alt5_2026-09-23.csv`). Registry: `argus_sim.ring_layout.LAYOUTS`.

## A170r_1200_telescopes.csv (array layout)

PROVENANCE: github.com/argus-hdps/array_arrangement @ 9206cc3b13236a4c9fda2b1164db9e634f7ca241,
`array_layout/A170r_1200_telescopes.csv`. Its data rows are identical to that file (sha256 of the file shipped here:
0d2b3154541c5f083e57e8183756696cb234ff8e3e8e660bcf8eb4c045b8c9d1). The generator at the same commit writes the same rows:
`argus_array_layout.get_layout('A170r_1200')` then `write_telescope_csv(cams, layout_name='A170r_1200')`. The layout is described in
`array_layout/README_LAYOUT_A170r_1200.md` of that repository.

1200 OTAs in 22 rings (zenith distance 1.0-51.24 deg, pitch 2.392494 deg) on 8 subarrays, 4 mount types x 2
(subarray = 2*type + parity; 133/133/151/151/159/159/157/157 OTAs). Each ring is split between the two mounts of one
type at alternate stations. The camera field is 3.252897 deg (along the ring) x 2.442494 deg (radial). The file's
`az_deg` is a station bearing, with the footprint at `az_deg + 180` in the generator's frame. With the generator's
`fov_hits_xyz` and `tracking_matrix`, CSV azimuth A falls at compass azimuth A + 90 with the polar axis at north, so
the loader adds 90 deg.

The layout is registered as `A170r_1200` (the default) with the band table below. `A170r_1200-strips` is the same
layout with bands assigned by the strip rule.

## A170r_1200_bands_alt5_2026-09-23.csv (A170r_1200 band table)

sha256 51f8b087a78b1f69244a4ebfd334aeb50462eafc21dc6a9959f7755460d4da4c; 600 OTAs in beta (`b`, the ArgusSim `g`
band) and 600 in rho. The assignment maximises band alternation between successive 16-minute pointings (ratchets)
while balancing the two bands on every 0.5-deg dec line. The layout is E/W mirror-symmetric: 598 twin pairs of OTAs
have identical dec-line coverage and sit on opposite mounts. The table gives every pair opposite bands, so each
0.5-deg dec line is balanced exactly, or to 3.7% on the lines of ring 0. The array-frame alternation between
successive pointings is 0.8492 with overlaps credited (upper bound 0.9608; the strip rule gives 0.653). The exact
coverage engine gives 0.8493 over one summer season.
