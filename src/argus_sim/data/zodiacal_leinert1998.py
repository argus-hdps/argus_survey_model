"""Zodiacal light surface brightness from Leinert et al. (1998), Table 17.

Values in S10(V) units, tabulated as a function of helioecliptic longitude
(solar elongation along the ecliptic) and absolute ecliptic latitude.
Original source: Levasseur-Regourd & Dumont (1980, A&A 84, 277).

``ZODIACAL_S10`` holds the transcribed table with NaN where the published
table has no entry.  ``ZODIACAL_S10_FILLED`` sets each blank to the last
tabulated value in its row, which keeps brightness non-increasing in |beta|
without adding values the table does not give.
"""

import numpy as np

ELONGATION_DEG = np.array([30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180])

ABS_BETA_DEG = np.array([0, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90])

_ = np.nan

# fmt: off
# Shape: (len(ELONGATION_DEG), len(ABS_BETA_DEG))
ZODIACAL_S10 = np.array([
    [3740, 1610,  640,  275,  150,  100,   78,   46,   38,   36,    _],  # elong=30
    [1150,  642,  320,  163,  105,   78,   64,   43,   37,   36,    _],  # elong=45
    [ 437,  296,  180,  113,   82,   65,   55,   41,   36,   35,    _],  # elong=60
    [ 288,  198,  130,   90,   69,   56,   48,   38,   35,    _,    _],  # elong=75
    [ 221,  157,  105,   76,   60,   50,   43,   36,   34,    _,    _],  # elong=90
    [ 182,  132,   92,   68,   55,   46,   40,   35,    _,    _,    _],  # elong=105
    [ 161,  116,   82,   62,   50,   43,   38,   34,    _,    _,    _],  # elong=120
    [ 153,  109,   77,   58,   47,   41,   36,    _,    _,    _,    _],  # elong=135
    [ 153,  107,   75,   56,   45,   39,    _,    _,    _,    _,    _],  # elong=150
    [ 168,  114,   78,   57,   45,    _,    _,    _,    _,    _,    _],  # elong=165
    [ 198,  126,   82,   58,    _,    _,    _,    _,    _,    _,    _],  # elong=180
], dtype=np.float64)
# fmt: on

del _


def _hold_last_tabulated(table: np.ndarray) -> np.ndarray:
    filled = table.copy()
    for row in filled:
        last = row[0]
        for j in range(row.size):
            if np.isnan(row[j]):
                row[j] = last
            else:
                last = row[j]
    return filled


ZODIACAL_S10_FILLED = _hold_last_tabulated(ZODIACAL_S10)
