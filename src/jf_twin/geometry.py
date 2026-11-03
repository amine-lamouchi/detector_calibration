"""
Detector geometry: where each JUNGFRAU pixel sits, seen from the sample.

Arrays are shaped like psana's raw data, (n_modules, rows, cols); here one
module, centred 45 mm above the beam. A module is 512 x 1024 pixels of 75 um,
made of 2 x 4 ASICs of 256 x 256. A smaller `module_shape` keeps the central
rows/cols (a crop, the pixel size stays 75 um) so the simulation runs fast.
"""
from dataclasses import dataclass

import numpy as np

PIXEL = 75e-6  # m
MODULE_SHAPE = (512, 1024)
ASIC = 256
CENTER = (0.0, 45e-3)  # module centre (x, y) [m], beam at (0, 0)


@dataclass
class Geometry:
    distance: float = 0.1  # sample to detector [m]
    module_shape: tuple = MODULE_SHAPE

    def __post_init__(self):
        rows, cols = self.module_shape
        if rows > MODULE_SHAPE[0] or cols > MODULE_SHAPE[1]:
            raise ValueError(f"module_shape must fit in {MODULE_SHAPE}")
        self.row0 = (MODULE_SHAPE[0] - rows) // 2
        self.col0 = (MODULE_SHAPE[1] - cols) // 2
        r = np.arange(self.row0, self.row0 + rows)
        c = np.arange(self.col0, self.col0 + cols)
        dy = ((MODULE_SHAPE[0] - 1) / 2 - r) * PIXEL  # row 0 at the top
        dx = (c - (MODULE_SHAPE[1] - 1) / 2) * PIXEL
        self.x = np.broadcast_to(CENTER[0] + dx[None, :], (rows, cols))[None]
        self.y = np.broadcast_to(CENTER[1] + dy[:, None], (rows, cols))[None]
        self.shape = self.x.shape
        self.n_modules = 1

    @property
    def solid_angle(self):
        """Solid angle of each pixel [sr], flat detector normal to the beam."""
        cos = np.cos(np.arctan2(np.hypot(self.x, self.y), self.distance))
        return PIXEL**2 * cos**3 / self.distance**2

    def asic_index(self):
        """ASIC number 0..7 within the module, (n_modules, rows, cols)."""
        r = np.arange(self.row0, self.row0 + self.module_shape[0]) // ASIC
        c = np.arange(self.col0, self.col0 + self.module_shape[1]) // ASIC
        return np.broadcast_to(r[:, None] * 4 + c[None, :], self.shape)
