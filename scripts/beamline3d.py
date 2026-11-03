"""
3D view of the simulated measurement with xrt's xrtGlow, in the style of the
tfs_alignment viewer: the beam through the hutch attenuator onto the copper
foil, and the detector lit by the foil's fluorescence. Used by movie.py; needs
`pip install -e ".[movie]"` and a screen.

A cartoon, not to scale along the beam: the attenuator is drawn just upstream of
the foil and the beam ~1 mm wide. The blades are the simulator's (material and
thickness), and so is the foil-detector geometry (detector 100 mm away, 45 mm
above the beam). Beam rays are traced by xrt and coloured by density as on the
tfs YAG images, fainter and fewer after the attenuator. Each dot on the detector
is a pixel, coloured by the gain stage it is in.
"""
import functools
import io
import sys
import types
import warnings

import numpy as np

# xrt's Qt layer imports QtWebEngineWidgets (only for its help pages): give it an empty stand-in
sys.modules.setdefault("PyQt5.QtWebEngineWidgets", types.ModuleType("PyQt5.QtWebEngineWidgets"))

import matplotlib  # noqa: E402
import xrt.backends.raycing as raycing  # noqa: E402
import xrt.backends.raycing.materials as rm  # noqa: E402
import xrt.backends.raycing.oes as roe  # noqa: E402
import xrt.backends.raycing.run as rr  # noqa: E402
import xrt.backends.raycing.screens as rsc  # noqa: E402
import xrt.backends.raycing.sources as rs  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402

from jf_twin.attenuator import MFX_FILTERS  # noqa: E402

XRT_TESTED = "1.6.2"  # the patches below rely on xrtGlow internals of this version

Y_SOURCE, Y_ATT, Y_FOIL = 600, 820, 960   # positions along the beam [mm]; the attenuator is moved close
BLADE = 3.0           # blade half size [mm]
LIFT = 7.5            # a blade out of the beam sits this much higher [mm]
BEAM_SIGMA = 0.45     # [mm]
PIXEL_STEP = 4        # one dot every 4 pixels on the detector
FLUO_SHOWN = 0.06     # fraction of the foil-to-pixel rays drawn
SEAM = 3              # pixels each side of a chip border left dark, so the 8 chips show

COLORS = {"Si": (0.62, 0.64, 0.7, 0.8), "Cu": (0.85, 0.5, 0.25, 1.0), "housing": (0.16, 0.16, 0.18, 1.0)}
GNUPLOT = matplotlib.colormaps["gnuplot"]  # Matt's colormap, as in the tfs viewer


def _cosmetic(patch):
    """Run a patch of xrtGlow internals; if xrt changed and it fails, warn and go on without it."""
    @functools.wraps(patch)
    def safe(*args, **kwargs):
        try:
            return patch(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            import xrt
            warnings.warn(f"3D view: {patch.__name__} skipped ({error!r}); tested with xrt {XRT_TESTED}, "
                          f"installed {xrt.__version__}", stacklevel=2)
    return safe


def beam_brightness(t):
    """How bright the beam is drawn for transmission t: 0..1 on a log scale."""
    return float(np.clip(1 + np.log10(max(t, 1e-6)) / 4.5, 0.06, 1))


class Scene:
    """
    The beam line in xrtGlow. `show(...)` sets blades, beam and detector and
    redraws; `grab()` returns the 3D view as a PIL image.
    """

    def __init__(self, geometry, gain_colors, size=(1100, 720), camera=((0, 10, 30), (16, 7, 16), (0, 925, 22))):
        from xrt.gui import xrtGlow as xrtglow
        from xrt.gui.commons import qt

        _patch_glow()
        self.app = qt.QApplication.instance() or qt.QApplication([])
        self.gain_rgba = np.array([to_rgba(c) for c in gain_colors])
        self.pixel_z = geometry.y[0, :, 0] * 1e3     # height of each pixel row above the beam [mm]
        self.pixel_x = geometry.x[0, 0, :] * 1e3
        self.detector_y = Y_FOIL + geometry.distance * 1e3
        rows, cols = np.meshgrid(np.arange(1, len(self.pixel_z), PIXEL_STEP),
                                 np.arange(1, len(self.pixel_x), PIXEL_STEP), indexing="ij")
        self.dots = rows.ravel(), cols.ravel()       # the pixels drawn as dots
        self.chip_row0, self.chip_col0 = geometry.row0, geometry.col0   # crop offset in the module
        self.rng = np.random.default_rng(0)
        self.gains = np.zeros(geometry.shape[1:], int)
        self.state = {"beam": 0.0, "t": 1.0, "detector": None}
        self.label_xy = {}  # label -> (x, y) position in the grabbed image, as fractions

        bl = self.bl = raycing.BeamLine()
        bl.source = rs.GeometricSource(
            bl, "source", center=[0, Y_SOURCE, 0], nrays=4000, distx="normal", dx=BEAM_SIGMA, distz="normal",
            dz=BEAM_SIGMA, distxprime="normal", dxprime=1e-6, distzprime="normal", dzprime=1e-6,
            distE="lines", energies=(9500,))
        # traced: a thin transparent plate at the attenuator's exit, where the beam changes brightness
        self.exit = roe.Plate(bl, "att_exit", center=[0, Y_ATT + 70, 0], pitch=np.pi / 2,
                              material=rm.Material("Be", rho=1.848, kind="plate"), t=1e-3,
                              limPhysX=[-10, 10], limPhysY=[-10, 10])
        self.foil_screen = rsc.Screen(bl, "foil_screen", center=[0, Y_FOIL, 0])
        self.fluo = rs.GeometricSource(bl, "fluo", center=[0, Y_FOIL, 0], nrays=len(self.dots[0]), distx=None,
                                       distz=None, distxprime=None, distzprime=None, distE="lines",
                                       energies=(8040,))
        self.detector_screen = rsc.Screen(bl, "detector_screen", center=[0, self.detector_y, 0])

        # drawn only: blades, foil, detector housing, and invisible anchors for the labels
        def plate(name, y, t, half=(BLADE, BLADE), z=0.0):
            return roe.Plate(bl, name, center=[0, y, z], pitch=np.pi / 2, material=None, t=t,
                             limPhysX=[-half[0], half[0]], limPhysY=[-half[1], half[1]])

        thickness = np.array([t for _, t in MFX_FILTERS]) * 1e-3
        ys = Y_ATT + np.concatenate([[0], np.cumsum(thickness[:-1] + 4)])
        self.blades = [plate(f"blade_{i:02d}", y, t, z=LIFT) for i, (y, t) in enumerate(zip(ys, thickness))]
        half = (np.ptp(self.pixel_x) / 2 + 1.5, np.ptp(self.pixel_z) / 2 + 1.5)
        z_det = (self.pixel_z.max() + self.pixel_z.min()) / 2
        self.drawn = [*self.blades, plate("foil", Y_FOIL, 0.05, (4, 4)),
                      plate("housing", self.detector_y + 0.3 + 12, 12, half, z=z_det),
                      plate("beam_label", Y_ATT - 10, 0, z=-3),
                      plate("att_label", (ys[0] + ys[-1]) / 2, 0, z=LIFT + BLADE + 4),
                      plate("foil_label", Y_FOIL, 0, z=-7),
                      plate("detector_label", self.detector_y, 0, z=z_det + half[1] + 9)]
        self.materials = {**{b.name: COLORS[m] for b, (m, _) in zip(self.blades, MFX_FILTERS)},
                          "foil": COLORS["Cu"], "housing": COLORS["housing"]}
        self.labels = {"beam_label": "X-ray beam", "att_label": "attenuator", "foil_label": "Cu foil", "detector_label": "detector"}
        self.footprints = ("foil_screen", "detector_screen")

        rr.run_process = self._run_process
        rr.run_process(bl)
        self.glow = xrtglow.xrtGlow(self._rays())
        self.glow.bl, bl.blViewer = bl, self.glow
        self.glow.customGlWidget.scene = self
        self.glow.resize(size[0] + 450, size[1])
        self.glow.canvasSplitter.setSizes([size[0], 450])
        self.glow.show()
        self.camera = camera
        self._style()

    # --- ray tracing ---------------------------------------------------------

    def _run_process(self, bl):
        # Beams live only in outDict: xrt names them from this function's locals (prepare_flow)
        outDict = {"source": bl.source.shine()}
        outDict["att_exit"], outDict["att_exit_local1"], outDict["att_exit_local2"] = \
            self.exit.double_refract(outDict["source"])
        outDict["foil_screen"] = self.foil_screen.expose(outDict["att_exit"])
        outDict["fluo"] = self.fluo.shine()
        self._aim(outDict["fluo"])
        outDict["detector_screen"] = self.detector_screen.expose(outDict["fluo"])
        bl.prepare_flow()               # records the flow on the first call only
        bl.beamsDict.update(outDict)    # later calls: hand xrtGlow the new beams
        return outDict

    def _aim(self, fluo):
        """Point one fluorescence ray at each pixel drawn."""
        rows, cols = self.dots
        aim = np.c_[self.pixel_x[cols], np.full(len(rows), self.detector_y - Y_FOIL), self.pixel_z[rows]]
        aim /= np.linalg.norm(aim, axis=1, keepdims=True)
        fluo.a[:], fluo.b[:], fluo.c[:] = aim.T

    def _rays(self):
        """Traced rays for xrtGlow, plus the drawn-only bodies."""
        rays = self.bl.export_to_glow()
        for element in self.drawn:
            rays[2][element.name] = [element, 1]
        return rays

    def colors(self, start, end=None):
        """RGBA of ray segments (start, end points, global mm) or of footprints (start only)."""
        s = self.state
        hit = start if end is None else end
        rgba = np.zeros((len(start), 4), np.float32)
        on_detector = np.abs(hit[:, 1] - self.detector_y) < 1
        # detector: each pixel in the colour of its gain stage (or all in one colour in the dark run)
        col = np.rint((hit[on_detector, 0] - self.pixel_x[0]) / (self.pixel_x[1] - self.pixel_x[0])).astype(int)
        row = np.rint((hit[on_detector, 2] - self.pixel_z[0]) / (self.pixel_z[1] - self.pixel_z[0])).astype(int)
        row, col = np.clip(row, 0, self.gains.shape[0] - 1), np.clip(col, 0, self.gains.shape[1] - 1)
        if s["detector"] is not None:
            if end is None:
                rgba[on_detector] = to_rgba(s["detector"])
        else:
            rgba[on_detector] = self.gain_rgba[self.gains[row, col]]
            if end is None:
                rgba[on_detector, 3] = s["beam"]
            else:
                rgba[on_detector, 3] = 0.35 * s["beam"] * (self.rng.random(on_detector.sum()) < FLUO_SHOWN)
        if end is None:  # dark seams at the chip borders (the real module has no gap there, its edge pixels are larger)
            rgba[np.flatnonzero(on_detector)[self._seam(row, col)], 3] = 0
        # beam: coloured by density, dimmer and sparser after the attenuator
        beam = ~on_detector
        density = np.exp(-(start[beam, 0] ** 2 + start[beam, 2] ** 2) / (2 * BEAM_SIGMA ** 2))
        after = start[beam, 1] > self.exit.center[1] - 1 if end is not None else np.ones(beam.sum(), bool)
        b = np.where(after, beam_brightness(s["t"]), 1.0)
        rgba[beam] = GNUPLOT((0.15 + 0.85 * np.sqrt(density)) * (0.35 + 0.65 * b))
        keep = self.rng.random(beam.sum()) < np.where(after, b, 1.0)
        rgba[beam, 3] = (0.9 if end is None else 0.35) * s["beam"] * keep
        return rgba

    def _seam(self, row, col):
        """Pixels next to a border between two chips (not the module edge)."""
        near = np.zeros(len(row), bool)
        for index, n_chips in ((row + self.chip_row0, 2), (col + self.chip_col0, 4)):
            border = np.rint(index / 256)
            near |= (np.abs(index + 0.5 - 256 * border) <= SEAM) & (border > 0) & (border < n_chips)
        return near

    # --- view ------------------------------------------------------------------

    @_cosmetic
    def _style(self):
        from xrt.gui.commons import qt

        rotations, scale, center = self.camera
        glow, widget = self.glow, self.glow.customGlWidget
        glow.gridPanel.setChecked(False)
        widget.lineWidth, widget.pointSize = 1.0, 4
        widget.lineOpacity = widget.pointOpacity = 1.0  # alpha comes from Scene.colors
        widget.globalNorm = True
        root = glow.segmentsModelRoot
        for row in range(1, root.rowCount()):  # columns: name, footprints, surface, label
            name = root.child(row, 0).text()
            root.child(row, 1).setCheckState(qt.Checked if name in self.footprints else qt.Unchecked)
            root.child(row, 2).setCheckState(qt.Checked if name in self.materials else qt.Unchecked)
            root.child(row, 3).setCheckState(qt.Checked if name in self.labels else qt.Unchecked)
        widget.rotations = np.float32([[rotations[0], 1, 0, 0], [rotations[1], 0, 1, 0], [rotations[2], 0, 0, 1]])
        widget.updateQuats()
        glow.updateRotationFromGL(widget.rotations)
        glow.updateScaleFromGL(list(scale))
        widget.coordOffset, widget.tVec = list(center), np.float32([0, 0, 0])
        widget.populateVerticesArray(root)
        widget.glDraw()

    def show(self, att, t, beam=1.0, gains=None, detector=None):
        """
        att: insertion of each blade, 0 (out) .. 1 (in); t: its transmission.
        beam: 0 (off) .. 1. gains: gain stage per pixel (rows, cols).
        detector: one colour for the whole detector (dark run), or None.
        """
        for blade, v in zip(self.blades, att):
            blade.center = [0, blade.center[1], LIFT * (1 - v)]
        self.state = {"beam": beam, "t": t, "detector": detector}
        if gains is not None:
            self.gains = gains
        self._run_process(self.bl)
        self.glow.updateOEsList(self._rays())
        self._style()

    def grab(self):
        """The 3D view as a PIL image."""
        from PIL import Image
        from xrt.gui.commons import qt

        widget = self.glow.customGlWidget
        widget.glDraw()
        widget.glDraw()  # grabFrameBuffer returns the frame drawn before the last one
        buffer = qt.QtCore.QBuffer()
        buffer.open(qt.QtCore.QIODevice.ReadWrite)
        widget.grabFrameBuffer().save(buffer, "PNG")
        return Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGB")


@_cosmetic
def _patch_glow():
    """
    xrtGlow cosmetics: our colours for rays and footprints (Scene.colors),
    material colours for the bodies, label positions for the movie. Patches
    populateVerticesArray, plotOeSurface, setMaterial and drawText: check after
    xrt upgrades.
    """
    from OpenGL import GL as gl, GLU
    from xrt.gui.commons import qt
    from xrt.gui.xrtGlow import xrtGlWidget

    if getattr(xrtGlWidget, "_jf_patched", False):
        return
    populate, plot_surface = xrtGlWidget.populateVerticesArray, xrtGlWidget.plotOeSurface
    set_material, draw_text = xrtGlWidget.setMaterial, xrtGlWidget.drawText

    def populateVerticesArray(self, root):
        populate(self, root)
        scene = getattr(self, "scene", None)
        if scene is None:
            return
        offset = np.asarray(self.coordOffset)
        if getattr(self, "verticesArray", None) is not None and len(self.verticesArray):
            v = self.verticesArray + offset
            self.raysColor = np.repeat(scene.colors(v[0::2], v[1::2]), 2, axis=0)
        if getattr(self, "footprintsArray", None) is not None and len(self.footprintsArray):
            self.dotsColor = scene.colors(self.footprintsArray + offset)

    def plotOeSurface(self, oe, is2ndXtal):
        scene = getattr(self, "scene", None)
        self._body = scene.materials.get(oe.name) if scene is not None else None
        translucent = self._body is not None and self._body[3] < 1
        if translucent:
            gl.glDepthMask(gl.GL_FALSE)
        try:
            return plot_surface(self, oe, is2ndXtal)
        finally:
            if translucent:
                gl.glDepthMask(gl.GL_TRUE)
            self._body = None

    def setMaterial(self, mat):
        body = getattr(self, "_body", None)
        if body is None:
            return set_material(self, mat)
        r, g, b, a = body
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, [0.35 * r, 0.35 * g, 0.35 * b, a])
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, [r, g, b, a])
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, [0.6, 0.6, 0.6, a])
        gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_EMISSION, [0.25 * r, 0.25 * g, 0.25 * b, a])
        gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 60)

    def drawText(self, coord, text, *args, useCaption=False, **kwargs):
        scene = getattr(self, "scene", None)
        if useCaption and scene is not None:  # labels: only note where they land, the movie draws them
            name = text.split(":")[0].strip()
            if name in scene.labels:
                view = gl.glGetIntegerv(gl.GL_VIEWPORT)
                x, y, _ = GLU.gluProject(*coord, model=gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX),
                                         proj=gl.glGetDoublev(gl.GL_PROJECTION_MATRIX), view=view)
                scene.label_xy[scene.labels[name]] = (x / view[2], 1 - y / view[3])
            return None
        return draw_text(self, coord, text, *args, useCaption=useCaption, **kwargs)

    def handler(mode, context, message):
        if "non-exposed window" not in message:
            print(message)

    qt.QtCore.qInstallMessageHandler(handler)
    xrtGlWidget.populateVerticesArray, xrtGlWidget.plotOeSurface = populateVerticesArray, plotOeSurface
    xrtGlWidget.setMaterial, xrtGlWidget.drawText = setMaterial, drawText
    xrtGlWidget._jf_patched = True
