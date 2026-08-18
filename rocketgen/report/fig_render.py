"""Render an exported STL to a report figure, off screen, with a locked camera.

    .venv/Scripts/python.exe -m rocketgen.report.fig_render --oml spline

The render is scripted, not screen-grabbed, for the reason every figure in this repository is
scripted: the report gets regenerated. A locked camera also makes two renders of two different
shapes actually comparable, which a hand-taken view never is.

Nothing here is a measurement. The mesh is a picture. `docs/NTOP_NOTES.md` and CLAUDE.md
section 4 point 7 both say the notebook's own `mass_properties` beats the exported mesh by
about 16x on the smoke sphere, so every number in the report comes from the notebook.
"""
from __future__ import annotations

import os

from .figstyle import case_dir, out_path, select_study

#: Camera azimuth and elevation for the two standard views, degrees.
ISO_VIEW = (1.0, -1.35, 0.62)
SIDE_VIEW = (0.0, -1.0, 0.06)

#: Body colour and the light-grey ground the vehicle sits against. Matched to `figstyle`.
BODY_COLOUR = "#b9c22e"
EDGE_COLOUR = "#2b2b2b"


def render(stl_path: str, png_path: str, view: tuple[float, float, float],
           window: tuple[int, int] = (1800, 700), zoom: float = 1.0) -> str:
    """One off-screen render of `stl_path` to `png_path`."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    mesh = pv.read(stl_path)

    plotter = pv.Plotter(off_screen=True, window_size=list(window))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        color=BODY_COLOUR,
        smooth_shading=True,
        specular=0.30,
        specular_power=18.0,
        ambient=0.28,
        diffuse=0.72,
        show_edges=False,
    )
    # A silhouette rather than the full wireframe: at this triangle count edges turn the body
    # into a grey block and hide the shape the figure exists to show.
    plotter.add_silhouette(mesh, color=EDGE_COLOUR, line_width=1.6)
    plotter.enable_parallel_projection()
    plotter.camera_position = [
        (view[0], view[1], view[2]),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    plotter.reset_camera()
    plotter.camera.zoom(zoom)
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    plotter.screenshot(png_path)
    plotter.close()
    _crop_white(png_path)
    return png_path


def _crop_white(png_path: str, pad: int = 12) -> None:
    """Trim the white margin off a render.

    A parallel-projection camera framed on the whole bounding sphere leaves most of the frame
    empty, and a figure that is 70 percent white reads as a mistake. Cropping to the content
    box is also what makes two renders comparable: both end up scaled by their own subject.
    """
    from PIL import Image, ImageChops

    img = Image.open(png_path).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    box = ImageChops.difference(img, bg).getbbox()
    if box is None:
        return
    left = max(box[0] - pad, 0)
    upper = max(box[1] - pad, 0)
    right = min(box[2] + pad, img.width)
    lower = min(box[3] + pad, img.height)
    img.crop((left, upper, right, lower)).save(png_path)


def make_figure(path: str | None = None) -> str:
    """Two panels, iso above and side below, written as one PNG."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    stl = os.path.join(case_dir(), "converged", "geom", "sv1.stl")
    if not os.path.isfile(stl):
        raise SystemExit(
            f"{stl} is missing. Run `run_sv1.py --stage converged`, which is the only stage "
            "that turns the exports on."
        )

    tmp_iso = out_path("_sv1_iso_raw.png")
    tmp_side = out_path("_sv1_side_raw.png")
    render(stl, tmp_iso, ISO_VIEW, window=(1800, 760), zoom=1.30)
    render(stl, tmp_side, SIDE_VIEW, window=(1800, 520), zoom=1.55)

    with plt.rc_context({"figure.facecolor": "white", "savefig.facecolor": "white"}):
        fig, axes = plt.subplots(2, 1, figsize=(9.4, 3.6),
                                 gridspec_kw={"height_ratios": [1.15, 1.0]})
        for ax, png, title in (
            (axes[0], tmp_iso, "(a) isometric"),
            (axes[1], tmp_side, "(b) side, showing the splined nose and boattail"),
        ):
            ax.imshow(mpimg.imread(png))
            ax.set_axis_off()
            ax.set_title(title, loc="left", fontsize=8.5, family="monospace")
        fig.subplots_adjust(left=0.005, right=0.995, top=0.945, bottom=0.005, hspace=0.16)
        path = path or out_path("sv1_iso.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
    for f in (tmp_iso, tmp_side):
        try:
            os.remove(f)
        except OSError:
            pass
    return path


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(description=__doc__)
    _ap.add_argument("--oml", default="ogive", choices=["ogive", "spline"],
                     help="which study to render; spline reads runs/SV-1_spline")
    select_study(_ap.parse_args().oml)
    print(make_figure())
