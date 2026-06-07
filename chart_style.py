"""
chart_style.py — matplotlib styling for the comp-finance blog
Apply once at the top of any analysis script:

    from chart_style import apply, PALETTE, save_fig
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Colour palette ────────────────────────────────────────────────────────────
# Mirrors the blog's CSS custom properties.

BG_PAGE   = "#0f0f0f"   # --bg-primary   (page background)
BG_PANEL  = "#1a1a1a"   # --bg-secondary (viz panel, figure bg)
BG_AXES   = "#141414"   # code-block bg  (slightly darker axes)
BG_CARD   = "#242424"   # card highlight  (legend, annotations)

TEXT_HIGH  = "#f8fafc"   # --text-primary
TEXT_MED   = "#cbd5e1"   # --text-secondary
TEXT_LOW   = "#94a3b8"   # --text-muted   (tick labels, minor annotations)
BORDER     = "#2d2d2d"   # --bg-tertiary  (spines, grid)

# Ordered palette — use in sequence for multi-series charts
PALETTE = [
    "#34d399",   # emerald      primary series
    "#60a5fa",   # blue         secondary series
    "#f59e0b",   # amber        tertiary / accent
    "#f87171",   # red          negative / warning
    "#a78bfa",   # violet       fifth series
    "#22c55e",   # green        sixth series
    "#38bdf8",   # sky          seventh series
    "#fb923c",   # orange       eighth series
]

# Single-purpose aliases
C_PRIMARY  = PALETTE[0]   # emerald  — "biased" / main line
C_BLUE     = PALETTE[1]   # blue     — "reference" / comparison line
C_AMBER    = PALETTE[2]   # amber    — highlights, annotations
C_RED      = PALETTE[3]   # red      — negative returns, warnings
C_VIOLET   = PALETTE[4]   # violet   — fifth series


def apply(font: str = "DejaVu Sans") -> None:
    """
    Set global rcParams. Call once before any plt.figure() calls.

    Args:
        font: Matplotlib font family. 'Inter' works if the font is installed
              (pip install matplotlib; then place Inter .ttf in mpl font dir).
              Falls back to 'DejaVu Sans' which ships with matplotlib.
    """
    mpl.rcParams.update({
        # Figure
        "figure.facecolor":      BG_PANEL,
        "figure.edgecolor":      BG_PANEL,
        "figure.dpi":            150,
        "figure.figsize":        (10, 5),

        # Axes
        "axes.facecolor":        BG_AXES,
        "axes.edgecolor":        BORDER,
        "axes.labelcolor":       TEXT_MED,
        "axes.titlecolor":       TEXT_HIGH,
        "axes.titlesize":        13,
        "axes.titleweight":      "bold",
        "axes.titlepad":         12,
        "axes.labelsize":        10,
        "axes.labelpad":         8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.prop_cycle":       mpl.cycler(color=PALETTE),

        # Grid
        "axes.grid":             True,
        "grid.color":            BORDER,
        "grid.linewidth":        0.6,
        "grid.alpha":            0.6,
        "grid.linestyle":        "--",

        # Ticks
        "xtick.color":           TEXT_LOW,
        "ytick.color":           TEXT_LOW,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        "xtick.major.pad":       5,
        "ytick.major.pad":       5,
        "xtick.direction":       "out",
        "ytick.direction":       "out",

        # Lines
        "lines.linewidth":       1.8,
        "lines.solid_capstyle":  "round",
        "patch.linewidth":       0.5,

        # Legend
        "legend.facecolor":      BG_CARD,
        "legend.edgecolor":      BORDER,
        "legend.labelcolor":     TEXT_MED,
        "legend.fontsize":       9,
        "legend.framealpha":     1.0,
        "legend.borderpad":      0.6,
        "legend.labelspacing":   0.4,

        # Font
        "font.family":           font,
        "font.size":             10,

        # Saving
        "savefig.facecolor":     BG_PANEL,
        "savefig.edgecolor":     BG_PANEL,
        "savefig.bbox":          "tight",
        "savefig.pad_inches":    0.2,
        "savefig.dpi":           150,
    })


def save_fig(fig: "plt.Figure", path: str, **kwargs) -> None:
    """
    Save a figure with blog-consistent settings.
    Equivalent to fig.savefig(path) but enforces facecolor and dpi.

    Usage:
        fig, ax = plt.subplots()
        ...
        save_fig(fig, "figures/fig1_cumulative_returns.png")
    """
    fig.savefig(
        path,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0.2,
        **kwargs,
    )


def styled_title(ax: "mpl.axes.Axes", title: str, subtitle: str = "") -> None:
    """
    Two-line title: bold primary in TEXT_HIGH, smaller subtitle in TEXT_LOW.

    Usage:
        styled_title(ax, "Cumulative Returns", "Log-scale, equal-weight monthly")
    """
    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", loc="left",
                     color=TEXT_HIGH, fontsize=12, fontweight="bold",
                     linespacing=1.6)
        # Dim the subtitle line via a second text object for finer control
        ax.set_title(title, loc="left",
                     color=TEXT_HIGH, fontsize=12, fontweight="bold")
        ax.text(0, 1.04, subtitle, transform=ax.transAxes,
                color=TEXT_LOW, fontsize=9, va="bottom")
    else:
        ax.set_title(title, loc="left",
                     color=TEXT_HIGH, fontsize=12, fontweight="bold")


def annotate_band(ax: "mpl.axes.Axes",
                  y_upper, y_lower,
                  color: str = C_AMBER,
                  alpha: float = 0.12,
                  label: str = "") -> None:
    """
    Fill between two series with a semi-transparent band (e.g. survivorship premium).

    Usage:
        annotate_band(ax, biased_returns, pit_returns, color=C_AMBER,
                      label="Survivorship premium")
    """
    ax.fill_between(range(len(y_upper)), y_lower, y_upper,
                    color=color, alpha=alpha, label=label, linewidth=0)
