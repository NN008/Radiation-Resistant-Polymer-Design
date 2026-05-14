import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

def draw_pipeline(out_png="Dataset_Enrichment_Pipeline_FINAL.png",
                  out_svg="Dataset_Enrichment_Pipeline_FINAL.svg"):
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.axis("off")

    # ----- layout params -----
    W, H   = 2.8, 1.25      # box size
    GX, GY = 0.6, 1.6       # gaps
    x0     = 1.2
    y_top  = 5.7
    y_bot  = 3.1

    xs = [x0 + i*(W+GX) for i in range(4)]  # 4 columns

    # dynamic margins so nothing is cut off
    x_min = x0 - 1.0
    x_max = xs[-1] + W + 1.0
    y_min = y_bot - 1.0
    y_max = y_top + H + 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # title
    ax.text((x_min+x_max)/2, y_max-0.3,
            "Dataset Enrichment and Label Completion Pipeline",
            ha="center", va="top", fontsize=14, fontweight="bold")

    labels = [
        "Raw datasets\n(PI1M, JCIM, Bicerano)",
        "Combine & remove\nduplicates",
        "Compute RDKit\ndescriptors",
        "Drop entries with\nfailed descriptors",
        "Train RF predictors\n(Tg & MAC)",
        "Impute missing labels\n(Tg, MAC)",
        "Remove outliers\n(cleaned values)",
        "Final enriched dataset\n(ready for training)",
    ]

    # box + badge
    def add_box(step, x, y, text):
        ax.add_patch(FancyBboxPatch((x, y), W, H,
                                    boxstyle="round,pad=0.22,rounding_size=0.06",
                                    ec="black", lw=1.2, fc="#ECF3FB"))
        # move number tighter into top-left
        cx, cy, r = x + 0.15, y + H - 0.15, 0.16
        ax.add_patch(Circle((cx, cy), r, fc="black", ec="black", lw=0.8))
        ax.text(cx, cy, f"{step}", ha="center", va="center",
                fontsize=8, color="white")
        ax.text(x + W/2, y + H/2, text, ha="center", va="center",
                fontsize=10)

    # draw boxes (snake 2×4)
    for i in range(4):
        add_box(i+1, xs[i], y_top, labels[i])
    add_box(5, xs[3], y_bot, labels[4])
    add_box(6, xs[2], y_bot, labels[5])
    add_box(7, xs[1], y_bot, labels[6])
    add_box(8, xs[0], y_bot, labels[7])

    # arrows
    def arrow(p1, p2):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
                                     mutation_scale=12, lw=1.2, color="black"))

    # top row →
    for i in range(3):
        arrow((xs[i] + W, y_top + H/2), (xs[i+1], y_top + H/2))
    # down at right
    arrow((xs[3] + W/2, y_top), (xs[3] + W/2, y_bot + H))
    # bottom row ←
    arrow((xs[3], y_bot + H/2), (xs[2] + W, y_bot + H/2))
    arrow((xs[2], y_bot + H/2), (xs[1] + W, y_bot + H/2))
    arrow((xs[1], y_bot + H/2), (xs[0] + W, y_bot + H/2))

    # generous figure margins (no tight bbox, avoids clipping)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.08)
    fig.savefig(out_png, dpi=400)   # PNG
    fig.savefig(out_svg)            # SVG (use this in LaTeX)

# run it
draw_pipeline()
