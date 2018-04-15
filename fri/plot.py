import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

# Get three colors for each relevance type
color_palette_3 = sns.color_palette(palette="muted", n_colors=3)


def plot_relevance_bars(ax: matplotlib.axes.Axes, ranges, ticklabels=None, classes=None, numbering=True):
    N = len(ranges)

    # Ticklabels
    if ticklabels is None:
        ticks = np.arange(N) + 1
    else:
        ticks = list(ticklabels)
        if numbering:
            for i in range(N):
                ticks[i] += " - {}".format(i + 1)

    # Interval sizes
    ind = np.arange(N) + 1
    width = 0.6
    upper_vals = ranges[:, 1]
    lower_vals = ranges[:, 0]
    height = upper_vals - lower_vals
    # Minimal height to make very small intervals visible
    height[height < 0.001] = 0.001

    # Bar colors
    if classes is None:
        new_classes = np.zeros(N).astype(int)
        color = [color_palette_3[c] for c in new_classes]
    else:
        color = [color_palette_3[c] for c in classes]

    # Plot the bars
    bars = ax.bar(ind, height, width, bottom=lower_vals, tick_label=ticks, align="center", edgecolor=["black"] * N,
                  linewidth=1.3, color=color)

    ax.set_xticklabels(ticks)
    if ticklabels is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    # ax.tick_params(rotation="auto")
    # Limit the y range to 0,1 or 0,L1
    ax.set_ylim([0, max(ranges[:, 1]) * 1.1])

    ax.set_ylabel('relevance', fontsize=19)
    ax.set_xlabel('feature', fontsize=19)

    if classes is not None:
        relevance_classes = ["Irrelevant", "Weakly relevant", "Strongly relevant"]
        patches = []
        for i, rc in enumerate(relevance_classes):
            patch = mpatches.Patch(color=color_palette_3[i], label=rc)
            patches.append(patch)

        ax.legend(handles=patches)

    return bars

def plotIntervals(ranges, ticklabels=None, invert=False, classes=None):
    # Figure Parameters
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    N = len(ranges)

    out = plot_relevance_bars(ax, ranges, ticklabels=ticklabels, classes=classes)
    fig.autofmt_xdate()
    # Invert the xaxis for cases in which the comparison with other tools
    if invert:
        plt.gca().invert_xaxis()
    return fig


def plot_dendrogram_and_intervals(intervals, linkage, figsize=(13, 7), ticklabels=None, classes=None,
                                  **kwargs):
    fig, (ax2, ax) = plt.subplots(2, 1, figsize=figsize)
    # Top dendrogram plot
    d = dendrogram(
        linkage,
        color_threshold=0,
        leaf_rotation=0.,  # rotates the x axis labels
        leaf_font_size=12.,  # font size for the x axis labels
        ax=ax2,
        **kwargs
    )
    # Get index determined through linkage method and dendrogram
    rearranged_index = d['leaves']
    ranges = intervals[rearranged_index]

    if ticklabels is None:
        ticks = np.array(rearranged_index)
        ticks += 1  # Index starting at 1
    else:
        ticks = list(ticklabels[rearranged_index])
        for i in range(len(intervals)):
            ticks[i] += " - {}".format(rearranged_index[i] + 1)

    plot_relevance_bars(ax, ranges, ticklabels=ticks,
                        classes=classes[rearranged_index] if classes is not None else None, numbering=False)

    ax.margins(x=0)
    ax2.set_xticks([])
    ax2.margins(x=0)
    plt.tight_layout()

    return fig
