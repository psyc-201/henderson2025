import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import copy

def set_all_font_sizes(fs):
    
    plt.rc('font', size=fs)          # controls default text sizes
    plt.rc('axes', titlesize=fs)     # fontsize of the axes title
    plt.rc('axes', labelsize=fs)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fs)    # fontsize of the tick labels
    plt.rc('legend', fontsize=fs)    # legend fontsize
    plt.rc('figure', titlesize=fs)  # fontsize of the figure title


def plot_multi_bars(
    mean_data,
    err_data=None,
    point_data=None,
    add_ss_lines=False,
    colors=None,
    space=0.3,
    space_inner = 0, 
    xticklabels=None,
    ylabel=None,
    ylim=None,
    horizontal_line_pos=0,
    title=None,
    legend_labels=None,
    legend_overlaid=False,
    legend_separate=True,
    add_brackets=None,
    bracket_text=None,
    err_capsize=None,
    fig_size=(12, 6),
):

    """Function to create a bar plot with multiple series of data next to each other.
    Allows adding error bars to each bar and adding significance brackets.

    Args:
        mean_data (array): heights of bars to plot; shape [nlevels1 x nlevels2]
            where nlevels1 is the length of each series (i.e. number of clusters of bars),
            and nlevels2 is the number of series (i.e. number of bars per cluster).

        err_data (array, optional): symmetrical error bar lengths, should be same
            size as mean_data.
        colors (array, optional): list of colors, [nlevels2 x 3] (or [nlevels2 x 4]
            if alpha channel)
        space (float, optional): how big of a space between each bar cluster? max is 0.45.
        xticklabels (1d array or list list, optional): name for each bar "cluster",
            should be [nlevels1] in length.
        ylabel (string, optional): yaxis label
        ylim (2-tuple, optional): yaxis limits
        horizontal_line_pos (float, optional): position to draw a horizontal line on plot.
        title (string, optional): title
        legend_labels (list of strings, optional): labels for each series in the
            plot, [nlevels2] length
        legend_overlaid (boolean, optional): want legend drawn windowed on top of the plot?
        legend_separate (boolean, optional): want legend as a separate axis?
        add_brackets (1d array or list of bools, optional): want to draw brackets over each
            pair of bars? This only applies if nlevels2==2. Must be [nlevels1] in length.
        bracket_text (1d array or list of strings, optional): text to draw over
            each bracket (if drawing brackets.) Must be [nlevels1] in length.
        fig_size (2-tuple, optional): size to draw the entire figure

    """
    assert space < 0.45 and space > 0
    assert len(mean_data.shape) == 2
    nlevels1, nlevels2 = mean_data.shape
    if err_data is not None and len(err_data) == 0:
        err_data = None
    if point_data is not None:
        assert(point_data.shape[1]==nlevels1)
        assert(point_data.shape[2]==nlevels2)
        
    edge_pos = [-0.5 + space, 0.5 - space]
    bar_width = (edge_pos[1] - edge_pos[0] - space_inner*(nlevels2-1)) / nlevels2
    offsets = np.linspace(
        edge_pos[0] + bar_width / 2, edge_pos[1] - bar_width / 2, nlevels2
    )
    if colors is None:
        colors = cm.tab10(np.linspace(0, 1, nlevels2))

    fh = plt.figure(figsize=fig_size)
    ax = plt.subplot(1, 1, 1)
    lh = []
    for ll in range(nlevels2):

        h = plt.bar(
            np.arange(nlevels1) + offsets[ll],
            mean_data[:, ll],
            width=bar_width,
            color=colors[ll, :],
        )
        lh.append(h)
        if err_data is not None:
            assert err_data.shape[0] == nlevels1 and err_data.shape[1] == nlevels2
            plt.errorbar(
                np.arange(nlevels1) + offsets[ll],
                mean_data[:, ll],
                err_data[:, ll],
                ecolor="k",
                zorder=20,
                capsize=err_capsize,
                ls="none",
            )
        if point_data is not None:
            for pp in range(point_data.shape[0]):
                plt.plot(np.arange(nlevels1) + offsets[ll], point_data[pp,:,ll], \
                         '.', color=[0.8, 0.8, 0.8], zorder=15)
                
    if add_ss_lines and point_data is not None:
        for ll in range(nlevels1):           
            for pp in range(point_data.shape[0]):
                plt.plot(ll+offsets, point_data[pp,ll,:],'-', color=[0.8, 0.8, 0.8], zorder=15)
                
    if xticklabels is not None:
        assert len(xticklabels) == nlevels1
        plt.xticks(
            np.arange(nlevels1),
            xticklabels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
    if ylim is not None and ylim != []:
        plt.ylim(ylim)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if horizontal_line_pos is not None:
        plt.axhline(horizontal_line_pos, color=[0.8, 0.8, 0.8])
    if title is not None:
        plt.title(title)

    if legend_overlaid and legend_labels is not None:
        assert len(legend_labels) == nlevels2
        ax.legend(lh, legend_labels)

    if add_brackets is not None:
        assert len(add_brackets) == nlevels1
        assert bracket_text is None or len(bracket_text) == nlevels1
        orig_ylim = ax.get_ylim()
        vert_space = 0.02 * (orig_ylim[1] - orig_ylim[0])
        ymax = orig_ylim[1]

        for xx in np.where(add_brackets)[0]:

            # vertical position of the label is always above the bars,
            # or above the x-axis if bars are negative.
            if err_data is not None:
                max_ht = np.max([np.max(mean_data[xx, :] + err_data[xx, :]), 0])
            else:
                max_ht = np.max([np.max(mean_data[xx, :]), 0])
            brack_bottom = max_ht + vert_space * 2
            brack_top = max_ht + vert_space * 3
            text_lab_ht = max_ht + vert_space * 4
            ymax = np.max([ymax, text_lab_ht + vert_space * 3])

            bracket_x1 = np.mean(offsets[0:int(nlevels2/2)])
            bracket_x2 = np.mean(offsets[int(nlevels2/2):])
            
            plt.plot(
                [xx + bracket_x1, xx + bracket_x1, xx + bracket_x2, xx + bracket_x2],
                [brack_bottom, brack_top, brack_top, brack_bottom],
                "-",
                color="k",
                zorder=20
            )

            if bracket_text is not None:
                ax.annotate(
                    bracket_text[xx],
                    xy=(xx, text_lab_ht),
                    zorder=20,
                    color="k",
                    ha="center",
                    fontsize=12,
                )

        if ylim is None or ylim == []:
            # adjust max y limit so text doesn't get cut off.
            plt.ylim([orig_ylim[0], ymax])

    if legend_separate and legend_labels is not None:
        assert len(legend_labels) == nlevels2
        plt.figure()
        for ll in range(nlevels2):
            plt.plot(0, ll, "-", color=colors[ll, :], linewidth=15)
        plt.legend(legend_labels)

    return fh
       