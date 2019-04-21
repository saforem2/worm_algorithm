import numpy as np
import matplotlib.pyplot as plt

def errorbar_plot(values, labels, out_file, limits=None, Tc_line=None,
                  legend_loc=None, markersize=5, num_graphs=None, 
                  reverse_colors=False):
    markers = ['s', 'H', 'd', 'v', 'p', 'P']
    colors = ['#2A9Df8', '#FF920B', '#65e41d', '#be67ff', '#ff7e79', '#959595']
    markeredgecolors = ['#0256a3', '#ed4c18',  '#00B000', '#6633cc',
                        '#ee2324','#1c2022']
    x_values = values['x']
    y_values = values['y']
    assert x_values.shape == y_values.shape, ('x and y data have different'
                                              + ' shapes.')
    fig_labels = labels['fig_labels']
    x_label = labels['x_label']
    y_label = labels['y_label']
    if legend_loc is None:
        legend_loc = 'best'
    if num_graphs is None:
        num_graphs = len(fig_labels)
    else:
        num_graphs = num_graphs
    try:
        y_err = values['y_err']
    except KeyError:
        #  y_err = num_graphs*[np.zeros(y_values.shape)]
        y_err = num_graphs*[0]
    x_lim = limits.get('x_lim')
    y_lim = limits.get('y_lim')
    if reverse_colors:
        colors = colors[:num_graphs][::-1]
        markeredgecolors = markeredgecolors[:num_graphs][::-1]
        markers = markers[:num_graphs][::-1]
    
    fig, ax = plt.subplots()
    if Tc_line is not None:
        ax.axvline(x=Tc_line, linestyle='--', color='k')
    for i in range(num_graphs):
        try:
            ax.errorbar(x_values[i], y_values[i], yerr=y_err[i],
                        label=fig_labels[i], marker=markers[i],
                        markersize=markersize, fillstyle='full',
                        color=colors[i], markeredgecolor=markeredgecolors[i],
                        ls='-', lw=2., elinewidth=2., capsize=2., capthick=2.)
        except ValueError:
            ax.errorbar(x_values, y_values, yerr=y_err,
                        label=fig_labels, marker=markers,
                        markersize=markersize, fillstyle='full',
                        color=colors[0], markeredgecolor=markeredgecolors[0],
                        ls='-', lw=2., elinewidth=2., capsize=2., capthick=2.)

        #    continue
            #  import pdb
            #  pdb.set_trace()
    leg = ax.legend(loc=legend_loc, markerscale=1.5, fontsize=14)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    
    fig.tight_layout()
    print(f"Saving file to: {out_file}")
    fig.savefig(out_file, dpi=400, bbox_inches='tight')
    return fig, ax

