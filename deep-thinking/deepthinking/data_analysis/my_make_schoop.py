""" make_schoop.py
    For generating schoopy plots

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import argparse
from datetime import datetime

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from make_table import get_table


def get_schoopy_plot(all_table, legends, error_bars=True):
    fig, ax = plt.subplots(figsize=(20, 9))

    colors = ['Blues', 'autumn', 'PiYG_r', 'winter', 'viridis']

    for i, table in enumerate(all_table):
        models = set(table.model)
        test_datas = set(table.test_data)
        alphas = set(table.alpha)

        palette = sns.color_palette('husl')[i:i+1]

        g = sns.lineplot(data=table,
                     x="test_iter",
                     y="test_acc_mean",
                     hue="model",
                     # size=3,
                     sizes=(2, 8),
                     style="test_data" if len(test_datas) > 1 else None,
                     palette=palette,
                     dashes=True,
                     units=None,
                     legend=None,
                     label=legends[i],
                     lw=6,
                     ax=ax)

        # leg = g.axes.flat[0].get_legend()
        # new_title = "Methods"
        # leg.set_title(new_title)
        # leg.set_text(legends[i])

        if error_bars and "test_acc_sem" in table.keys():
            for model in models:
                for test_data in test_datas:
                    for alpha in alphas:
                        data = table[(table.model == model) &
                                     (table.test_data == test_data) &
                                     (table.alpha == alpha)]
                        plt.fill_between(data.test_iter,
                                         data.test_acc_mean - data.test_acc_sem,
                                         data.test_acc_mean + data.test_acc_sem,
                                         alpha=0.1, color="k")

    plt.legend(loc='lower right', labels=legends)

    tr = table.max_iters.max()  # training regime number
    ax.fill_between([0, tr], [105, 105], alpha=0.3, label="Training Regime")
    return ax


def main():
    parser = argparse.ArgumentParser(description="Analysis parser")
    parser.add_argument("--filepath_unpruned", type=str, default=None)
    parser.add_argument("--filepath_rand", type=str, default=None)
    parser.add_argument("--filepath_mag", type=str, default=None)
    parser.add_argument("--filepath_snip", type=str, default=None)
    parser.add_argument("--filepath_grasp", type=str, default=None)

    parser.add_argument("--plot_name", type=str, default=None, help="where to save image?")
    parser.add_argument("--plot_title", type=str, default=None, help="title?")

    parser.add_argument("--alpha_list", type=float, nargs="+", default=None,
                        help="only plot models with alphas in given list")
    parser.add_argument("--filter", type=float, default=None,
                        help="cutoff for filtering by training acc?")
    parser.add_argument("--max_iters_list", type=int, nargs="+", default=None,
                        help="only plot models with max iters in given list")
    parser.add_argument("--model_list", type=str, nargs="+", default=None,
                        help="only plot models with model name in given list")
    parser.add_argument("--width_list", type=str, nargs="+", default=None,
                        help="only plot models with widths in given list")
    parser.add_argument("--max", action="store_true", help="add max values to table?")
    parser.add_argument("--min", action="store_true", help="add min values too table?")
    parser.add_argument("--xlim", type=float, nargs="+", default=None, help="x limits for plotting")
    parser.add_argument("--ylim", type=float, nargs="+", default=None, help="y limits for plotting")
    args = parser.parse_args()

    filepaths = []
    legends = []
    if args.filepath_unpruned is not None:
        filepaths.append(args.filepath_unpruned)
        legends.append('Unpruned')
    if args.filepath_rand is not None:
        filepaths.append(args.filepath_rand)
        legends.append('Random')
    if args.filepath_mag is not None:
        filepaths.append(args.filepath_mag)
        legends.append('Magnitude')
    if args.filepath_snip is not None:
        filepaths.append(args.filepath_snip)
        legends.append('SNIP')
    if args.filepath_grasp is not None:
        filepaths.append(args.filepath_grasp)
        legends.append('GraSP')

    if args.plot_name is None:
        now = datetime.now().strftime("%m%d-%H.%M")
        args.plot_name = f"schoop{now}.png"
        plot_title = "Schoopy Plot"

    if args.plot_title is None:
        plot_title = args.plot_name[:-4]
    else:
        plot_title = args.plot_title

    # get table of results
    all_table = []
    for filepath in filepaths:
        print('extracing from {}...'.format(filepath))
        table = get_table(filepath,
                      args.max,
                      args.min,
                      filter_at=args.filter,
                      max_iters_list=args.max_iters_list,
                      alpha_list=args.alpha_list,
                      width_list=args.width_list,
                      model_list=args.model_list)

        # reformat and reindex table for plotting purposes
        table.columns = table.columns.map("_".join)
        table.columns.name = None
        table = table.reset_index()
        print(table.round(2).to_markdown())

        all_table.append(table)

    ax = get_schoopy_plot(all_table, legends)

    ax.legend(fontsize=26, loc="upper left", bbox_to_anchor=(1.0, 0.8))
    x_max = table.test_iter.max()
    # x = np.arange(20, x_max + 1, 10 if (x_max <= 100) else 100)
    x = np.arange(0, x_max + 1, 10 if (x_max <= 100) else 100)
    ax.tick_params(axis="y", labelsize=34)
    ax.set_xticks(x)
    # ax.set_xticklabels(x, fontsize=34, rotation=37)
    ax.set_xticklabels(x, fontsize=34, rotation=0)
    if args.xlim is None:
        ax.set_xlim([x.min() - 0.5, x.max() + 0.5])
    else:
        ax.set_xlim(args.xlim)
    if args.ylim is None:
        ax.set_ylim([0, 103])
    else:
        ax.set_ylim(args.ylim)
    ax.set_xlabel("Test-Time Iterations", fontsize=34)
    ax.set_ylabel("Accuracy (%)", fontsize=34)
    ax.set_title(plot_title, fontsize=34)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.tight_layout()

    plt.savefig(args.plot_name)
    # plt.show()


if __name__ == "__main__":
    main()
