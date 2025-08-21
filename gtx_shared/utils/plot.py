import os
import sys
from gtx_shared.base import SingletonMeta
from gtx_shared.utils import io, GtxScreenLogger, QError, QWarning, QNote

try:
    import matplotlib.pyplot as plt
except ImportError:
    _enable_plot = False
else:
    _enable_plot = True


class Plotter(metaclass=SingletonMeta):
    counter = 0
    figure_dict = {}

    def __init__(self):
        if not _enable_plot:
            GtxScreenLogger().warning2user(
                QWarning.MATPLOTLIB, "Please install matplotlib for visualization."
            )
            sys.exit(1)
        self._dir = ".gtx_quant_stat_figures"
        io.create_work_dir(self._dir)

    def plot_hist(self, name, data):
        plot_title = "_".join([name, "hist"])
        if plot_title in self.figure_dict:
            GtxScreenLogger().info("Finish visualization.")
            sys.exit(0)

        self.figure_dict[plot_title] = True
        plt.figure(self.counter)
        self.counter += 1
        plt.hist(data, bins=20, facecolor="blue", edgecolor="black", alpha=0.7)
        plt.xlabel("value")
        plt.ylabel("frequency")
        plt.title(plot_title)
        plot_title = plot_title.replace("/", "_")
        plt.savefig(os.path.join(self._dir, ".".join([plot_title, "svg"])))
        plt.close()
