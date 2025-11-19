from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List

import matplotlib.pyplot as plt

from ..logger import LOGGER


@dataclass
class MultiFilePlotter:
    """
    A class to handle interactive navigation between plots for multiple files.

    Attributes
    ----------
    fpaths : List[Path]
        List of file paths to plot.
    perform_plot : Callable[[Path], None]
        Function to perform the plotting for a given file.
    current_index : int
        The index of the currently displayed file.
    """

    fpaths: List[Path]
    """List of file paths to plot."""
    perform_plot: Callable[[Path], None]
    """Function to perform the plotting for a given file."""
    current_index: int = field(default=0, init=False)

    def _update_plot(self):
        """
        Update the plot for the current file index.

        Clears the current figure and redraws the plot for the file at the
        current index.
        """
        plt.clf()  # Clear the current figure
        fpath = self.fpaths[self.current_index]
        self.perform_plot(fpath)
        plt.title(
            f"File: {fpath.name} ({self.current_index + 1}/{len(self.fpaths)}) [use arrow keys]"
        )
        plt.draw()

    def _on_next(self, event=None):
        """
        Move to the next plot.
        """
        self.current_index = (self.current_index + 1) % len(self.fpaths)
        self._update_plot()

    def _on_previous(self, event=None):
        """
        Move to the previous plot.
        """
        self.current_index = (self.current_index - 1) % len(self.fpaths)
        self._update_plot()

    def _add_ui(self):
        """
        Add UI elements for navigation.

        This method sets up the key press event handler for navigating
        between plots.
        """

        # Wrap the key event handler to avoid an issue due to function being unhashable
        def on_key_wrapper(event):
            if event.key == "right":
                self._on_next()
            elif event.key == "left":
                self._on_previous()
            elif event.key == "escape":
                LOGGER.info("Exiting interactive plotting.")
                plt.close()

        plt.connect("key_press_event", on_key_wrapper)

    def show(self):
        """
        Start the interactive plotting session.

        Opens a Matplotlib figure and allows the user to navigate between
        plots using the left and right arrow keys. Pressing the Escape key
        exits the session.
        """
        self._update_plot()
        self._add_ui()
        plt.show()
