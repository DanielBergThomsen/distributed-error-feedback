"""Plotting utilities for visualization of results.

This module provides functions for creating standardized plots including contour plots
and line plots with consistent styling and formatting.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib import rc
from matplotlib import colors

def set_matplotlib_style():
    """Sets up standard matplotlib styling for consistent plots.
    
    Configures:
        - Serif font family with Computer Modern
        - LaTeX text rendering
        - AMS math package for mathematical notation
    """
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('text.latex', preamble=r'\usepackage{amsmath}')


def standard_textbox(text, override_kwargs=None):
    """Creates standardized textbox parameters for plot annotations.

    Args:
        text: Text to display in the textbox.
        override_kwargs: Dictionary of parameters to override defaults.

    Returns:
        dict: Textbox parameters for matplotlib text function.
    """
    props = dict(
        boxstyle='round',
        facecolor=(0.93, 0.93, 0.93),
        alpha=1.0,
        linewidth=0.5,
        edgecolor='gray'
    )
    txtbox_args = {
        'x': 0.05,
        'y': 0.92,
        's': text,
        'ha': 'left',
        'va': 'top',
        'bbox': props,
        'fontsize': 20
    }
    if override_kwargs is not None:
        txtbox_args.update(override_kwargs)
    return txtbox_args

def contour_plot(data, cmap="RdBu_r", 
                figsize=(5, 4), dpi=150, 
                lineplot_data=None,
                txtbox_kwargs=None, 
                increasing_colorbar=True,
                colorbar_kwargs=None,
                overlay_kwargs=None,
                vmin=None, vmax=None, center=True,
                ax=None, save_file=None, add_colorbar=True, 
                colorbar_label=None, xlabel=None, ylabel=None, title=None, 
                return_plt=False, label_size=14, tick_size=12, 
                inset_zoom_regions=None, **kwargs):
    """Creates a contour plot with standardized styling and formatting.

    Args:
        data: List of tuples (xs, ys, zs) or (xs, ys, zs, overlay) for each subplot.
        cmap: Colormap to use for the contour plot.
        figsize: Tuple of (width, height) for the figure.
        dpi: Dots per inch for the figure.
        lineplot_data: Optional list of (x, y, style_dict) tuples for line overlays.
        txtbox_kwargs: Optional list of textbox parameters for each subplot.
        increasing_colorbar: Whether colorbar values increase left to right.
        colorbar_kwargs: Additional arguments for colorbar customization.
        overlay_kwargs: Arguments for overlay contour plot if provided.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        center: Whether to center the colormap at zero.
        ax: Unused; contour_plot currently creates its own figure and axes.
        save_file: Optional path to save the figure.
        add_colorbar: Whether to add a colorbar.
        colorbar_label: Label for the colorbar.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        title: Title for the plot.
        return_plt: Whether to return the plot object instead of showing.
        label_size: Font size for axis labels.
        tick_size: Font size for tick labels.
        inset_zoom_regions: Optional list of zoom regions for inset axes. Each region should be 
                           a tuple (xmin, xmax, ymin, ymax) or None for no inset. If a single tuple 
                           is provided, it will be applied to all subplots.
        **kwargs: Additional arguments passed to contourf.

    Returns:
        If return_plt is True, returns the last contour set created.
        Otherwise, displays the plot.
    """
    # Initialize default kwargs
    colorbar_kwargs = colorbar_kwargs or {}
    overlay_kwargs = overlay_kwargs or {}

    # Standardize matplotlib settings
    set_matplotlib_style()

    # Create figure and axes
    fig, axes = plt.subplots(1, len(data), figsize=figsize, dpi=dpi, 
                            sharex=True, sharey=True)

    # Determine color scaling
    all_zs = np.concatenate([d[2] for d in data])
    vmin = np.nanmin(all_zs) if vmin is None else vmin
    vmax = np.nanmax(all_zs) if vmax is None else vmax

    # Create normalization
    if center:
        abs_max = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Handle inset_zoom_regions parameter
    if inset_zoom_regions is not None:
        if isinstance(inset_zoom_regions, tuple) and len(inset_zoom_regions) == 4:
            # Single zoom region applied to all subplots
            inset_zoom_regions = [inset_zoom_regions] * len(data)
        elif len(inset_zoom_regions) != len(data):
            raise ValueError(f"inset_zoom_regions must have length {len(data)} or be a single tuple")

    # Plot contours
    contours = []
    for i, d in enumerate(data):
        # Extract data
        if len(d) == 3:
            xs, ys, zs = d
            overlay = None
        elif len(d) == 4:
            xs, ys, zs, overlay = d

        ax = axes[i] if len(data) > 1 else axes

        # Plot overlay if provided
        if overlay is not None:
            ax.contourf(xs, ys, overlay, **overlay_kwargs)

        # Plot main contour
        contour = ax.contourf(xs, ys, zs, cmap=cmap, norm=norm, **kwargs)
        contour.set_edgecolor('face')
        contours.append(contour)

        # Add lineplot if provided
        if lineplot_data is not None:
            ax.plot(lineplot_data[i][0], lineplot_data[i][1], **lineplot_data[i][2])

        # Set labels and title
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=label_size)
        ax.set_xlabel(xlabel, fontsize=label_size)
        ax.set_title(title)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)

        # Add textbox if provided
        if txtbox_kwargs is not None:
            ax.text(**txtbox_kwargs[i], transform=ax.transAxes)

        # Create inset axes if specified
        if inset_zoom_regions is not None and inset_zoom_regions[i] is not None:
            xmin, xmax, ymin, ymax = inset_zoom_regions[i]
            
            # Create inset axes using the simpler approach
            axins = ax.inset_axes(
                [0.5, 0.5, 0.47, 0.47],
                xlim=(xmin, xmax), ylim=(ymin, ymax), 
                xticklabels=[], yticklabels=[]
            )
            
            # Plot the same data in the inset
            if overlay is not None:
                axins.contourf(xs, ys, overlay, **overlay_kwargs)
            axins.contourf(xs, ys, zs, cmap=cmap, norm=norm, **kwargs)
            axins.set_edgecolor('face')
            
            # Indicate the zoom region on the main plot
            ax.indicate_inset_zoom(axins, edgecolor="black")

    # Add colorbar
    if add_colorbar:

        # Add a single colorbar for all subplots
        contour_for_colorbar = contours[np.nanargmax([np.nanmax(d[2]) for d in data])] if increasing_colorbar else contours[np.nanargmin([np.nanmin(d[2]) for d in data])]
        cbar = fig.colorbar(contour_for_colorbar, ax=axes, 
                            cmap=cmap,
                            norm=norm,
                            orientation='horizontal', 
                            fraction=0.1, pad=0.1, location='top', 
                            **colorbar_kwargs)
        if colorbar_label:
            cbar.set_label(colorbar_label, fontsize=label_size, labelpad=10)

        if not increasing_colorbar:
            cbar.ax.invert_xaxis()

    # Save figure if requested
    if save_file is not None:
        fig.savefig(save_file, bbox_inches='tight')

    # Return or show
    if return_plt:
        return contour
    else:
        plt.show()

def line_plot(data, figsize=(4, 4), dpi=150, 
              txtbox_kwargs=None, 
              ax=None, 
              xlabel=None, ylabel=None, title=None,
              plt_legend=False, label_size=14,
              tick_size=12,
              save_file=None, return_plt=False, **kwargs):
    """Creates a line plot with standardized styling and formatting.

    Args:
        data: List of tuples (xs, ys, style_dict) for each line to plot.
        figsize: Tuple of (width, height) for the figure.
        dpi: Dots per inch for the figure.
        txtbox_kwargs: Optional textbox parameters.
        ax: Optional existing axes to plot on.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        title: Title for the plot.
        plt_legend: Whether to show the legend.
        label_size: Font size for axis labels.
        tick_size: Font size for tick labels.
        save_file: Optional path to save the figure.
        return_plt: Whether to return the plot object instead of showing.
        **kwargs: Additional arguments passed to plot.

    Returns:
        If return_plt is True, returns the axes object.
        Otherwise, displays the plot.
    """
    set_matplotlib_style()

    # Create or use existing axes
    if ax is None:
        ax = plt.figure(figsize=figsize, dpi=dpi).add_subplot(111)

    # Plot each line
    for (xs, ys, subplot_kwargs) in data:
        ax.plot(xs, ys, **subplot_kwargs, **kwargs)

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    ax.set_title(title)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)

    # Add legend if requested
    if plt_legend:
        ax.legend()

    # Add textbox if provided
    if txtbox_kwargs is not None:
        ax.text(**txtbox_kwargs, transform=ax.transAxes)

    # Save figure if requested
    if save_file is not None:
        # Save a standalone copy when the axes belongs to a GridSpec.
        if ax.get_gridspec() is not None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            new_ax = fig.add_subplot(111)
            
            # Recreate plot
            for (xs, ys, subplot_kwargs) in data:
                new_ax.plot(xs, ys, **subplot_kwargs, **kwargs)
            
            # Set labels and title
            new_ax.set_xlabel(xlabel, fontsize=label_size)
            new_ax.set_ylabel(ylabel, fontsize=label_size)
            new_ax.set_title(title)
            new_ax.tick_params(axis='both', which='major', labelsize=tick_size)
            
            if plt_legend:
                new_ax.legend()
            if txtbox_kwargs is not None:
                new_ax.text(**txtbox_kwargs, transform=new_ax.transAxes)
            
            # Save and cleanup
            fig.tight_layout()
            fig.savefig(save_file, bbox_inches='tight')
            plt.close(fig)
        else:
            # Save original figure
            ax.figure.tight_layout()
            ax.figure.savefig(save_file, bbox_inches='tight')

    # Return or show
    if return_plt:
        return ax
    else:
        plt.show()
