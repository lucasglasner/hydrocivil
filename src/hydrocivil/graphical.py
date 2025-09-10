'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner,
 Modified time: 2025-09-10 16:40:13
 Description:
 Dependencies:
'''

from .misc import minradius_from_centroid

# ---------------------------------------------------------------------------- #


def centergraph2polygon(polygon, ax):
    """Centers the plot view on the given polygon.

    Sets the x and y limits of the provided matplotlib axis (ax) so that the
    polygon is centered and fits within the view, based on its centroid and
    minimum radius.

    Args:
        polygon: The polygon object to center the view on.
        ax: The matplotlib axis to adjust.
    """
    ds, c = minradius_from_centroid(polygon)
    cx, cy = c.x.item(), c.y.item()
    ax.set_xlim(cx - ds*0.99, cx + ds*0.99)
    ax.set_ylim(cy - ds*0.99, cy + ds*0.99)


def add_colorbar(mappeable, fig, ax, orientation='vertical', **kwargs):
    """Adds a colorbar to the figure next to the given axis.

    Creates and positions a colorbar for the provided mappable object
    (e.g., image or collection) in the specified orientation ('vertical' or
    'horizontal') relative to the axis.

    Args:
        mappeable: The matplotlib mappable object to which the colorbar applies.
        fig: The matplotlib figure to add the colorbar to.
        ax: The axis next to which the colorbar will be placed.
        orientation (str, optional): Orientation of the colorbar
            ('vertical' or 'horizontal'). Defaults to 'vertical'.
        **kwargs: Additional keyword arguments passed to fig.colorbar().

    Raises:
        ValueError: If orientation is not 'vertical' or 'horizontal'.

    Returns:
        The created colorbar object.
    """
    box = ax.get_position()
    if orientation == 'vertical':
        cax = fig.add_axes([box.xmax + 0.01, box.ymin, 0.02, box.height])
    elif orientation == 'horizontal':
        cax = fig.add_axes([box.xmin, box.ymin - 0.07, box.width, 0.02])
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")
    cbar = fig.colorbar(mappeable, cax=cax, orientation=orientation, **kwargs)
    return cbar
