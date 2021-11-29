#!/usr/bin/env python3
import plenoptic as po


def example_metamer_figure(ref_images, vrange=(0, 1), **kwargs):
    """Create a figure showing example metamers for arbitrary number of models.

    Parameters
    ----------
    ref_images : tensor
        4d tensor of images
    vrange : tuple or str
        Vrange to pass to imshow. See docstring of pyrtools.imshow for details.
    kwargs :
        Keys should be the model names and values should be 4d tensor of
        images, same number of batches as ref_images, containing the metamers
        for that model corresponding to that ref_image.

    Returns
    -------
    fig : plt.Figure
        the created figure.

    """
    titles = ['Target images'] + list(kwargs.keys())
    # this gets all the images in a single list, interleaving them so we have
    # the first ref_image, first image from the first model (whose name is
    # title[1]), etc. this way of going through the images reduces them to 3d
    # tensors, so we unsqueeze one time so imshow (which expects 4d tensors) is
    # happy
    imgs = [z_.unsqueeze(0) for z in zip(ref_images, *[kwargs[k]
                                                       for k in titles[1:]])
            for z_ in z]
    col_wrap = len(titles)
    fig = po.imshow(imgs, title=None, vrange=vrange, col_wrap=col_wrap,
                    as_rgb=True, zoom=1)
    for ax, title in zip(fig.axes[:col_wrap], titles):
        ax.set_title(title)
    return fig
