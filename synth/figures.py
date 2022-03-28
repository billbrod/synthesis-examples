#!/usr/bin/env python3
import plenoptic as po
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import numpy as np


def remap_model_name(model_name):
    """Get a nicer name for use in figures."""
    if 'RGC' in model_name:
        model_name = 'fov_lum(' + model_name.split('scaling-')[-1] + ')'
    elif 'V1' in model_name:
        model_name = 'fov_energy(' + model_name.split('scaling-')[-1] + ')'
    model_name = ' '.join([s.capitalize() for s in model_name.split('_')])
    if model_name == 'Mse':
        model_name = 'MSE'
    elif model_name == 'Ssim':
        model_name = 'SSIM'
    elif 'Ps Texture' in model_name:
        model_name = model_name.replace("Ps", "PS")
    elif 'Vgg' in model_name:
        model_name = model_name.replace("Vgg", "VGG")
    return model_name


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
        ax.set_title(remap_model_name(title))
    # this is the space between axes
    space = 10 / imgs[0].shape[-1]
    # this line separates the target image axes from the metamer axes, all the
    # way across the figure
    line = mpl.lines.Line2D([1+.5*space, 1+.5*space], [1+3*space, -2-3*space],
                            color='k',
                            transform=fig.axes[0].transAxes)
    fig.add_artist(line)
    return fig


def _add_initial_noise(img, initial_noise, seed=0, allowed_range=(0, 255)):
    """Add noise to img like during MAD initialization."""
    po.tools.set_seed(seed)
    img = (img + initial_noise * torch.randn_like(img))
    return img.clamp(*allowed_range)


def example_mad_figure(ref_image, image_metric1_min, image_metric1_max,
                       image_metric2_min, image_metric2_max, metric1_name,
                       metric2_name, noise_seed=0, noise_level=20,
                       vrange=(0, 1), rescale=True, annotate=False):
    """Create a figure showing example MAD images for a single model.

    This looks like Figure 8 from [1]_, except that we have a small inset
    showing the difference between each synthesized image and the reference
    image.

    Parameters
    ----------
    ref_image : tensor
        4d tensor of a single image.
    image_metric{1,2}_{max,min} : tensor
        4d tensors, each containing the mad image synthesized to max (or min)
        metric 1 (or 2).
    metric{1,2}_name : str
        str giving the name of metric1 and metric2, for labeling purposes
    noise_seed : int
        RNG seed to use for generating noise for initial image
    nosie_level : float
        Std dev of Gaussian noise to add to ref_image to get the initial image.
    vrange : tuple or str
        Vrange to pass to imshow for the main images (all difference images are
        plotted with 'indep0'). See docstring of pyrtools.imshow for details.
    rescale : bool, optional
        We have an issue with plotting color images with values between 0 and
        255 (matplotlib automatically clips them), so if rescale is True, we
        divide by 255 before plotting each image (make sure vrange is
        appropriately set then!). We do that within the function, rather than
        expecting the user to pass those images to us, because the images
        probably need to lie between 0 and 255 for the generating of the
        initial image.
    annotate : bool, optional
        If True, add boxes around the MAD axes that match the colors used in
        the plenoptic simple MAD example notebook.

    Returns
    -------
    fig : plt.Figure
        the created figure.

    References
    ----------
    .. [1] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
       competition: A methodology for comparing computational models of perceptual
       discriminability. Journal of Vision, 8(12), 1–13.
       http://dx.doi.org/10.1167/8.12.8

    """
    initial_img = _add_initial_noise(ref_image, noise_level, noise_seed)
    ax_kwargs = {'frameon': False, 'xticks': [], 'yticks': []}
    # add 1 so there's a bit of a buffer here
    ax_size = [s//2+1 for s in ref_image.shape[-2:]]
    # this way we have 10 pixels between axes in each direction
    wspace = 10 / ax_size[1]
    hspace = 10 / ax_size[0]
    ppi = 96
    fig = plt.figure(dpi=ppi,
                     figsize=((10*ax_size[1]+wspace*ax_size[1]*9)/ppi,
                              (8*ax_size[0]+hspace*ax_size[0]*7)/ppi))
    # this way the axes use the full figure
    gs = mpl.gridspec.GridSpec(8, 10, wspace=wspace, hspace=hspace, top=1,
                               right=1, left=0, bottom=0)
    # create all the axes
    ref_ax = fig.add_subplot(gs[:2, :2], **ax_kwargs)
    init_ax = fig.add_subplot(gs[3:5, 4:6], **ax_kwargs)
    mad_axes = [
        fig.add_subplot(gs[3:5, 1:3], **ax_kwargs),
        fig.add_subplot(gs[3:5, 7:9], **ax_kwargs),
        fig.add_subplot(gs[:2, 4:6], **ax_kwargs),
        fig.add_subplot(gs[-2:, 4:6], **ax_kwargs),
    ]
    mad_diff_axes = [
        fig.add_subplot(gs[3:4, 0:1], **ax_kwargs),
        fig.add_subplot(gs[3:4, 9:10], **ax_kwargs),
        fig.add_subplot(gs[:1, 3:4], **ax_kwargs),
        fig.add_subplot(gs[-2:-1, 6:7], **ax_kwargs),
    ]
    # plot the images
    if rescale:
        ref_image = ref_image / 255
        initial_img = initial_img / 255
    po.imshow(ref_image, ax=ref_ax, zoom=1, title=None, vrange=vrange,
              as_rgb=True if ref_image.shape[1] == 3 else False)
    po.imshow(initial_img, ax=init_ax, zoom=1, title=None, vrange=vrange,
              as_rgb=True if initial_img.shape[1] == 3 else False)
    images = [image_metric1_min, image_metric1_max, image_metric2_min,
              image_metric2_max]
    for im, ax, diff_ax in zip(images, mad_axes, mad_diff_axes):
        if rescale:
            im = im / 255
        po.imshow(im, ax=ax, zoom=1, title=None, vrange=vrange,
                  as_rgb=True if im.shape[1] == 3 else False)
        # average over the channels so we only have a single image to plot
        po.imshow((im-ref_image).mean(1, True), ax=diff_ax, zoom=.5,
                  title=None, vrange='indep0')
    fontsize = plt.rcParams['font.size'] - 2
    ref_ax.set_title("Reference image", fontsize=fontsize)
    ref_ax.annotate('', xytext=(1, .25), xy=(2.2, -.6),
                    xycoords='axes fraction',
                    arrowprops={'color': 'k',})
    ref_ax.text(1.3, .1, 'Initial distortion', transform=ref_ax.transAxes)
    init_ax.annotate('', xytext=(0, .5), xy=(-.5-2*wspace, .5),
                     xycoords='axes fraction',
                     arrowprops={'edgecolor': 'C1', 'facecolor': 'none',
                                 'shrink': .05})
    init_ax.text(-.3, .55,
                 f'Min {metric1_name}\nfixed {metric2_name}',
                 transform=init_ax.transAxes, ha='center', va='bottom',
                 fontsize=fontsize)
    init_ax.annotate('', xytext=(1, .5), xy=(1.5+2*wspace, .5),
                     xycoords='axes fraction',
                     arrowprops={'color': 'C1', 'shrink': .05})
    init_ax.text(1.3, .55,
                 f'Max {metric1_name}\nfixed {metric2_name}',
                 transform=init_ax.transAxes, ha='center', va='bottom',
                 fontsize=fontsize)
    init_ax.annotate('', xytext=(.5, 1), xy=(.5, 1.5+2*hspace),
                     xycoords='axes fraction',
                     arrowprops={'edgecolor': 'C0', 'shrink': .05,
                                 'facecolor': 'none'})
    init_ax.text(.55, 1.3,
                 f'Min {metric2_name}\nfixed {metric1_name}',
                 transform=init_ax.transAxes, va='center', ha='left',
                 fontsize=fontsize)
    init_ax.annotate('', xytext=(.5, 0), xy=(.5, -.5-2*hspace),
                     xycoords='axes fraction',
                     arrowprops={'color': 'C0', 'shrink': .05})
    init_ax.text(.55, -.3,
                 f'Max {metric2_name}\nfixed {metric1_name}',
                 transform=init_ax.transAxes, va='center', ha='left',
                 fontsize=fontsize)
    if annotate:
        _annotate_example_mad_figure(fig, mad_axes + [init_ax, ref_ax])
    return fig


def _annotate_example_mad_figure(fig, axes):
    """Draw boxes around example mad figure to match simple MAD example."""
    colors = ['C1', 'C1', 'C0', 'C0']
    linestyles = ['--', '-', '--', '-']
    for ax, c, sty in zip(axes, colors, linestyles):
        rect = mpl.patches.Rectangle((0, 0), 1, 1,
                                     edgecolor=c, linestyle=sty, facecolor='none',
                                     transform=ax.transAxes)
        fig.add_artist(rect)
    # initial image
    rect = mpl.patches.Rectangle((0, 0), 1, 1,
                                 edgecolor='k', linestyle='-', facecolor='none',
                                 transform=axes[-2].transAxes)
    fig.add_artist(rect)
    # reference image
    rect = mpl.patches.Rectangle((0, 0), 1, 1,
                                 edgecolor='r', linestyle='-', facecolor='none',
                                 transform=axes[-1].transAxes)
    fig.add_artist(rect)


def mad_noise_levels_figure(ref_image, min_images, max_images, noise_levels,
                            metric_name, noise_seed=0, vrange=(0, 1),
                            rescale=True):
    """Create figure showing synthesized mad images for different noise levels.

    Note that these should be one half of a full MAD set, that is, they should
    all be holding one metric constant while the other is max'ed or min'ed.
    It's assumed that the constant metric will be MSE.

    This looks like Figure 9 from [1]_, except that we have a small inset
    showing the difference between each synthesized image and the reference
    image.

    Parameters
    ----------
    ref_image : tensor
        4d tensor of a single image.
    min_images : tensor
        4d tensor of images where the synthesis metric was minimized.
    max_images : tensor
        4d tensor of images where the synthesis metric was maximized.
    noise_levels : tensor
        4d tensor with the noise levels used to initialize the differnet
        images. Noise values should be along the first dimension.
    metric_name : str
        str giving the name of the synthesis metric, for labeling purposes
    noise_seed : int
        RNG seed to use for generating noise for initial image
    vrange : tuple or str
        Vrange to pass to imshow for the main images (all difference images are
        plotted with 'indep0'). See docstring of pyrtools.imshow for details.
    rescale : bool, optional
        We have an issue with plotting color images with values between 0 and
        255 (matplotlib automatically clips them), so if rescale is True, we
        divide by 255 before plotting each image (make sure vrange is
        appropriately set then!). We do that within the function, rather than
        expecting the user to pass those images to us, because the images
        probably need to lie between 0 and 255 for the generating of the
        initial image.

    Returns
    -------
    fig : plt.Figure
        the created figure.

    References
    ----------
    .. [1] Wang, Z., & Simoncelli, E. P. (2008). Maximum differentiation (MAD)
       competition: A methodology for comparing computational models of perceptual
       discriminability. Journal of Vision, 8(12), 1–13.
       http://dx.doi.org/10.1167/8.12.8

    """
    assert len(max_images) == len(min_images) and len(max_images) == len(noise_levels)
    assert noise_levels.ndim == 4, "noise_levels must be a 4d tensor or torch's implicit reshaping won't work!"
    # if ref image is RGB, then they all will be
    as_rgb = True if ref_image.shape[1] == 3 else False
    # this will add the varying noise levels all to a single sample of noise,
    # but this is what's done in for this synthesis (they all used the same
    # seed)
    initial_images = _add_initial_noise(ref_image, noise_levels, noise_seed)
    if rescale:
        initial_images = initial_images / 255
        min_images = min_images / 255
        max_images = max_images / 255
    fig = po.imshow([initial_images, min_images, max_images],
                    col_wrap=len(noise_levels), vrange=vrange, title=None,
                    as_rgb=as_rgb)
    # label the noise along the top row, taking advantage of the fact that zip
    # stops when the shortest iterable runs out
    for ax, n in zip(fig.axes, noise_levels):
        ax.set_title(f'MSE: {int(n.square().item())}')
    # want these to be the same size as the title
    for i, title in enumerate(['Initial image', f'Min {metric_name}',
                               f'Max {metric_name}']):
        fig.axes[i*len(noise_levels)].set_ylabel(title,
                                                 fontsize=plt.rcParams['axes.titlesize'])
    # this is the space between axes
    space = 10 / ref_image.shape[-1]
    # this line separates the initial image axes from the MAD image axes, all
    # the way across the figure and a bit more
    line = mpl.lines.Line2D([-3*space, 6+8*space], [-space/2, -space/2],
                            color='k', linestyle='--',
                            transform=fig.axes[0].transAxes)
    fig.add_artist(line)
    return fig


def simple_mad_level_set(seed=160):
    """Generate the level set figure from the plenoptic simple MAD notebook.

    Parameters
    ----------
    seed : int, optional
        Seed to determine initial image. Default value is a good one.

    Returns
    -------
    fig : plt.Figure
        Figure containing plot

    """
    img = torch.tensor([.5, .5], dtype=torch.float32).reshape((1, 1, 1, 2))

    def l1_norm(x, y):
        return torch.norm(x-y, 1)
    metrics = [po.tools.optim.l2_norm, l1_norm]
    all_mad = {}

    # this gets us all four possibilities
    for t, (m1, m2) in itertools.product(['min', 'max'], zip(metrics, metrics[::-1])):
        name = f'{t}_{m1.__name__.capitalize()}'
        po.tools.set_seed(seed)
        all_mad[name] = po.synth.MADCompetition(img, m1, m2, t, metric_tradeoff_lambda=1e4)
        optim = torch.optim.Adam([all_mad[name].synthesized_signal], lr=.0001)
        print(f"Synthesizing {name}")
        all_mad[name].synthesize(store_progress=True, max_iter=2000, optimizer=optim,
                                 stop_criterion=1e-6)

    # double-check that these are all equal.
    assert all([torch.allclose(all_mad['min_L2_norm'].initial_signal, v.initial_signal) for v in all_mad.values()])

    pal = {'l1_norm': 'C0', 'l2_norm': 'C1'}

    l1 = po.to_numpy(torch.norm(all_mad['max_L2_norm'].reference_signal - all_mad['max_L2_norm'].initial_signal, 1))
    l2 = po.to_numpy(torch.norm(all_mad['max_L2_norm'].reference_signal - all_mad['max_L2_norm'].initial_signal, 2))
    ref = po.to_numpy(all_mad['max_L2_norm'].reference_signal.squeeze())
    init = po.to_numpy(all_mad['max_L2_norm'].initial_signal.squeeze())

    def circle(origin, r, n=1000):
        theta = 2*np.pi/n*np.arange(0, n+1)
        return np.array([origin[1]+r*np.cos(theta), origin[0]+r*np.sin(theta)])

    def diamond(origin, r, n=1000):
        theta = 2*np.pi/n*np.arange(0, n+1)
        rotation = np.pi/4
        square_correction = (np.abs(np.cos(theta-rotation)-np.sin(theta-rotation)) + np.abs(np.cos(theta-rotation)+np.sin(theta-rotation)))
        square_correction /= square_correction[0]
        r = r / square_correction
        return np.array([origin[1]+r*np.cos(theta), origin[0]+r*np.sin(theta)])
    l2_level_set = circle(ref, l2,)
    l1_level_set = diamond(ref, l1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    size = plt.rcParams['lines.markersize']**2
    ax.scatter(*ref, label='reference', c='r', s=size)
    ax.scatter(*init, label='initial', c='k', s=size)
    ax.plot(*l1_level_set, pal['l1_norm']+'--', label='L1 norm level set')
    ax.plot(*l2_level_set, pal['l2_norm']+'--', label='L2 norm level set')
    size = (plt.rcParams['lines.markersize']/1.5)**2
    for k, v in all_mad.items():
        ec = pal[v.fixed_metric.__name__]
        fc = 'none' if 'min' in k else ec
        ax.scatter(*v.synthesized_signal.squeeze().detach(), fc=fc, ec=ec,
                   label=k.replace('_', ' '), s=size)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set(xlabel='Pixel 1 value', ylabel='Pixel 2 value')

    return fig
