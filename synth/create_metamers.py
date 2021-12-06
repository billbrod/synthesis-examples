#!/usr/bin/env python3
"""create metamers

"""
from . import utils
import re
import torch
import torchvision
import plenoptic as po
import pyrtools as pt
import time
import numpy as np
import os.path as op
import imageio
import matplotlib as mpl
from skimage import color
import warnings
import sys
import yaml
with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
    fov_path = yaml.safe_load(f)['FOVEATED_METAMERS_PATH']
sys.path.append(op.join(fov_path, 'extra_packages'))
import plenoptic_part as pop


def setup_initial_image(initial_image_type, image):
    r"""Set up the initial image.

    Parameters
    ----------
    initial_image_type : {'white', 'pink', 'gray', 'blue'} or path to file
        What to use for the initial image. If 'white', we use white
        noise. If 'pink', we use pink noise
        (``pyrtools.synthetic_images.pink_noise(fract_dim=1)``). If
        'blue', we use blue noise
        (``pyrtools.synthetic_images.blue_noise(fract_dim=1)``). If
        'gray', we use a flat image with values of .5 everywhere. If
        path to a file, that's what we use as our initial image (and so
        the seed will have no effect on this).
    image : torch.Tensor
        The reference image tensor

    Returns
    -------
    initial_image : torch.Tensor
        The initial image to pass to metamer.synthesize

    """
    if initial_image_type == 'white':
        initial_image = torch.rand_like(image, dtype=torch.float32)
    elif initial_image_type == 'gray':
        initial_image = .5 * torch.ones_like(image, dtype=torch.float32)
    elif initial_image_type == 'pink':
        # this `.astype` probably isn't necessary, but just in case
        initial_image = pt.synthetic_images.pink_noise(image.shape[-2:]).astype(np.float32)
        # need to rescale this so it lies between 0 and 1
        initial_image += np.abs(initial_image.min())
        initial_image /= initial_image.max()
        initial_image = torch.Tensor(initial_image).unsqueeze(0).unsqueeze(0)
    elif initial_image_type == 'blue':
        # this `.astype` probably isn't necessary, but just in case
        initial_image = pt.synthetic_images.blue_noise(image.shape[-2:]).astype(np.float32)
        # need to rescale this so it lies between 0 and 1
        initial_image += np.abs(initial_image.min())
        initial_image /= initial_image.max()
        initial_image = torch.Tensor(initial_image).unsqueeze(0).unsqueeze(0)
    elif op.isfile(initial_image_type):
        warnings.warn("Using image %s as initial image!" % initial_image_type)
        initial_image = imageio.imread(initial_image_type)
        initial_image = convert_im_to_float(initial_image)
        initial_image = torch.tensor(initial_image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    else:
        raise Exception("Don't know how to handle initial_image_type %s! Must be one of {'white',"
                        " 'gray', 'pink', 'blue'}" % initial_image_type)
    return torch.nn.Parameter(initial_image)


def setup_image(image, n_channels=1):
    r"""Set up the image.

    We load in the image, if it's not already done so (converting it to
    gray-scale in the process), make sure it lies between 0 and 1, and
    make sure it's a tensor of the correct type and specified device

    Parameters
    ----------
    image : str or array_like
        Either the path to the file to load in or the loaded-in
        image. If array_like, we assume it's already 2d (i.e.,
        grayscale)
    n_channels : int, optional
        How many channels the image should have. Will always be grayscale, but
        we may duplicate along one of the channels so we can feed this into
        VGG16

    Returns
    -------
    image : torch.Tensor
        The image tensor, ready to go

    """
    if isinstance(image, str):
        print("Loading in reference image from %s" % image)
        image = imageio.imread(image)
    if image.dtype == np.uint8:
        warnings.warn("Image is int8, with range (0, 255)")
        image = utils.convert_im_to_float(image)
    elif image.dtype == np.uint16:
        warnings.warn("Image is int16 , with range (0, 65535)")
        image = utils.convert_im_to_float(image)
    else:
        warnings.warn("Image is float 32, so we assume image range is (0, 1)")
        if image.max() > 1:
            raise Exception("Image is neither int8 nor int16, but its max is greater than 1!")
    # we use skimage.color.rgb2gray in order to handle rgb
    # correctly. this uses the ITU-R 601-2 luma transform, same as
    # matlab. we do this after the above, because it changes the image
    # dtype to float32
    if image.ndim == 3:
        # then it's a color image, and we need to make it grayscale
        image = color.rgb2gray(image)
    image = torch.tensor(image, dtype=torch.float32)
    while image.ndimension() < 4:
        image = image.unsqueeze(0)
    return image.repeat(1, n_channels, 1, 1)


def setup_model(model_name, image, min_ecc, max_ecc, cache_dir,
                normalize_dict=None):
    r"""Set up the model.

    We initialize the model, with the specified parameters, and return it.

    `model_name` must be 'VGG16_poolN' (where N is an int between 1 and 5),
    'PSTexture' or one of our foveated models. If a foveated model, it must be
    constructed of several parts, for which you have several chocies:
    `'{visual_area}{options}_{window_type}_scaling-{scaling}'`:
    - `visual_area`: which visual area we're modeling.`'RGC'` (retinal
      ganglion cells, `plenoptic.simul.PooledRGC` class) or
      `'V1'` (primary visual cortex,
      `plenoptic.simul.PrimaryVisualCortex` class)
    - `options`: you can additionally include the following strs,
      separated by `_`:
      - `'norm'`: if included, we normalize the models' `cone_responses`
        and (if V1) `complex_cell_responses` attributes. In this case,
        `normalize_dict` must also be set (and include those two
        keys). If not included, the model is not normalized
        (normalization makes the optimization easier because the
        different scales of the steerable pyramid have different
        magnitudes).
      - `s#` (V1 only), where `#` is an integer. The number of scales to
        inlude in the steerable pyramid that forms the basis fo the `V1`
        models. If not included, will use 4.
    - `window_type`: `'gaussian'` or `'cosine'`. whether to build the
      model with gaussian or raised-cosine windows. Regardless, scaling
      will always give the ratio between the FWHM and eccentricity of
      the windows, but the gaussian windows are much tighter packed, and
      so require more windows (and thus more memory), but also seem to
      have fewer aliasing issues.
    - `scaling`: float giving the scaling values of these models

    The recommended model_name values that correspond to our foveated models
    are: `RGC_norm_gaussian_scaling-{scaling}`,
    `V1_norm_s6_gaussian_scaling-{scaling}` (pick whatever scaling value you
    like).

    For the other model_name choices:

    - PSTexture: the Portilla-Simoncelli texture stats with n_scales=4,
      n_orientations=4, spatial_corr_width=9, use_true_correlations=True

    - VGG16_poolN: pretrained VGG16 from torchvision, through Nth max pooling
      layer (where N is an int from 1 to 5)

    Parameters
    ----------
    model_name : str
        str specifying which of the models we should initialize. See above for
        more details.
    image : torch.tensor or np.array
        The image we will call the model on. This is only necessary
        because we need to know how big it is; we just use its shape
    min_ecc : float
        The minimum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    max_ecc : float
        The maximum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    normalize_dict : dict or None, optional
        If a dict, should contain the stats to use for normalization. If
        None, we don't normalize. This can only be set (and must be set)
        if the model is "V1_norm". In any other case, we'll throw an
        Exception.

    Returns
    -------
    model : torch.nn.Module
        A ventral stream model, ready to use

    """
    if 'gaussian' in model_name:
        window_type = 'gaussian'
        t_width = None
        std_dev = 1
    elif 'cosine' in model_name:
        window_type = 'cosine'
        t_width = 1
        std_dev = None
    if model_name.startswith('RGC') or model_name.startswith('V1'):
        model_name, scaling = re.findall('([a-zA-z_0-9]+)_scaling-([0-9.]+)', model_name)[0]
        scaling = float(scaling)
        if 'norm' not in model_name:
            if normalize_dict:
                raise Exception(f"Cannot normalize model {model_name} (norm must be part of model_name to do so)!")
            normalize_dict = {}
        if not normalize_dict and 'norm' in model_name:
            raise Exception(f"If model_name is {model_name}, normalize_dict must be set!")
        if model_name.startswith('RGC'):
            model = pop.PooledRGC(scaling, image.shape[-2:],
                                  min_eccentricity=min_ecc,
                                  max_eccentricity=max_ecc,
                                  window_type=window_type,
                                  transition_region_width=t_width,
                                  cache_dir=cache_dir,
                                  std_dev=std_dev,
                                  normalize_dict=normalize_dict)
        elif model_name.startswith('V1'):
            try:
                num_scales = int(re.findall('_s([0-9]+)_', model_name)[0])
            except (IndexError, ValueError):
                num_scales = 4
            try:
                moments = int(re.findall('_m([0-9]+)_', model_name)[0])
                moments = list(range(2, moments+1))
            except (IndexError, ValueError):
                moments = []
            model = pop.PooledV1(scaling, image.shape[-2:],
                                 min_eccentricity=min_ecc,
                                 max_eccentricity=max_ecc,
                                 std_dev=std_dev,
                                 transition_region_width=t_width,
                                 cache_dir=cache_dir,
                                 normalize_dict=normalize_dict,
                                 num_scales=num_scales,
                                 window_type=window_type,
                                 moments=moments)
    elif model_name == 'PSTexture':
        model = po.simul.PortillaSimoncelli(image.shape[-2:], n_scales=4,
                                            n_orientations=4,
                                            spatial_corr_width=9,
                                            use_true_correlations=True)
    elif 'VGG16' in model_name:
        model = torchvision.models.vgg16(pretrained=True).eval()
        # through the first max pooling layer
        if 'pool1' in model_name:
            model = torch.nn.Sequential(*list(model.children())[0][:5])
        # through the second max pooling layer
        elif 'pool2' in model_name:
            model = torch.nn.Sequential(*list(model.children())[0][:10])
        # etc
        elif 'pool3' in model_name:
            model = torch.nn.Sequential(*list(model.children())[0][:17])
        elif 'pool4' in model_name:
            model = torch.nn.Sequential(*list(model.children())[0][:24])
        elif 'pool5' in model_name:
            model = torch.nn.Sequential(*list(model.children())[0][:31])
        else:
            raise Exception(f"Don't know what to do with model_name {model_name}!")
    else:
        raise Exception("Don't know how to handle model_name %s" % model_name)
    return model


def setup_device(*args, gpu_id=None):
    r"""Setup device and get everything onto it

    This simple function checks whether ``torch.cuda.is_available()``
    and ``gpu_id`` is not None. If not, we use the cpu as the device

    We then call a.to(device) for every a in args (so this can be called
    with an arbitrary number of objects, each of which just needs to have
    .to method).

    Note that we always return a list (even if you only pass one item),
    so if you pass a single object, you'll need to either grab it
    specifically, either by doing ``im = setup_device(im,
    gpu_id=0)[0]`` or ``im, = setup_device(im)`` (notice the
    comma).

    Parameters
    ----------
    args :
        Some number of torch objects that we want to get on the proper
        device
    gpu_id : int or None, optional
        If not None, the GPU we will use. If None, we run on CPU. We
        don't do anything clever to handle that here, but the
        contextmanager utils.get_gpu_id does, so you should use that to
        make sure you're using a GPU that exists and is available (see
        Snakefile for example). Note that, to set this,
        you must set it as a keyword, i.e., ``setup_device(im, 0)``
        won't work but ``setup_device(im, gpu_id=True)`` will (this is
        because the ``*args`` in our function signature will greedily
        grab every non-keyword argument).

    Returns
    -------
    args : list
        Every item we were passed in arg, now on the proper device

    """
    if gpu_id is not None:
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available but gpu_id is not None!")
        device = torch.device("cuda:%s" % gpu_id)
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print("On device %s" % device)
    if dtype is not None:
        print("Changing dtype to %s" % dtype)
        args = [a.to(dtype) for a in args]
    return [a.to(device) for a in args]


def add_center_to_image(model, image, reference_image):
    r"""Add the reference image center to an image

    The VentralStream class of models will do nothing to the center of
    the image (they don't see the fovea), so we add the fovea to the
    image before synthesis.

    Parameters
    ----------
    model : plenoptic.simul.VentralStream
        The model used to create the metamer. Specifically, we need its
        windows attribute
    image : torch.Tensor
        The image to add the center back to
    reference_image : torch.Tensor
        The reference/target image for synthesis
        (``metamer.base_signal``); the center comes from this image.

    Returns
    -------
    recentered_image : torch.Tensor
        ``image`` with the reference image center added back in

    """
    model(image)
    rep = model.representation['mean_luminance']
    dummy_ones = torch.ones_like(rep)
    windows = model.PoolingWindows.project(dummy_ones).squeeze().to(image.device)
    # these aren't exactly zero, so we can't convert it to boolean
    anti_windows = 1 - windows
    return ((windows * image) + (anti_windows * reference_image))


def save(save_path, metamer):
    """Save Metamer object and its outputs.

    We save the object itself, plus:
    - The finished metamer in its original float32 format (with
      values between 0 and 1, as a numpy array), at
      ``os.path.splitext(save_path)[0] + "_metamer.npy"``.
    - The finished metamer 8-bit image, at
      ``os.path.splitext(save_path)[0] + "_metamer.png"``.
    - Picture showing synthesis progress summary at
      ``os.path.splitext(save_path)[0] + "_synthesis.png"``.

    Parameters
    ----------
    save_path : str
        The path to save the MADCompetition object at, which we use as a
        starting-point for the other save paths
    metamer : plenoptic.synth.MADCompetition
        The Metamer object after synthesis

    """
    if metamer.model(metamer.synthesized_signal).ndimension() == 4:
        # these VGG representations have many channels, plotting them all takes
        # too much time
        plot_model_response_error = False
    else:
        plot_model_response_error = True
    print("Saving at %s" % save_path)
    if hasattr(metamer.model, 'PoolingWindows'):
        # If we're using one of our foveated models, we add the center back at
        # the end because our gradients are not exactly zero in the center, and
        # thus those pixels end up getting moved around a little bit. Not
        # entirely sure why, but probably not worth tracing down, since we're
        # interested in the periphery
        metamer.synthesized_signal = torch.nn.Parameter(add_center_to_image(metamer.model,
                                                                            metamer.synthesized_signal,
                                                                            metamer.target_signal))
    metamer.save(save_path)
    # save png of mad
    metamer_path = op.splitext(save_path)[0] + "_metamer.png"
    metamer_image = po.to_numpy(metamer.synthesized_signal).squeeze()
    print("Saving metamer float32 array at %s" % metamer_path.replace('.png', '.npy'))
    np.save(metamer_path.replace('.png', '.npy'), metamer_image)
    print("Saving metamer image at %s" % metamer_path)
    if metamer_image.ndim == 3:
        # then this is an RGB, from the VGG16 models, and we want to move the
        # channels dim to the last dimension
        metamer_image = metamer_image.transpose(1, 2, 0)
    imageio.imwrite(metamer_path, utils.convert_im_to_int(metamer_image))
    synthesis_path = op.splitext(save_path)[0] + "_synthesis.png"
    print(f"Saving synthesis image at {synthesis_path}")
    fig, _ = po.synth.metamer.plot_synthesis_status(metamer,
                                                    model_response_error=plot_model_response_error)
    fig.savefig(synthesis_path)


def main(model_name, image, seed=0, min_ecc=.5, max_ecc=15, learning_rate=1,
         max_iter=100, stop_criterion=1e-4, stop_iters_to_check=50,
         save_path=None, initial_image='white', gpu_id=None, cache_dir=None,
         normalize_dict=None, optimizer='Adam', loss_func='mse',
         range_penalty_lambda=.1, coarse_to_fine=False,
         coarse_to_fine_kwargs={}, continue_path=None, num_threads=None):
    r"""Create metamer images.

    Given a model_name, model parameters, a target image, and some
    optimization parameters, we do our best to synthesize a metamer,
    saving the outputs after it finishes.

    `model_name` must either a model, 'VGG16_poolN' (where N is an int between
    1 and 5), 'PSTexture' or one of our foveated models. If a foveated model,
    it must be constructed of several parts, for which you have several
    chocies:
    `'{visual_area}{options}_{window_type}_scaling-{scaling}'`:
    - `visual_area`: which visual area we're modeling.`'RGC'` (retinal
      ganglion cells, `plenoptic.simul.PooledRGC` class) or
      `'V1'` (primary visual cortex,
      `plenoptic.simul.PrimaryVisualCortex` class)
    - `options`: you can additionally include the following strs,
      separated by `_`:
      - `'norm'`: if included, we normalize the models' `cone_responses`
        and (if V1) `complex_cell_responses` attributes. In this case,
        `normalize_dict` must also be set (and include those two
        keys). If not included, the model is not normalized
        (normalization makes the optimization easier because the
        different scales of the steerable pyramid have different
        magnitudes).
      - `s#` (V1 only), where `#` is an integer. The number of scales to
        inlude in the steerable pyramid that forms the basis fo the `V1`
        models. If not included, will use 4.
    - `window_type`: `'gaussian'` or `'cosine'`. whether to build the
      model with gaussian or raised-cosine windows. Regardless, scaling
      will always give the ratio between the FWHM and eccentricity of
      the windows, but the gaussian windows are much tighter packed, and
      so require more windows (and thus more memory), but also seem to
      have fewer aliasing issues.
    - `scaling`: float giving the scaling values of these models

    The recommended model_name values that correspond to our foveated models
    are: `RGC_norm_gaussian_scaling-{scaling}`,
    `V1_norm_s6_gaussian_scaling-{scaling}` (pick whatever scaling value you
    like).

    For the other model_name choices:

    - PSTexture: the Portilla-Simoncelli texture stats with n_scales=4,
      n_orientations=4, spatial_corr_width=9, use_true_correlations=True

    - VGG16_poolN: pretrained VGG16 from torchvision, through Nth max pooling
      layer (where N is an int from 1 to 5)

    If you want to resume synthesis from an earlier run that didn't
    finish, set `continue_path` to the path of the `.pt` file created by
    that earlier run. We will then load it in and continue. For right
    now, we don't do anything to make sure that the arguments you pass
    to the function are the same as the first time, we just use the ones
    passed in. Generally, they should be identical, with the exception
    of learning_rate (which can be None to resume where you left off)
    and max_iter (which gives the number of extra iterations you want to
    do). Specifically, I think things might get weird if you do this
    initially on a GPU and then try to resume on a CPU (or vice versa),
    for example. When resuming, there's always a slight increase in the
    loss that, as far as I can tell, is unavoidable; it goes away
    quickly (and the loss continues its earlier trend) and so I don't
    think is an issue.

    Parameters
    ----------
    model_name : str
        str specifying which of the model we should use. See above for more
        details.
    image : str or array_like
        Either the path to the file to load in or the loaded-in
        image. If array_like, we assume it's already 2d (i.e.,
        grayscale)
    seed : int, optional
        The number to use for initializing numpy and torch's random
        number generators
    min_ecc : float, optional
        The minimum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    max_ecc : float, optional
        The maximum eccentricity for the pooling windows (see
        plenoptic.simul.VentralStream for more details)
    learning_rate : float, optional
        The learning rate to pass to metamer.synthesize's optimizer
    max_iter : int, optional
        The maximum number of iterations we allow the synthesis
        optimization to run for
    stop_criterion : float, optional
        The stop criterion. If the loss has changed by less than this over the
        past stop_iters_to_check iterations, we quit out.
    stop_iters_to_check : int, optional
        How many iterations back to check in order to see if the loss has
        stopped decreasing.
    save_path : str or None, optional
        If a str, the path to the file to save the metamer object to. If
        None, we don't save the synthesis output (that's probably a bad
        idea)
    initial_image : {'white', 'pink', 'gray', 'blue'} or path to a file
        What to use for the initial image. If 'white', we use white
        noise. If 'pink', we use pink noise
        (``pyrtools.synthetic_images.pink_noise(fract_dim=1)``). If
        'blue', we use blue noise
        (``pyrtools.synthetic_images.blue_noise(fract_dim=1)``). If
        'gray', we use a flat image with values of .5 everywhere. If
        path to a file, that's what we use as our initial image (and so
        the seed will have no effect on this).
        std dev of Gaussian noise added to image to initialize synthesis.
    gpu_id : int or None, optional
        If not None, the GPU we will use. If None, we run on CPU. We
        don't do anything clever to handle that here, but the
        contextmanager utils.get_gpu_id does, so you should use that to
        make sure you're using a GPU that exists and is available (see
        Snakefile for example)
    cache_dir : str or None, optional
        The directory to cache the windows tensor in. If set, we'll look
        there for cached versions of the windows we create, load them if
        they exist and create and cache them if they don't. If None, we
        don't check for or cache the windows.
    normalize_dict : str or None, optional
        If a str, the path to the dictionary containing the statistics to use
        for normalization for the model. If None, we don't normalize anything
    optimizer: {'Adam', 'SGD'}
        The choice of optimization algorithm
    loss_func : {'mse', 'l2'}, optional
        Which loss function to use.
    range_penalty_lambda :
        Lambda to multiply by range penalty and add to loss.
    coarse_to_fine : { 'together', 'separate', False}, optional
        If False, don't do coarse-to-fine optimization. Else, there
        are two options for how to do it:
        - 'together': start with the coarsest scale, then gradually
          add each finer scale. this is like blurring the objective
          function and then gradually adding details and is probably
          what you want.
        - 'separate': compute the gradient with respect to each
          scale separately (ignoring the others), then with respect
          to all of them at the end.
    coarse_to_fine_kwargs : dict, optional
        Dictionary of args for coarse to fine optimization. See
        Metamer.synthesize() docstring for details.
    continue_path : str or None, optional
        If None, we synthesize a new metamer. If str, this should be the
        path to a previous synthesis run, which we are resuming. In that
        case, you may set learning_rate to None (in which case we resume
        where we left off) and set max_iter to a different value (the
        number of extra iterations to run) otherwise the rest of the
        arguments should be the same as the first run.
    num_threads : int or None, optional
        If int, the number of CPU threads to use. If None, we don't restrict it
        and so we'll use all available resources. If using the GPU, this won't
        matter (all costly computations are done on the GPU). If one the CPU,
        we seem to only improve performance up to ~12 threads (at least with
        RGC model), and actively start to harm performance as we get above 40.

    """
    print("Using seed %s" % seed)
    if num_threads is not None:
        print(f"Using {num_threads} threads")
        torch.set_num_threads(num_threads)
    else:
        print("Not restricting number of threads, will probably use max "
              f"available ({torch.get_num_threads()})")
    po.tools.set_seed(seed)
    image = setup_image(image, 3 if 'VGG16' in model_name else 1)
    print(f"Using initial image {initial_image}")
    initial_image = setup_initial_image(initial_image, image)
    # this will be false if normalize_dict is None or an empty list
    if normalize_dict:
        normalize_dict = torch.load(normalize_dict)
    model = setup_model(model_name, image, min_ecc, max_ecc, cache_dir,
                        normalize_dict)
    model_str = f"Using model {model_name}"
    if model_name.startswith('RGC') or model_name.startswith("V1"):
        model_str += f" from {min_ecc} degrees to {max_ecc} degrees"
    print(model_str)
    image, initial_image, model = setup_device(image, initial_image, model, gpu_id=gpu_id)
    store_progress = max(10, max_iter//100)
    if loss_func == 'mse':
        loss_func = po.tools.optim.mse
    elif loss_func == 'l2':
        loss_func = po.tools.optim.l2_norm
    else:
        raise Exception(f"Don't know how to handle loss_func {loss_func}!")
    metamer = po.synth.Metamer(image, model, loss_func, range_penalty_lambda,
                               initial_image=initial_image)
    print(f"Using optimizer {optimizer}")
    if optimizer == 'Adam':
        opt = torch.optim.Adam([metamer.synthesized_signal], lr=learning_rate, amsgrad=True)
    elif optimizer == 'SGD':
        opt = torch.optim.SGD([metamer.synthesized_signal], lr=learning_rate)
    if continue_path is not None:
        print("Resuming synthesis saved at %s" % continue_path)
        metamer = metamer.load(continue_path)
        opt = None
    print(f"Using learning rate {learning_rate}, stop_criterion {stop_criterion} (stop_iters_to_check "
          f"{stop_iters_to_check}), and max_iter {max_iter}")
    print(f"Using coarse-to-fine {coarse_to_fine} with kwargs {coarse_to_fine_kwargs}")
    start_time = time.time()
    metamer.synthesize(max_iter=max_iter, optimizer=opt,
                       store_progress=store_progress,
                       stop_criterion=stop_criterion,
                       stop_iters_to_check=stop_iters_to_check,
                       coarse_to_fine=coarse_to_fine,
                       coarse_to_fine_kwargs=coarse_to_fine_kwargs)
    duration = time.time() - start_time
    print(f"Synthesis took {duration} seconds")
    # make sure everything's on the cpu for saving
    metamer.to('cpu')
    if save_path is not None:
        save(save_path, metamer)
