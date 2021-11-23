#!/usr/bin/env python3
"""various utils
"""
import numpy as np
import warnings
import GPUtil
import os
import yaml
import os.path as op
from contextlib import contextmanager
from itertools import cycle


def create_image(image_type, image_size, save_path=None, period=4):
    r"""Create a simple image

    Parameters
    ----------
    image_type : {'plaid', 'checkerboard'}
        What type of image to create
    image_size : tuple
        2-tuple of ints, specifying the image size
    save_path : str or None, optional
        If a str, the path to save the padded image at. If None, we
        don't save
    period : int, optional
        If image_type is 'plaid' or 'checkerboard', what period to use
        for the square waves that we use to generate them.

    Returns
    -------
    image : np.array
        The image we created

    """
    if image_type in ['plaid', 'checkerboard']:
        image = pt.synthetic_images.square_wave(image_size, period=period)
        image += pt.synthetic_images.square_wave(image_size, period=period, direction=np.pi/2)
        image += np.abs(image.min())
        image /= image.max()
        if image_type == 'checkerboard':
            image = np.where((image < .75) & (image > .25), 1, 0)
    else:
        raise Exception("Don't know how to handle image_type %s!" % image_type)
    if save_path is not None:
        imageio.imwrite(save_path, image)
    return image


def get_ref_image_full_path(image_name,
                            preproc_methods=['full', 'gamma-corrected',
                                             'range', 'degamma',
                                             'downsample'],
                            downsample=False):
    """Check whether image is in ref_image or ref_image_preproc dir.

    Parameters
    ----------
    image_name : str
        name of the (e.g., like those seen in `config.yml:
        DEFAULT_METAMERS: image_name`)
    preproc_methods : list, optional
        list of preproc methods we may have applied. probably shouldn't
        change this
    downsample : bool or int, optional
        whether we want the downsampled version of the ref images or not. If
        True, we downsample by 2. If an int, we downsample by that amount.

    Returns
    -------
    path : str
        full path to the reference image

    """
    with open(op.join(op.dirname(op.realpath(__file__)), '..', 'config.yml')) as f:
        defaults = yaml.safe_load(f)
        template = defaults['REF_IMAGE_TEMPLATE_PATH']
        DATA_DIR = defaults['DATA_DIR']
    if any([i in image_name for i in preproc_methods]):
        template = template.replace('ref_images', 'ref_images_preproc')
    if downsample:
        if downsample is True:
            downsample = 2
        if 'range' in image_name:
            image_name = image_name.replace('_ran', f'_downsample-{downsample}_ran')
        else:
            image_name += f'_downsample-{downsample}'
    template = template.format(image_name=image_name, DATA_DIR=DATA_DIR)
    # the next bit will remove all slashes from the string, so we need to
    # figure out whether we want to start with os.sep or not
    if template.startswith('/'):
        start = os.sep
    else:
        start = ''
    # this makes sure we're using the right os.sep and also removes any double
    # slashes we might have accidentally introduced
    return start + op.join(*template.split('/'))


def convert_im_to_float(im):
    r"""Convert image from saved data type to float.

    Images are saved as either 8 or 16 bit integers, and for our
    purposes we generally want them to be floats that lie between 0 and
    1. In order to properly convert them, we divide the image by the
    maximum value its dtype can take (255 for 8 bit, 65535 for 16 bit).

    Note that for this to work, it should be called right after the
    image was loaded in; most manipulations will implicitly convert the
    image to a float, and then we cannot determine what to divide it by.

    Parameters
    ----------
    im : numpy array or imageio Array
        The image to convert

    Returns
    -------
    im : numpy array
        The converted image, now with dtype=np.float32 and all values
        between 0 and 1

    """
    return im / np.iinfo(im.dtype).max


def convert_im_to_int(im, dtype=np.uint8):
    r"""Convert image from float to 8 or 16 bit image.

    We work with float images that lie between 0 and 1, but for saving
    them (either as png or in a numpy array), we want to convert them to
    8 or 16 bit integers. This function does that by multiplying it by
    the max value for the target dtype (255 for 8 bit 65535 for 16 bit)
    and then converting it to the proper type.

    We'll raise an exception if the max is higher than 1, in which case
    we have no idea what to do.

    Parameters
    ----------
    im : numpy array
        The image to convert
    dtype : {np.uint8, np.uint16}
        The target data type

    Returns
    -------
    im : numpy array
        The converted image, now with dtype=dtype

    """
    if im.max() > 1:
        if im.max() - 1 < 1e-4:
            warnings.warn("There was a precision/rounding error somewhere and im.max is "
                          f"{im.max()}. Setting that to 1 and converting anyway")
            im = np.clip(im, 0, 1)
        else:
            raise Exception("all values of im must lie between 0 and 1, but max is %s" % im.max())
    if im.min() < 0:
        if abs(im.min()) < 1e-4:
            warnings.warn("There was a precision/rounding error somewhere and im.min is "
                          f"{im.min()}. Setting that to 0 and converting anyway")
            im = np.clip(im, 0, 1)
        else:
            raise Exception("all values of im must lie between 0 and 1, but min is %s" % im.min())
    return (im * np.iinfo(dtype).max).astype(dtype)


@contextmanager
def get_gpu_id(get_gid=True, n_gpus=4, on_cluster=False):
    """Get next available GPU and lock it.

    Note that the lock file created will be at
    /tmp/LCK_gpu_{allocated_gid}.lock

    This is based on the solution proposed at
    https://github.com/snakemake/snakemake/issues/281#issuecomment-610796104
    and then modified slightly

    Parameters
    ----------
    get_gid : bool, optional
        if True, return the ID of the first available GPU. If False,
        return None. This weirdness is to allow us to still use this
        contextmanager when we don't actually want to create a lockfile
    n_gpus : int, optional
        number of GPUs on this device
    on_cluster : bool, optional
        whether we're on a cluster or not. if so, then we just return the gid
        for the first available GPU, since the job scheduler has taken care of
        this for us. We don't use dotlockfile in this case

    Returns
    -------
    allocated_gid : int
        the ID of the GPU to use

    """
    allocated_gid = None
    if not get_gid:
        avail_gpus = []
    else:
        avail_gpus = GPUtil.getAvailable(order='memory', maxLoad=.1, maxMemory=.1,
                                         includeNan=False, limit=n_gpus)
    for gid in cycle(avail_gpus):
        # just grab first gpu in this case
        if on_cluster:
            allocated_gid = gid
            break
        # then we've successfully created the lockfile
        if os.system(f"dotlockfile -r 1 /tmp/LCK_gpu_{gid}.lock") == 0:
            allocated_gid = gid
            break
    try:
        yield allocated_gid
    finally:
        os.system(f"dotlockfile -u /tmp/LCK_gpu_{allocated_gid}.lock")
