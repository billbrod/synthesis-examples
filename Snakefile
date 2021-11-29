# required to fix this strange problem:
# https://stackoverflow.com/questions/64797838/libgcc-s-so-1-must-be-installed-for-pthread-cancel-to-work
import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')
import os
import math
import re
import imageio
import os.path as op
import numpy as np
from synth import utils

configfile:
    # config is in the same directory as this file
    op.join(op.dirname(op.realpath(workflow.snakefile)), 'config.yml')
if not op.isdir(config["DATA_DIR"]):
    raise Exception("Cannot find the dataset at %s" % config["DATA_DIR"])
# for some reason, I can't get os.system('module list') to work
# properly on NYU Greene (it always returns a non-zero exit
# code). However, they do have the CLUSTER environmental variable
# defined, so we can use that
if os.system("module list") == 0 or os.environ.get("CLUSTER", None):
    # then we're on the cluster
    ON_CLUSTER = True
else:
    ON_CLUSTER = False
wildcard_constraints:
    num="[0-9]+",
    period="[0-9]+",
    size="[0-9,]+",
    bits="[0-9]+",
    img_preproc="full|degamma|gamma-corrected|gamma-corrected_full|range-[,.0-9]+|gamma-corrected_range-[,.0-9]+|downsample-[0-9.]+_range-[,.0-9]+",
    preproc="|_degamma|degamma",
    gpu="0|1",
    gammacorrected='|_gamma-corrected',
    context="paper|poster",
    synth_target="min|max"
ruleorder:
    preproc_image > crop_image > generate_image

# these are strings so we can control exactly how they're formatted in the path
RANGE_PENALTIES = {
    ('RGC_norm_gaussian_scaling-0.06', 'reptil_skin_size-256,256'): '10',
    ('VGG16_pool4', 'einstein_size-256,256'): '1e4',
    ('VGG16_pool3', 'reptil_skin_size-256,256'): '1e4',
    ('VGG16_pool3', 'checkerboard_period-64_range-.1,.9_size-256,256'): '1e5',
    ('PSTexture', 'checkerboard_period-64_range-.1,.9_size-256,256'): '1e4',
}

# this is ugly, but it's easiest way to just replace the one format
# target while leaving the others alone
DATA_DIR = config['DATA_DIR']
if not DATA_DIR.endswith('/'):
    DATA_DIR += '/'
REF_IMAGE_TEMPLATE_PATH = config['REF_IMAGE_TEMPLATE_PATH'].replace("{DATA_DIR}/", DATA_DIR)
# the regex here removes all string formatting codes from the string,
# since Snakemake doesn't like them
METAMER_TEMPLATE_PATH = re.sub(":.*?}", "}", config['METAMER_TEMPLATE_PATH'].replace("{DATA_DIR}/", DATA_DIR))
MAD_TEMPLATE_PATH = re.sub(":.*?}", "}", config['MAD_TEMPLATE_PATH'].replace("{DATA_DIR}/", DATA_DIR))
METAMER_LOG_PATH = METAMER_TEMPLATE_PATH.replace('metamers/{model_name}', 'logs/metamers/{model_name}').replace('_metamer.png', '.log')
MAD_LOG_PATH = MAD_TEMPLATE_PATH.replace('mad_images/fix-', 'logs/mad_images/fix-').replace('_mad.png', '.log')
CONTINUE_TEMPLATE_PATH = (METAMER_TEMPLATE_PATH.replace('metamers/{model_name}', 'metamers_continue/{model_name}')
                          .replace("{clamp_each_iter}/", "{clamp_each_iter}/attempt-{num}_iter-{extra_iter}"))
CONTINUE_LOG_PATH = CONTINUE_TEMPLATE_PATH.replace('metamers_continue/{model_name}', 'logs/metamers_continue/{model_name}').replace('_metamer.png', '.log')
TEXTURE_DIR = config['TEXTURE_DIR']
if TEXTURE_DIR.endswith(os.sep) or TEXTURE_DIR.endswith('/'):
    TEXTURE_DIR = TEXTURE_DIR[:-1]
if len(os.listdir(TEXTURE_DIR)) <= 800 and 'textures-subset-for-testing' not in TEXTURE_DIR:
    raise Exception(f"TEXTURE_DIR {TEXTURE_DIR} is incomplete!")


rule crop_image:
    input:
        op.join(config['DATA_DIR'], 'ref_images', '{image_name}.tiff')
    output:
        op.join(config['DATA_DIR'], 'ref_images', '{image_name}_size-{size}.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_name}_size-{size}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_name}_size-{size}_benchmark.txt')
    run:
        import imageio
        import contextlib
        from skimage import color
        import synth
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                im = imageio.imread(input[0])
                curr_shape = np.array(im.shape)[:2]
                target_shape = [int(i) for i in wildcards.size.split(',')]
                print(curr_shape, target_shape)
                if len(target_shape) == 1:
                    target_shape = 2* target_shape
                target_shape = np.array(target_shape)
                crop_amt = curr_shape - target_shape
                # this is ugly, but I can't come up with an easier way to make
                # sure that we skip a dimension if crop_amt is 0 for it
                cropped_im = im
                for i, c in enumerate(crop_amt):
                    if c == 0:
                        continue
                    else:
                        if i == 0:
                            cropped_im = cropped_im[c//2:-c//2]
                        elif i == 1:
                            cropped_im = cropped_im[:, c//2:-c//2]
                        else:
                            raise Exception("Can only crop up to two dimensions!")
                cropped_im = color.rgb2gray(cropped_im)
                imageio.imwrite(output[0], synth.utils.convert_im_to_int(cropped_im, np.uint16))
                # tiffs can't be read in using the as_gray arg, so we
                # save it as a png, and then read it back in as_gray and
                # save it back out
                cropped_im = imageio.imread(output[0], as_gray=True)
                imageio.imwrite(output[0], cropped_im.astype(np.uint16))


rule preproc_image:
    input:
        op.join(config['DATA_DIR'], 'ref_images', '{preproc_image_name}_size-{size}.png')
    output:
        op.join(config['DATA_DIR'], 'ref_images_preproc', '{preproc_image_name}_{img_preproc}_size-{size}.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_image_preproc',
                '{preproc_image_name}_{img_preproc}_size-{size}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_image_preproc',
                '{preproc_image_name}_{img_preproc}_size-{size}_benchmark.txt')
    run:
        import imageio
        import contextlib
        import numpy as np
        from skimage import transform
        import synth
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                im = imageio.imread(input[0])
                dtype = im.dtype
                im = np.array(im, dtype=np.float32)
                print("Original image has dtype %s" % dtype)
                if 'full' in wildcards.img_preproc:
                    print("Setting image to use full dynamic range")
                    # set the minimum value to 0
                    im = im - im.min()
                    # set the maximum value to 1
                    im = im / im.max()
                elif 'range' in wildcards.img_preproc:
                    a, b = re.findall('range-([.0-9]+),([.0-9]+)', wildcards.img_preproc)[0]
                    a, b = float(a), float(b)
                    print(f"Setting range to {a:02f}, {b:02f}")
                    if a > b:
                        raise Exception("For consistency, with range-a,b preprocessing, b must be"
                                        " greater than a, but got {a} > {b}!")
                    # set the minimum value to 0
                    im = im - im.min()
                    # set the maximum value to 1
                    im = im / im.max()
                    # and then rescale
                    im = im * (b - a) + a
                else:
                    print("Image will *not* use full dynamic range")
                    im = im / np.iinfo(dtype).max
                if 'gamma-corrected' in wildcards.img_preproc:
                    print("Raising image to 1/2.2, to gamma correct it")
                    im = im ** (1/2.2)
                if 'downsample' in wildcards.img_preproc:
                    downscale = float(re.findall('downsample-([.0-9]+)_', wildcards.img_preproc)[0])
                    im = transform.pyramid_reduce(im, downscale)
                # always save it as 16 bit
                print("Saving as 16 bit")
                im = synth.utils.convert_im_to_int(im, np.uint16)
                imageio.imwrite(output[0], im)


rule generate_image:
    output:
        op.join(config['DATA_DIR'], 'ref_images', '{image_type}_period-{period}_size-{size}.png')
    log:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_type}_period-{period}_size-'
                '{size}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'ref_images', '{image_type}_period-{period}_size-'
                '{size}_benchmark.txt')
    run:
        import synth
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                size = [int(s) for s in wildcards.size.split(',')]
                synth.utils.create_image(wildcards.image_type, size, output[0],
                                         int(wildcards.period))

                
rule preproc_textures:
    input:
        TEXTURE_DIR
    output:
        directory(TEXTURE_DIR + "_{preproc}")
    log:
        op.join(config['DATA_DIR'], 'logs', '{preproc}_textures.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', '{preproc}_textures_benchmark.txt')
    run:
        import imageio
        import contextlib
        from glob import glob
        import os.path as op
        import os
        from skimage import color
        import synth
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                os.makedirs(output[0])
                for i in glob(op.join(input[0], '*.jpg')):
                    im = imageio.imread(i)
                    im = synth.utils.convert_im_to_float(im)
                    if im.ndim == 3:
                        # then it's a color image, and we need to make it grayscale
                        im = color.rgb2gray(im)
                    if 'degamma' in wildcards.preproc:
                        # 1/2.2 is the standard encoding gamma for jpegs, so we
                        # raise this to its reciprocal, 2.2, in order to reverse
                        # it
                        im = im ** 2.2
                    # save as a 16 bit png
                    im = synth.utils.convert_im_to_int(im, np.uint16)
                    imageio.imwrite(op.join(output[0], op.split(i)[-1].replace('jpg', 'png')), im)


rule gen_norm_stats:
    input:
        TEXTURE_DIR + "{preproc}"
    output:
        # here V1 and texture could be considered wildcards, but they're
        # the only we're doing this for now
        op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture{preproc}_'
                'norm_stats-{num}.pt' )
    log:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture'
                '{preproc}_norm_stats-{num}.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture'
                '{preproc}_norm_stats-{num}_benchmark.txt')
    params:
        index = lambda wildcards: (int(wildcards.num) * 100, (int(wildcards.num)+1) * 100)
    run:
        import contextlib
        import sys
        sys.path.append(op.join(op.dirname(op.realpath(__file__)), 'extra_packages'))
        import plenoptic_part as pop
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # scaling doesn't matter here
                v1 = pop.PooledV1(1, (512, 512), num_scales=6)
                pop.optim.generate_norm_stats(v1, input[0], output[0], (512, 512),
                                             index=params.index)


# we need to generate the stats in blocks, and then want to re-combine them
rule combine_norm_stats:
    input:
        lambda wildcards: [op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture'
                                   '{preproc}_norm_stats-{num}.pt').format(num=i, **wildcards)
                           for i in range(math.ceil(len(os.listdir(TEXTURE_DIR))/100))]
    output:
        op.join(config['DATA_DIR'], 'norm_stats', 'V1_texture{preproc}_norm_stats.pt' )
    log:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture'
                '{preproc}_norm_stats.log')
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'norm_stats', 'V1_texture'
                '{preproc}_norm_stats_benchmark.txt')
    run:
        import torch
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                combined_stats = {}
                to_combine = [torch.load(i) for i in input]
                for k, v in to_combine[0].items():
                    if isinstance(v, dict):
                        d = {}
                        for l in v:
                            s = []
                            for i in to_combine:
                                s.append(i[k][l])
                            d[l] = torch.cat(s, 0)
                        combined_stats[k] = d
                    else:
                        s = []
                        for i in to_combine:
                            s.append(i[k])
                        combined_stats[k] = torch.cat(s, 0)
                torch.save(combined_stats, output[0])


def get_mem_estimate(wildcards, partition=None):
    r"""estimate the amount of memory that this will need, in GB
    """
    mem = 16
    if partition == 'rusty':
        if int(wildcards.gpu) == 0:
            # in this case, we *do not* want to specify memory (we'll get the
            # whole node allocated but slurm could still kill the job if we go
            # over requested memory)
            mem = ''
        else:
            # we'll be plugging this right into the mem request to slurm, so it
            # needs to be exactly correct
            mem = f"{mem}GB"
    return mem


rule cache_windows:
    output:
        op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}_e0-{min_ecc}_'
                'em-{max_ecc}_w-{t_width}_{window_type}.pt')
    log:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-'
                '{min_ecc}_em-{max_ecc}_w-{t_width}_{window_type}.log')
    benchmark:
        op.join(config["DATA_DIR"], 'logs', 'windows_cache', 'scaling-{scaling}_size-{size}_e0-'
                '{min_ecc}_em-{max_ecc}_w-{t_width}_{window_type}.benchmark.txt')
    resources:
        mem = get_mem_estimate,
    run:
        import contextlib
        import plenoptic as po
        import sys
        sys.path.append(op.join(op.dirname(op.realpath(__file__)), '..', 'extra_packages', 'pooling-windows'))
        import pooling
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                img_size = [int(i) for i in wildcards.size.split(',')]
                kwargs = {}
                if wildcards.window_type == 'cosine':
                    t_width = float(wildcards.t_width)
                    std_dev = None
                    min_ecc = float(wildcards.min_ecc)
                elif wildcards.window_type == 'gaussian':
                    std_dev = float(wildcards.t_width)
                    t_width = None
                    min_ecc = float(wildcards.min_ecc)
                pooling.PoolingWindows(float(wildcards.scaling), img_size, min_ecc,
                                       float(wildcards.max_ecc), cache_dir=op.dirname(output[0]),
                                       transition_region_width=t_width, std_dev=std_dev,
                                       window_type=wildcards.window_type, **kwargs)


def get_norm_dict(wildcards):
    # this is for metamers
    try:
        if 'norm' in wildcards.model_name:
            preproc = ''
            # lienar images should also use the degamma'd textures
            if 'degamma' in wildcards.image_name:
                preproc += '_degamma'
            return op.join(config['DATA_DIR'], 'norm_stats', f'V1_texture{preproc}'
                           '_norm_stats.pt')
        else:
            return []
    # this is for MAD images
    except AttributeError:
        norm_dicts = []
        if 'norm' in wildcards.fix_model_name:
            preproc = ''
            # lienar images should also use the degamma'd textures
            if 'degamma' in wildcards.image_name:
                preproc += '_degamma'
            norm_dicts.append(op.join(config['DATA_DIR'], 'norm_stats', f'V1_texture{preproc}'
                                      '_norm_stats.pt'))
        if 'norm' in wildcards.synth_model_name:
            preproc = ''
            # lienar images should also use the degamma'd textures
            if 'degamma' in wildcards.image_name:
                preproc += '_degamma'
            norm_dicts.append(op.join(config['DATA_DIR'], 'norm_stats', f'V1_texture{preproc}'
                                      '_norm_stats.pt'))
        return norm_dicts


def get_windows(wildcards):
    r"""determine the cached window path for the specified model
    """
    window_template = op.join(config["DATA_DIR"], 'windows_cache', 'scaling-{scaling}_size-{size}'
                              '_e0-{min_ecc:.03f}_em-{max_ecc:.01f}_w-{t_width}_{window_type}.pt')
    try:
        if 'size-' in wildcards.image_name:
            im_shape = wildcards.image_name[wildcards.image_name.index('size-') + len('size-'):]
            im_shape = im_shape.replace('.png', '')
            im_shape = [int(i) for i in im_shape.split(',')]
        else:
            try:
                im = imageio.imread(REF_IMAGE_TEMPLATE_PATH.format(image_name=wildcards.image_name))
                im_shape = im.shape
            except FileNotFoundError:
                raise Exception("Can't find input image %s or infer its shape, so don't know what "
                                "windows to cache!" %
                                REF_IMAGE_TEMPLATE_PATH.format(image_name=wildcards.image_name))
    except AttributeError:
        # then there was no wildcards.image_name, so grab the first one from
        # the DEFAULT_METAMERS list
        default_im = IMAGES[0]
        im_shape = default_im[default_im.index('size-') + len('size-'):]
        im_shape = im_shape.replace('.png', '')
        im_shape = [int(i) for i in im_shape.split(',')]
    try:
        max_ecc=float(wildcards.max_ecc)
        min_ecc=float(wildcards.min_ecc)
    except AttributeError:
        # then there was no wildcards.max/min_ecc, so grab the default values
        min_ecc = config['DEFAULT_METAMERS']['min_ecc']
        max_ecc = config['DEFAULT_METAMERS']['max_ecc']
    t_width = 1.0
    try:
        # this is for metamers
        model_names = [wildcards.model_name]
    except AttributeError:
        # this is for MAD
        model_names = [wildcards.fix_model_name, wildcards.synth_model_name]
    windows = []
    for mn in model_names:
        if 'cosine' in mn:
            window_type = 'cosine'
        elif 'gaussian' in mn:
            window_type = 'gaussian'
        try:
            scaling = mn.split('_scaling-')[1]
            mn = mn.split('_scaling-')[0]
        except IndexError:
            scaling = None
        if scaling is None:
            continue
        elif mn.startswith("RGC"):
            # RGC model only needs a single scale of PoolingWindows.
            size = ','.join([str(i) for i in im_shape])
            windows.append(window_template.format(scaling=scaling, size=size,
                                                  max_ecc=max_ecc, t_width=t_width,
                                                  min_ecc=min_ecc, window_type=window_type,))
        elif mn.startswith('V1'):
            # need them for every scale
            try:
                num_scales = int(re.findall('s([0-9]+)', mn)[0])
            except (IndexError, ValueError):
                num_scales = 4
            for i in range(num_scales):
                output_size = ','.join([str(int(np.ceil(j / 2**i))) for j in im_shape])
                windows.append(window_template.format(scaling=scaling, size=output_size,
                                                      max_ecc=max_ecc,
                                                      min_ecc=min_ecc,
                                                      t_width=t_width, window_type=window_type))
    return windows

def get_partition(wildcards, cluster):
    # if our V1 scaling value is small enough, we need a V100 and must specify
    # it. otherwise, we can use any GPU, because they'll all have enough
    # memory. The partition name depends on the cluster (greene or rusty), so
    # we have two different params, one for each, and the cluster config grabs
    # the right one. For now, greene doesn't require setting partition.
    if cluster not in ['greene', 'rusty']:
        raise Exception(f"Don't know how to handle cluster {cluster}")
    if int(wildcards.gpu) == 0:
        if cluster == 'rusty':
            return 'ccn'
        elif cluster == 'greene':
            return None
    else:
        if cluster == 'rusty':
            return 'gpu'
        elif cluster == 'greene':
            return None


def get_constraint(wildcards, cluster):
    if int(wildcards.gpu) > 0 and cluster == 'rusty':
        return 'v100-32gb'
    else:
        return ''


def get_cpu_num(wildcards):
    if int(wildcards.gpu) > 0:
        # then we're using the GPU and so don't really need CPUs
        cpus = 1
    else:
        try:
            # this is for Metamers
            models = [wildcards.model_name]
        except AttributeError:
            # this is for MAD
            models = [wildcards.fix_model_name, wildcards.synth_model_name]
        scaling = [float(mn.split('_scaling-')[1]) if 'scaling' in mn else 1
                   for mn in models]
        # want to get cpus based on the smallest scaling value
        scaling = min(scaling)
        # these are all based on estimates from rusty (which automatically
        # gives each job 28 nodes), and checking seff to see CPU usage
        if float(scaling) > .06:
            cpus = 21
        elif float(scaling) > .03:
            cpus = 26
        else:
            cpus = 28
    return cpus


def get_init_image(wildcards):
    if wildcards.init_type in ['white', 'gray', 'pink', 'blue']:
        return []
    else:
        try:
            # then this is just a nosie level, and there is no input required
            float(wildcards.init_type)
            return []
        except ValueError:
            return utils.get_ref_image_full_path(wildcards.init_type)

                           
rule create_metamers:
    input:
        ref_image = lambda wildcards: utils.get_ref_image_full_path(wildcards.image_name),
        windows = get_windows,
        norm_dict = get_norm_dict,
        init_image = get_init_image,
    output:
        METAMER_TEMPLATE_PATH.replace('_metamer.png', '.pt'),
        METAMER_TEMPLATE_PATH.replace('metamer.png', 'synthesis.png'),
        METAMER_TEMPLATE_PATH.replace('.png', '.npy'),
        report(METAMER_TEMPLATE_PATH),
    log:
        METAMER_LOG_PATH,
    benchmark:
        METAMER_LOG_PATH.replace('.log', '_benchmark.txt'),
    resources:
        gpu = lambda wildcards: int(wildcards.gpu),
        cpus_per_task = get_cpu_num,
        mem = get_mem_estimate,
        # this seems to be the best, anymore doesn't help and will eventually hurt
        num_threads = 9,
    params:
        rusty_mem = lambda wildcards: get_mem_estimate(wildcards, 'rusty'),
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
        # if we can use a GPU, synthesis doesn't take very long. If we can't,
        # it takes forever (7 days is probably not enough, but it's the most I
        # can request on the cluster -- will then need to manually ask for more
        # time).
        time = lambda wildcards: {1: '12:00:00', 0: '7-00:00:00'}[int(wildcards.gpu)],
        rusty_partition = lambda wildcards: get_partition(wildcards, 'rusty'),
        rusty_constraint = lambda wildcards: get_constraint(wildcards, 'rusty'),
    run:
        import synth
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                # bool('False') == True, so we do this to avoid that
                # situation
                if wildcards.coarse_to_fine == 'False':
                    coarse_to_fine = False
                    coarse_to_fine_kwargs = {}
                else:
                    coarse_to_fine = wildcards.coarse_to_fine
                    coarse_to_fine_kwargs = {'ctf_iters_to_check': int(wildcards.ctf_iters)}
                    try:
                        coarse_to_fine_kwargs['change_scale_criterion'] = float(wildcards.ctf_criterion)
                    except ValueError:
                        # the criterion can also be None
                        assert wildcards.ctf_criterion == 'None'
                        coarse_to_fine_kwargs['change_scale_criterion'] = None
                if wildcards.init_type not in ['white', 'blue', 'pink', 'gray']:
                    init_type = synth.utils.get_ref_image_full_path(wildcards.init_type)
                else:
                    init_type = wildcards.init_type
                if resources.gpu == 1:
                    get_gid = True
                elif resources.gpu == 0:
                    get_gid = False
                else:
                    raise Exception("Multiple gpus are not supported!")
                with synth.utils.get_gpu_id(get_gid, on_cluster=ON_CLUSTER) as gpu_id:
                    synth.create_metamers.main(wildcards.model_name,
                                               input.ref_image,
                                               int(wildcards.seed),
                                               float(wildcards.min_ecc),
                                               float(wildcards.max_ecc),
                                               float(wildcards.learning_rate),
                                               int(wildcards.max_iter),
                                               float(wildcards.stop_criterion),
                                               int(wildcards.stop_iters),
                                               output[0], init_type, gpu_id,
                                               params.cache_dir, input.norm_dict,
                                               wildcards.optimizer, wildcards.loss,
                                               float(wildcards.range_lambda),
                                               coarse_to_fine, coarse_to_fine_kwargs,
                                               num_threads=resources.num_threads)


rule create_mad_images:
    input:
        ref_image = lambda wildcards: utils.get_ref_image_full_path(wildcards.image_name),
        windows = get_windows,
        norm_dict = get_norm_dict,
        init_image = get_init_image,
    output:
        MAD_TEMPLATE_PATH.replace('_mad.png', '.pt'),
        MAD_TEMPLATE_PATH.replace('mad.png', 'synthesis.mp4'),
        MAD_TEMPLATE_PATH.replace('mad.png', 'synthesis.png'),
        MAD_TEMPLATE_PATH.replace('mad.png', 'image-diff.png'),
        MAD_TEMPLATE_PATH.replace('.png', '.npy'),
        report(MAD_TEMPLATE_PATH),
    log:
        MAD_LOG_PATH,
    benchmark:
        MAD_LOG_PATH.replace('.log', '_benchmark.txt'),
    resources:
        gpu = lambda wildcards: int(wildcards.gpu),
        cpus_per_task = get_cpu_num,
        mem = get_mem_estimate,
        # this seems to be the best, anymore doesn't help and will eventually hurt
        num_threads = 9,
    params:
        rusty_mem = lambda wildcards: get_mem_estimate(wildcards, 'rusty'),
        cache_dir = lambda wildcards: op.join(config['DATA_DIR'], 'windows_cache'),
        # if we can use a GPU, synthesis doesn't take very long. If we can't,
        # it takes forever (7 days is probably not enough, but it's the most I
        # can request on the cluster -- will then need to manually ask for more
        # time).
        time = lambda wildcards: {1: '12:00:00', 0: '7-00:00:00'}[int(wildcards.gpu)],
        rusty_partition = lambda wildcards: get_partition(wildcards, 'rusty'),
        rusty_constraint = lambda wildcards: get_constraint(wildcards, 'rusty'),
    run:
        import synth
        import contextlib
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                if resources.gpu == 1:
                    get_gid = True
                elif resources.gpu == 0:
                    get_gid = False
                else:
                    raise Exception("Multiple gpus are not supported!")
                # tradeoff_lambda can be a float or None
                try:
                    tradeoff_lambda = float(wildcards.tradeoff_lambda)
                except ValueError:
                    tradeoff_lambda = None
                fix_norm_dict, synth_norm_dict = None, None
                if 'norm' in wildcards.fix_model_name:
                    fix_norm_dict = input.norm_dict[0]
                    if 'norm' in wildcards.synth_model_name:
                        synth_norm_dict = input.norm_dict[1]
                elif 'norm' in wildcards.synth_model_name:
                    synth_norm_dict = input.norm_dict[0]
                with synth.utils.get_gpu_id(get_gid, on_cluster=ON_CLUSTER) as gpu_id:
                    synth.create_mad_images.main(wildcards.fix_model_name,
                                                 wildcards.synth_model_name,
                                                 input.ref_image,
                                                 wildcards.synth_target,
                                                 int(wildcards.seed),
                                                 float(wildcards.min_ecc),
                                                 float(wildcards.max_ecc),
                                                 float(wildcards.learning_rate),
                                                 int(wildcards.max_iter),
                                                 float(wildcards.stop_criterion),
                                                 int(wildcards.stop_iters),
                                                 output[0],
                                                 float(wildcards.init_type),
                                                 gpu_id, params.cache_dir,
                                                 fix_norm_dict,
                                                 synth_norm_dict,
                                                 wildcards.optimizer,
                                                 tradeoff_lambda,
                                                 float(wildcards.range_lambda),
                                                 num_threads=resources.num_threads)


def get_metamers(wildcards):
    template_path = op.join(config['DATA_DIR'], 'metamers', '{model}', '{image}', 'opt-Adam_loss-{loss}_penalty-{penalty}',
                            'stop-iters-50_ctf-{ctf}_ctf-crit-None_ctf-iters-{ctf_iters}',
                            'seed-0_init-white_lr-{lr}_e0-0.500_em-3.000_iter-5000_stop-crit-1e-09_gpu-1_metamer.png')
    images = ['einstein_size-256,256', 'reptil_skin_size-256,256', 'checkerboard_period-64_range-.1,.9_size-256,256']
    models = [f'RGC_norm_gaussian_scaling-{wildcards.scaling}', 'PSTexture', f'VGG16_pool{wildcards.poolN}']
    metamers = []
    for model in models:
        default_penalty = {'RGC': 1.5, 'VGG16': '1e3', 'PSTexture': '0.5'}[model.split('_')[0]]
        lr = .005 if model.startswith("VGG16") else .01
        ctf = 'together' if model.startswith("PSTexture") else False
        ctf_iters = 15 if model.startswith("PSTexture") else None
        loss = 'l2' if model.startswith("PSTexture") else 'mse'
        for im in images:
            penalty = RANGE_PENALTIES.get((model, im), default_penalty)
            metamers.append(template_path.format(model=model, image=im,
                                                 penalty=penalty, ctf=ctf,
                                                 ctf_iters=ctf_iters,
                                                 loss=loss, lr=lr))
    return metamers


rule example_metamer_figure:
    input:
        target_images = [op.join(config['DATA_DIR'], 'ref_images', 'einstein_size-256,256.png'),
                         op.join(config['DATA_DIR'], 'ref_images', 'reptil_skin_size-256,256.png'),
                         op.join(config['DATA_DIR'], 'ref_images_preproc', 'checkerboard_period-64_range-.1,.9_size-256,256.png')],
        metamers = get_metamers,
    output:
        op.join(config['DATA_DIR'], 'figures', '{context}', 'example_metamers_scaling-{scaling}_VGG16-pool{poolN}.svg'),
    log:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'example_metamers_scaling-{scaling}_VGG16-pool{poolN}.log'),
    benchmark:
        op.join(config['DATA_DIR'], 'logs', 'figures', '{context}', 'example_metamers_scaling-{scaling}_VGG16-pool{poolN}_benchmark.txt'),
    run:
        import synth
        import contextlib
        import matplotlib.pyplot as plt
        import torch
        import plenoptic as po
        with open(log[0], 'w', buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                style, _ = synth.style.plotting_style(wildcards.context)
                plt.style.use(style)
                # need to load these in separately because some are RGB, some
                # grayscale, and we need to turn them all into 3-channel images
                # for plotting
                imgs = []
                for img in input:
                    img = po.load_images(img, as_gray=False)
                    if img.shape[1] == 1:
                        img = img.repeat(1, 3, 1, 1)
                    imgs.append(img)
                imgs = torch.cat(imgs)
                n_imgs = len(input.target_images)
                models = {f'foveated_luminance({wildcards.scaling})': imgs[n_imgs:2*n_imgs],
                          'PS_texture': imgs[2*n_imgs:3*n_imgs],
                          f'VGG16_pool{wildcards.poolN}': imgs[3*n_imgs:4*n_imgs]}
                fig = synth.figures.example_metamer_figure(imgs[:n_imgs], **models)
                fig.savefig(output[0], bbox_inches='tight')
