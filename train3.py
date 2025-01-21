"""Training and evaluation of the GMM model on hyperspectral images."""
import argparse
import logging
import os
import pickle  # nosec
import time

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import crism_ml.preprocessing as cp
import crism_ml.plot as cpl
import crism_ml.lab as cl
from crism_ml.io import cache_to, loadmat, load_image, image_shape
from crism_ml.models import HBM, HBMPrior
from crism_ml import N_JOBS, CONF
from crism_ml.lab import FULL_NAMES

from scipy.ndimage import zoom
import json

import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# these classes have a default model weight vector associated with them
WEIGHT_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                  18, 19, 20, 23, 25, 26, 27, 29, 30, 31, 33, 34, 35, 40, 36,
                  37, 38, 39]  # 35


def iteration_weights(labels=None):
    """Return probability weights per class and model.

    Parameters
    ----------
    labels: list
        restrict the weights to the given class labels

    Returns
    -------
    ww_: ndarray
        array 15 x N classe with per-model and per-class weights
    """
    def _a(lst):  # for broadcasting
        return np.array([lst]).T

    ww_ = np.zeros((15, len(WEIGHT_CLASSES)))   # n iter, n classes
    # H2O ice, CO2 ice, gypsum
    ww_[np.r_[3:7, 8:12], :3] = _a([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    # hydroxyl sulfate
    ww_[3:11, 3] = [0.2, 0.1, 0.05, 0.05, 0.1, 0.2, 0.2, 0.1]
    # hematite
    ww_[3:14, 4] = [0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05]
    # nontronite, saponite
    ww_[np.r_[3:7, 9:14], 5:7] = _a([
        .2, 0.3, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05])
    # prehnite, jarosite, serpentine
    ww_[np.r_[3:7, 8:12], 7] = [0.2, 0.2, 0.05, 0.05, 0.1, 0.2, 0.1, 0.1]
    ww_[3:12, 8] = [0.2, 0.05, 0.2, 0.1, 0.05, 0.05, 0.15, 0.15, 0.05]
    ww_[3:13, 9] = [0.1, 0.1, 0.05, 0.05, 0.4, 0.05, 0.1, 0.05, 0.05, 0.05]
    ww_[3:10, 10] = [0.1, 0.2, 0.2, 0.1, 0.15, 0.15, 0.1]  # alunite
    # akaganeite, fe/ca co3, beidellite, kaolinite, bassanite
    ww_[np.r_[3:7, 10:14], 11] = [0.2, 0.05, 0.05, 0.1, 0.05, 0.1, 0.4, 0.05]
    ww_[np.r_[3:7, 8:12], 12] = [0.2, 0.2, 0.05, 0.05, 0.1, 0.2, 0.1, 0.1]
    ww_[3:12, 13:15] = _a([0.2, 0.2, 0.05, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05])
    ww_[np.r_[3:7, 10:13], 15] = [0.2, 0.1, 0.15, 0.15, 0.1, 0.2, 0.1]
    # epidote
    ww_[np.r_[3:7, 9:13, 14], 16] = [
        0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.2]
    # al smectite, mg sulfate, mg cl salt, illite, analcime, kieserite
    ww_[3:12, 17] = [0.2, 0.2, 0.05, 0.05, 0.05, 0.3, 0.05, 0.05, 0.05]
    ww_[np.r_[3:7, 10:13], 18:20] = _a([0.2, 0.1, 0.15, 0.15, 0.1, 0.2, 0.1])
    ww_[np.r_[3:7, 8:12], 20] = [0.2, 0.2, 0.05, 0.05, 0.1, 0.2, 0.1, 0.1]
    ww_[np.r_[3:7, 10:13], 21] = [0.2, 0.1, 0.15, 0.15, 0.1, 0.2, 0.1]
    ww_[3:12, 22] = [0.2, 0.05, 0.1, 0.1, 0.2, 0.05, 0.05, 0.05, 0.2]
    # hydrated silica, copiapite (hydrated sulfate)
    ww_[np.r_[3:7, 8:13], 23:25] = _a([
        0.2, 0.05, 0.05, 0.05, 0.4, 0.1, 0.05, 0.05, 0.05])
    # CO3, chlorite
    ww_[np.r_[3, 5:7, 9:14], 25] = [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ww_[np.r_[3:7, 8:12], 26] = [0.2, 0.1, 0.05, 0.05, 0.1, 0.2, 0.2, 0.1]
    # flat categories
    ww_[3:14, 27:34] = _a([
        0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.05])
    ww_[3, 34] = 1
    ww_[0:4, :] = ww_[3:4, :] / 4

    if labels is not None:
        missing = set(labels) - set(WEIGHT_CLASSES)
        if missing:
            missing = ', '.join(str(k) for k in missing)
            raise ValueError(f"Missing weights for classes: {missing}.")
        ww_ = ww_[:, [WEIGHT_CLASSES.index(c)
                      for c in labels if c in WEIGHT_CLASSES]]

    return ww_


def feat_masks(as_intervals=False):
    """Feature masks per iteration for normal and bland pixels.

    Parameters
    ----------
    as_intervals: bool
        return the interval starting and ending points instead of the indices

    Returns
    -------
    maskb: ndarray
        per-model band indices for the bland models
    masks: ndarray
        per-model band indices for the mineral models
    """
    def _to_list(mask):
        return [np.concatenate([np.arange(*t) for t in sl]) for sl in mask]

    masks = [[(3, 92, 4), (109, 142, 4), (159, 248, 4)],
             [(4, 93, 4), (110, 143, 4), (160, 245, 4)],
             [(5, 94, 4), (111, 144, 4), (161, 246, 4)],
             [(6, 95, 4), (112, 145, 4), (162, 247, 4)],
             [(44, 75)], [(104, 135)], [(114, 145)], [(154, 185)],
             [(164, 195)], [(174, 205)], [(184, 215)], [(194, 225)],
             [(204, 235)], [(214, 245)], [(59, 90)]]
    maskb = [[(4, 95), (109, 145), (159, 248)]]

    if as_intervals:
        return maskb, masks
    return _to_list(maskb), _to_list(masks)


def default_data_loader(datadir):
    """Load the default mineral dataset.

    Loads the ``CRISM_labeled_pixels_ratioed`` dataset; it is invoked by
    'load_data'.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored

    Returns
    -------
    pixspec: ndarray
        array n_spectra x 350 with the unratioed dataset spectra
    pixlabs: ndarray
        spectra labels
    pixims: ndarray
        ids of the images
    """
    data = loadmat(os.path.join(datadir, "CRISM_labeled_pixels_ratioed.mat"))
    pixspec = data['pixspec'][:, :cp.N_BANDS]
    pixims = data['pixims'].squeeze()
    pixlabs = cl.relabel(data['pixlabs'].squeeze(), cl.ALIASES_TRAIN)

    return pixspec, pixlabs, pixims


@cache_to("dataset.npz", use_version=True)
def load_data(datadir):
    """Load the mineral dataset and removes spikes.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored

    Returns
    -------
    pixspec: ndarray
        array n_spectra x 350 with the unratioed dataset spectra
    pixlabs: ndarray
        spectra labels
    pixims: ndarray
        ids of the images
    """
    loader = CONF.get('data_loader', None)
    loader = default_data_loader if loader is None else loader

    pixspec, pixlabs, pixims = loader(datadir)
    logging.info("Loaded ratioed dataset.")

    logging.info("Removing spikes...")
    pixspec = cp.remove_spikes(
        pixspec, CONF['despike_params']).astype(np.float32)
    logging.info("Done.")

    return pixspec, pixlabs, pixims


def default_unratioed_loader(datadir):
    """Load the default bland dataset.

    Loads the ``CRISM_bland_unratioed`` dataset; it is invoked by
    'load_data_unratioed'.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored

    Returns
    -------
    pixspec: ndarray
        array n_spectra x 350 with the unratioed dataset spectra
    pixims: ndarray
        ids of the images
    """
    data = loadmat(os.path.join(datadir, "CRISM_bland_unratioed.mat"))
    return data['pixspec'], data['pixims'].squeeze()


@cache_to("dataset_bland.npz", use_version=True)
def load_unratioed_data(datadir):
    """Load bland pixels to train blandess detectors.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored

    Returns
    -------
    pixspec: ndarray
        array n_spectra x 350 with the unratioed dataset spectra
    pixlabs: ndarray
        spectra labels (all set to 0)
    pixims: ndarray
        ids of the images
    """
    loader = CONF.get('bland_data_loader', None)
    loader = default_unratioed_loader if loader is None else loader

    pixspec, pixims = loader(datadir)
    logging.info("Loaded unratioed dataset.")

    bad = np.sum(cp.spikes(pixspec[:, :cp.N_BANDS], 27, 10, mask=True),
                 axis=1) > 0

    pixspec, pixims = pixspec[~bad, :], pixims[~bad]
    return pixspec, np.zeros_like(pixims, dtype=np.int16), pixims


@cache_to("bmodel.pkl", use_version=True)
def train_model_bland(datadir, fin):
    """Load dataset and train bland pixel model.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored
    fin: list
        list of feature masks (one per model)

    Returns
    -------
    bmodels: list[HBM]
        list of trained models
    """
    xtrain, ytrain, ids = load_unratioed_data(datadir)
    prior = HBMPrior(**CONF['bland_model_params'])

    def _train(fm_):
        gmm2 = HBM(prior=prior)
        gmm2.fit(cp.normr(xtrain[:, fm_]), ytrain, ids)
        return gmm2

    ts_ = time.time()
    if len(fin) == 1:
        bmodels = [_train(fin[0])]
    else:
        bmodels = Parallel(n_jobs=N_JOBS)(delayed(_train)(fm_) for fm_ in fin)
    logging.info("Training bland models took %.3f seconds", time.time() - ts_)

    return bmodels


@cache_to("model.pkl", use_version=True)
def train_model(datadir, fin):
    """Load dataset and train mineral model.

    Parameters
    ----------
    datadir: str
        the directory where the datasets are stored
    fin: list
        list of feature masks (one per model)

    Returns
    -------
    models: list[HBM]
        list of trained models
    """
    xtrain, ytrain, ids = load_data(datadir)

    def _train(fm_):
        gmm2 = HBM(only_class=True, prior=HBMPrior(**CONF['model_params']))
        gmm2.fit(cp.norm_minmax(xtrain[:, fm_], axis=1), ytrain, ids)
        return gmm2

    ts_ = time.time()
    if len(fin) == 1:
        models = [_train(fin[0])]
    else:
        models = Parallel(n_jobs=N_JOBS)(delayed(_train)(fm_) for fm_ in fin)
    logging.info("Training models took %.3f seconds", time.time() - ts_)

    return models


def compute_bland_scores(if_, bmodels):
    """Compute bland scores, with parallelism if using an ensemble.

    Parameters
    ----------
    if_: ndarray
        array n_spectra x n_channels of unratioed spectra
    bmodels: tuple
        tuple (models, fin) of list of models and list of their feature masks

    Returns
    -------
    slog: ndarray
        blandness scores for the spectra
    """
    models, fin = bmodels
    if len(models) == 1:  # single model, return it
        return models[0].predict_proba(cp.normr(if_[:, fin[0]]))[:, 0]

    # return ensemble average
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(lambda x, m: m.predict_proba(x)[:, 0])(
            cp.normr(if_[:, f]), m) for m, f in zip(*bmodels))
    return np.add.reduce(scores)


def compute_scores(if_, models, ww_):
    """Compute model scores, with parallelism if using an ensemble.

    Parameters
    ----------
    if_: ndarray
        array n_spectra x n_channels of ratioed spectra
    bmodels: tuple
        tuple (models, fin) of list of models and list of their feature masks

    Returns
    -------
    sumlog: ndarray
        array n_spectra x n_classes with classification scores for the spectra
    """
    mods, fin = models
    if len(mods) == 1:  # single model, return it
        return ww_[0] * mods[0].predict_proba(
            cp.norm_minmax(if_[:, fin[0]], axis=1), llh=False)

    # return ensemble average
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(lambda x, m: m.predict_proba(x, llh=False))(cp.norm_minmax(
            if_[:, f], axis=1), m) for m, f in zip(*models))
    return np.add.reduce([s * w for s, w in zip(scores, ww_)])


def _merge_clays(pred):
    """Merge Al clays together because they are hard to differentiate."""
    smectite, clays = CONF['clays']
    pred[np.isin(pred, clays)] = smectite
    return pred


def filter_predictions(probs, classes, merge_clays=True, thr=0.0, kls_thr=()):
    """Return predicted probabilities, removing low-confidence entries.

    Parameters
    ----------
    probs: ndarray
        array n_pixel x n_classes of prediction probabilities
    classes: ndarray
        map from class position in the probability array to class label
    merge_clay: bool
        merges predictions of Al smectites (default: True)
    thr: float
        global class confidence threshold (if kls_thr is empty)
    kls_thr: tuple
        pair of (low, high) thresholds for class confidences; the former for
        well represented classes (five or more dataset images) and the latter
        for the rest

    Returns
    -------
    pred: ndarray
        the filtered predictions, with low-confidence predictions set to 0
    pred0: ndarray
        the predictions before filtering
    pp_: ndarray
        the confidence of the predictions
    """
    pred = np.argmax(probs, axis=-1)
    pp_ = np.take_along_axis(probs, pred[..., None], axis=-1).squeeze()
    pred = _merge_clays(classes[pred]) if merge_clays else classes[pred]
    pred0 = pred.copy()

    logging.debug("Unique predictions before filtering: %s", np.unique(pred))
    logging.debug("Prediction confidence range: min=%.4f, max=%.4f, mean=%.4f", pp_.min(), pp_.max(), pp_.mean())

    if kls_thr:
        thr_low, thr_high = kls_thr
        high_classes = [1, 2, 3, 4, 10, 12, 16, 17, 25, 29, 33, 35, 40, 36, 37, 38, 39]
        pred[np.isin(pred, high_classes) & (pp_ < thr_high)] = 0
        pred[(~np.isin(pred, high_classes)) & (pp_ < thr_low)] = 0
    else:
        pred[pp_ < thr] = 0

    logging.debug("Unique predictions after filtering: %s", np.unique(pred))
    return pred, pred0, pp_

def _to_coords(indices, shape):
    """Convert a list of indices into a list of x,y coordinates."""
    coords = np.flip(np.stack(np.unravel_index(indices, shape)), axis=0)
    return coords.T.astype(np.uint16)


def _region_size(classes):
    """Get class threshold; by default, 5 for each class."""
    return np.full((len(classes),), 5)


def evaluate_regions(if_, im_shape, pred, pp_, **kwargs):
    """Compute contiguous regions and refine predictions.

    Parameters
    ----------
    if_: ndarray
        spectra after spike removal and before mineral classification
    im_shape: tuple
        shape of the image as (height, width)
    pred: ndarray
        image predictions as a flattened array
    pp_: ndarray
        prediction confidences as a flattened array
    if0: ndarray
        spectra after bad pixel removal but before spike removal
    dilate: int
        dilation to apply before finding the connected components
    region_size: function
        function returning the region size thresholds for the given classes

    Returns
    -------
    avgs: list
        a list of detected patches with the following fields:

        pred: int
            predicted class
        avg: ndarray
            average ratioed spectrum on the patch
        size: int
            size of the patch
        coords: ndarray
            coordinates of the pixels in the patch
        coords_full: ndarray
            coordinates of the pixels, including the ones with confidences
            lower than CONF['match_threshold']
    """
    if0 = kwargs.pop('if0', None)   # unratioed spectra
    dilate = kwargs.pop('dilate', 2)  # dilation factor for connected comp.
    region_size = kwargs.pop('region_size', _region_size)
    if kwargs:
        raise ValueError(f"Unrecognized parameters: {list(kwargs.keys())}")

    avgs = []
    for kls, thr, regs in cp.regions(pred.reshape(im_shape), _region_size, dilate=kwargs.get('dilate', 2)):
        for reg in regs:
            reg_good = np.full_like(pred, False, dtype=bool)
            reg_good[reg] = True
            reg_good &= pp_ > CONF['match_threshold']

            if np.sum(reg_good) < thr:
                continue

            avgs.append({
                'pred': kls,
                'avg': np.mean(if_[reg_good, :], axis=0),
                'size': np.sum(reg_good),
                'coords': _to_coords(reg_good.nonzero(), im_shape),
                'coords_full': _to_coords(reg, im_shape)
            })

    logging.debug("Number of regions evaluated: %d", len(avgs))
    return avgs


def _merge_region(regs, kls):
    """Merge region attributes."""
    def _l(lst, field):
        return [x[field] for x in lst]

    sizes = _l(regs, 'size')
    avg = sum(s*x for s, x in zip(sizes, _l(regs, 'avg'))) / np.sum(sizes)

    res = {'pred': kls, 'coords': np.concatenate(_l(regs, 'coords')),
           'coords_full': np.concatenate(_l(regs, 'coords_full')),
           'size': np.sum(sizes), 'avg': avg}
    if 'avg0' in regs[0]:
        res.update(avg0=sum(
            s*x for s, x in zip(sizes, _l(regs, 'avg0'))) / np.sum(sizes))

    return res


def merge_regions(avgs, merge_classes=True):
    """Merge regions with the same label.

    Parameters
    ----------
    avgs: list[dict]
        list of regions from 'evaluate_regions'
    merge_classes: bool
        merge classes with the same BROAD_NAMES (default: True)

    Returns
    -------
    regions: list[dict]
        merged regions with the same format as the input regions
    """
    regions = {}
    for avg in avgs:
        pred = avg['pred']
        # use alias if defined, otherwise keep class unchanged
        pred = cl.ALIASES_EVAL.get(pred, pred) if merge_classes else pred
        regions[pred] = regions.get(pred, []) + [avg]

    return [_merge_region(regs, kls) for kls, regs in regions.items()]

def export_envi_masks(pred, im_shape, output_dir, full_names):
    """
    Export predicted regions as ENVI mask files with mineral names in filenames.

    Parameters
    ----------
    pred : ndarray
        Flattened array of predicted class labels.
    im_shape : tuple
        Shape of the image as (height, width).
    output_dir : str
        Directory where ENVI mask files will be saved.
    full_names : dict
        Dictionary mapping class numbers to their mineral names.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Reshape the predictions to the image shape
    pred = pred.reshape(im_shape)

    # Get unique classes in predictions (excluding background, class 0)
    classes = np.unique(pred)
    classes = classes[classes != 0]

    for cls in classes:
        mask = (pred == cls).astype(np.uint8)  # Create binary mask

        # Use the mineral name from FULL_NAMES for the filename
        mineral_name = full_names.get(cls, f"Class_{cls}")
        sanitized_name = mineral_name.replace(" ", "_")  # Replace spaces with underscores

        mask_filename = os.path.join(output_dir, f"{sanitized_name}.img")

        # Save the mask as a binary file
        mask.tofile(mask_filename)

        # Create an ENVI header file
        header_filename = f"{mask_filename}.hdr"
        with open(header_filename, 'w') as hdr:
            hdr.write("ENVI\n")
            hdr.write(f"description = {{Mask for {mineral_name}}}\n")
            hdr.write(f"samples = {im_shape[1]}\n")
            hdr.write(f"lines = {im_shape[0]}\n")
            hdr.write("bands = 1\n")
            hdr.write("header offset = 0\n")
            hdr.write("file type = ENVI Standard\n")
            hdr.write("data type = 1\n")  # Byte data type
            hdr.write("interleave = bsq\n")
            hdr.write("byte order = 0\n")

        logging.info(f"ENVI mask and header created for {mineral_name} at {output_dir}")


def generate_boxplots(stats, pred, pp_, classes, output_dir):
    """
    Generate and save boxplots for prediction confidence values by mineral class.

    Parameters
    ----------
    stats : dict
        Statistics dictionary containing class-level information.
    pred : ndarray
        Flattened array of predicted class labels.
    pp_ : ndarray
        Flattened array of prediction confidences.
    classes : dict
        Dictionary mapping class numbers to class names.
    output_dir : str
        Directory to save the boxplot image.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Prepare data for boxplots
    class_confidences = {}
    for cls in np.unique(pred):
        if cls == 0:  # Skip background class
            continue
        cls_mask = pred == cls
        confidences = pp_[cls_mask]
        class_name = classes.get(cls, f"Unknown Class {cls}")
        if confidences.size > 0:
            class_confidences[class_name] = confidences

    # Generate boxplot
    plt.figure(figsize=(12, 8))
    labels, data = zip(*class_confidences.items())
    plt.boxplot(data, labels=labels, vert=True, patch_artist=True, notch=True)

    # Set plot attributes
    plt.title("Prediction Confidence by Mineral Class")
    plt.xlabel("Mineral Class")
    plt.ylabel("Confidence")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save the plot in the output directory
    boxplot_path = os.path.join(output_dir, "confidence_boxplots.png")
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    logging.info("Boxplot saved to %s", boxplot_path)

def generate_confidence_map(pp_, im_shape, save_path, upscale_factor=4):
    """Generate and save a high-resolution pixel-preserving confidence map.

    Parameters
    ----------
    pp_: ndarray
        Prediction confidences as a flattened array.
    im_shape: tuple
        Shape of the image as (height, width).
    save_path: str
        Path to save the confidence map.
    upscale_factor: int
        Factor to upscale the resolution of the confidence map.
    """
    # Reshape the confidence map
    confidence_map = pp_.reshape(im_shape)
    confidence_map = (confidence_map - confidence_map.min()) / (confidence_map.ptp() + 1e-8)

    # Upscale using nearest-neighbor interpolation to preserve sharpness
    high_res_map = zoom(confidence_map, upscale_factor, order=0)  # Nearest-neighbor interpolation

    # Save the confidence map as an image
    plt.figure(figsize=(10, 10))
    plt.imshow(high_res_map, cmap='viridis', interpolation='nearest')  # Explicitly disable smoothing
    plt.colorbar(label='Confidence')
    plt.title("Confidence Map")

    # Optionally add a grid overlay for clarity
    plt.grid(visible=True, color='white', linewidth=0.5)
    plt.savefig(f"{save_path}.png", dpi=600, bbox_inches='tight')
    plt.close()

    # Save the confidence map as a compressed NumPy file
    np.savez_compressed(f"{save_path}.npz", high_res_map)


def run_on_images(images, datadir, workdir, thresholds=(0.5, 0.7), plot=False, pred=False, stat=False, mask=False):
    os.makedirs(workdir, exist_ok=True)

    fin0, fin = feat_masks()
    bmodels = train_model_bland(datadir, fin0)
    models = train_model(datadir, fin)
    ww_ = iteration_weights(models[0].classes)

    for im_path in images:
        im_, _ = os.path.splitext(os.path.basename(im_path))
        logging.info("Processing: %s", im_)

        image_workdir = os.path.join(workdir, im_)
        os.makedirs(image_workdir, exist_ok=True)

        # Load image data
        mat = load_image(im_path)
        
        ts_ = time.time()
        if_, rem = cp.filter_bad_pixels(mat['IF'])
        logging.info("Removing bad pixels took %.3f seconds", time.time() - ts_)

        ts_ = time.time()
        im_shape = image_shape(mat)
        if1 = cp.remove_spikes_column(
            if_.reshape(*im_shape, -1), 3, 5).reshape(if_.shape)
        logging.info("Removing column spikes took %.3f seconds", time.time() - ts_)

        ts_ = time.time()
        slog = compute_bland_scores(if1, (bmodels, fin0))
        logging.info("Bland scores took %.3f seconds", time.time() - ts_)

        ts_ = time.time()
        slog_inf = cp.replace(slog, rem, -np.inf).reshape(im_shape)
        if2 = cp.ratio(if1.reshape(*im_shape, -1), slog_inf).reshape(if_.shape)
        logging.info("Ratioing took %.3f seconds", time.time() - ts_)

        ts_ = time.time()
        ifm = cp.remove_spikes(if2.copy(), CONF['despike_params'])
        logging.info("Spike removal took %.3f seconds", time.time() - ts_)

        # Classify
        ts_ = time.time()
        sumlog = compute_scores(ifm, (models, fin), ww_)
        logging.info("Classification took %.3f seconds", time.time() - ts_)

        pred, pred0, pp_ = filter_predictions(sumlog, models[0].classes,
                                              kls_thr=thresholds)
        ts_ = time.time()
        avgs = evaluate_regions(
            if2, im_shape, cp.replace(pred, rem, 0), pp_, if0=if_)
        regs = merge_regions(avgs)
        logging.info("Region parsing took %.3f seconds", time.time() - ts_)

        plotdir = os.path.join(image_workdir, "plots")
        os.makedirs(plotdir, exist_ok=True)

        # Initialize statistics dictionary
        stats = {
            "global": {
                "max_confidence": float(np.max(pp_)),
                "min_confidence": float(np.min(pp_)),
                "avg_confidence": float(np.mean(pp_)),
                "std_confidence": float(np.std(pp_))
            },
            "per_class": {}
        }

        # Export ENVI masks if the mask argument is specified
        if mask:
            masks_dir = os.path.join(image_workdir, "masks")
            export_envi_masks(pred, im_shape, masks_dir, FULL_NAMES)
            logging.info("ENVI masks saved in %s", masks_dir)

        if stat:
            logging.info("Generating statistical evaluation outputs.")
            pixspec, pixlabs, pixims = load_data(datadir)  # Load labeled data
            pixspec = cp.norm_minmax(pixspec, axis=1)  # Normalize

            # Generate predictions on labeled data
            sumlog_stat = compute_scores(pixspec, (models, fin), ww_)
            pred_stat, _, pp_stat = filter_predictions(sumlog_stat, models[0].classes)

            # Compute overall accuracy
            accuracy = accuracy_score(pixlabs, pred_stat)

            # Generate confusion matrix and classification report
            cm = confusion_matrix(pixlabs, pred_stat, labels=models[0].classes)
            class_names = [FULL_NAMES.get(cls, f"Class {cls}") for cls in models[0].classes]

            # Validate indices in models[0].classes
            valid_classes = set(range(len(class_names)))
            xticklabels = [class_names[cls] if cls in valid_classes else "Unknown" for cls in models[0].classes]
            yticklabels = [class_names[cls] if cls in valid_classes else "Unknown" for cls in models[0].classes]

            class_report = classification_report(pixlabs, pred_stat, target_names=[
                FULL_NAMES.get(cls, f"Class {cls}") for cls in models[0].classes
            ], output_dict=True)

            # Normalize the confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0 if any division by zero

            # Ensure statistics directory is defined
            stats_dir = os.path.join(image_workdir, "statistics")
            os.makedirs(stats_dir, exist_ok=True)

            np.savez_compressed(os.path.join(stats_dir, "confusion_matrix.npz"), cm=cm)
            with open(os.path.join(stats_dir, "classification_report.json"), "w") as f:
                json.dump(class_report, f, indent=4)

            # Plot confusion matrix
            plt.figure(figsize=(32, 24))  # Increase figure size for better visualization
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f",  # Use two decimal places
                cmap="Blues",
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                annot_kws={"size": 18},  # Adjust font size for better readability
            )
            plt.title("Confusion Matrix", fontsize=26)
            plt.xlabel("Predicted", fontsize=20)
            plt.ylabel("True", fontsize=20)
            plt.xticks(rotation=45, ha="right", fontsize=16)
            plt.yticks(fontsize=12)
            plt.tight_layout()  # Adjust layout to avoid clipping
            plt.savefig(os.path.join(stats_dir, "confusion_matrix.png"))
            plt.close()

            # Confidence analysis
            correct_confidences = pp_stat[pixlabs == pred_stat]
            incorrect_confidences = pp_stat[pixlabs != pred_stat]

            # Plot confidence distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(correct_confidences, kde=True, color='green', label='Correct Predictions', stat="density")
            sns.histplot(incorrect_confidences, kde=True, color='red', label='Incorrect Predictions', stat="density")
            plt.title("Confidence Distribution")
            plt.xlabel("Prediction Confidence")
            plt.ylabel("Density")
            plt.legend()
            plt.savefig(os.path.join(stats_dir, "confidence_distribution.png"))
            plt.close()

            # Map pixlabs to class names, ensuring indices are valid
            valid_classes = set(range(len(class_names)))
            mapped_labels = [
                class_names[int(cls)] if int(cls) in valid_classes else "Unknown" 
                for cls in pixlabs
            ]

            # Generate the boxplot
            plt.figure(figsize=(14, 8))
            sns.boxplot(x=mapped_labels, y=pp_stat, palette="Set2")
            plt.title("Confidence by Class")
            plt.xlabel("Mineral Class")
            plt.ylabel("Confidence")
            plt.xticks(rotation=45, ha='right')  # Rotate class names for better visibility
            plt.tight_layout()
            plt.savefig(os.path.join(stats_dir, "confidence_boxplot.png"))
            plt.close()

            logging.info("Statistical evaluation outputs saved in %s", stats_dir)

        # Save per-patch and per-image results
        with open(os.path.join(image_workdir, f"{im_}.pkl"), 'wb') as fid:
            pickle.dump([avgs, regs], fid, pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(os.path.join(image_workdir, im_),
                            [pp_, pred, pred0, slog])

        if pred.any():
            rem = rem.reshape(im_shape)
            im_fc = cpl.get_false_colors(if_, rem)
            cpl.show_classes(im_fc / 2, regs, crop_to=cp.crop_region(rem),
                             save_to=os.path.join(plotdir, f"{im_}.pdf"))

        # Save per-patch and per-image results
        with open(os.path.join(image_workdir, f"{im_}.pkl"), 'wb') as fid:
            pickle.dump([avgs, regs], fid, pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(os.path.join(image_workdir, im_),
                            [pp_, pred, pred0, slog])

    # After processing all images, save the statistics to a JSON file
    if stat:
        # Aggregate per-class statistics
        for cls, cls_stats in stats["per_class"].items():
            stats["per_class"][cls]["max_confidence"] = float(np.max(cls_stats["max_confidence"]))
            stats["per_class"][cls]["min_confidence"] = float(np.min(cls_stats["min_confidence"]))
            stats["per_class"][cls]["avg_confidence"] = float(np.mean(cls_stats["avg_confidence"]))
            stats["per_class"][cls]["std_confidence"] = float(np.mean(cls_stats["std_confidence"]))

        # Save confidence map and stats in the statistics directory
        statistics_dir = os.path.join(image_workdir, "statistics")
        os.makedirs(statistics_dir, exist_ok=True)

        confidence_map_path = os.path.join(statistics_dir, f"{im_}_confidence_map")
        generate_confidence_map(pp_, im_shape, confidence_map_path)
        logging.info("Confidence map saved to %s", confidence_map_path)

        stats_file_path = os.path.join(statistics_dir, 'confidence_stats.json')
        with open(stats_file_path, 'w') as f:
            json.dump(stats, f, indent=4)
        logging.info("Confidence statistics saved to %s", stats_file_path)

        # Generate and save boxplots
        generate_boxplots(stats, pred, pp_, FULL_NAMES, statistics_dir)

def get_parser():
    """Get parser for command-line arguments."""
    parser = argparse.ArgumentParser(
        description="A script to train the HBM models and run the evaluation"
                    "on a list of images")

    parser.add_argument('image', type=str, nargs='+',
                        help="CRISM images to process")
    parser.add_argument('--datapath', '-d', type=str, default="datasets",
                        help="Directory where the datasets are stored.")
    parser.add_argument('--workdir', '-w', type=str, default="workdir",
                        help="Directory where the results are stored.")
    parser.add_argument('--thr', '-t', type=float, nargs='+',
                        default=(0.5, 0.7),
                        help="Confidence thresholds for easy and hard classes")
    parser.add_argument("--plot", action="store_true",
                        help="Save detailed per-region plots")
    parser.add_argument("--pred", action="store_true",
                        help="Save the segmentation image")
    parser.add_argument("--stat", action="store_true",
                        help="Perform statistical analysis and generate a confidence map")
    parser.add_argument("--mask", action="store_true",
                        help="Export ENVI mask files for each mineral class")
    return parser

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = get_parser().parse_args()
    run_on_images(args.image, args.datapath, args.workdir, args.thr, args.plot, args.pred, args.stat, args.mask)
