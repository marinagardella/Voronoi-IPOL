#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
This program provides an implementation of the Voronoi method for Document Layout Analysis,
as described in the paper:

[1] K. Kise, A. Sato, y M. Iwata, 
    "Segmentation of Page Images Using the Area Voronoi Diagram», Computer Vision and Image Understanding", 
    vol. 70, n.º 3, pp. 370-382, jun. 1998, doi: 10.1006/cviu.1998.0684.

The implementation is made as faithful as possible to the original method as described in the paper.
Tweaks and potential improvements are optionally enabled by the user via command line switches.

Also, we aim at an efficient and self-contained implementation which is easily portable for comparison with
other methods.

Finally, the code is thoroughly documented and cross-referenced to the paper.

Authors:
* Marina Gardella <marigardella@gmail.com>
* Ignacio Ramírez <nacho@fing.edu.uy>

Other papers involved:

[1] I. Ramírez, “Practical Bulk Denoising Of Large Binary Images,” 
    in 2022 IEEE International Conference on Image Processing (ICIP), Bordeaux, France: 
    IEEE, Oct. 2022, pp. 196–200. doi: 10.1109/ICIP46576.2022.9897678.
"""

#==============================================================================
# IMPORTS
#==============================================================================
#
# core Python modules
#
import os
import math
import sys
import logging
import argparse
import multiprocessing as mp
import copy
#
# third-party modules
#
import numpy as np
from skimage import morphology as skmorpth
from skimage import filters as skfilters
from matplotlib import pyplot as plt

from scipy import spatial
from scipy import signal

from util import * 
#
#==============================================================================
# CONSTANTS
#==============================================================================
#
DPI = 150
#
#==============================================================================
# FUNCTIONS
#==============================================================================
#
def get_logger(fname):
    """
    Create a logger object
    """
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("VORONOI")
    flog = open(fname,'w')
    logger.addHandler(logging.StreamHandler(flog))
    return logger


def close_logger(logger):
    """
    Close a logger object
    """
    if logger is not None:
        for handler in logger.handlers:
            logger.removeHandler(handler)
            handler.close()


def get_binary_image(img,args,logger):
    """
    " The input to the pipeline described in [1] is assumed to be a binary image where the characters are ON pixels.
    " We make sure that this is the case regardless of the nature of the input image that was provided to this program.
    " This may imply:
    " - removing alpha channel information (transparency)
    " - converting from color (RGB) to grayscale
    " - binarization.
    " By default we use  Otsu method to binarize the image, but other methods are possible.
    " The main drawback of Otsu's method is that it is global and may not be the best choice 
    " for images that have a non-uniform background, such as shades from images scanned from books
    " or derived from pictures instead of a flatbed scanner.
    "
    """ 
    #
    # if 4 channels, assume 4th is alpha and discard it
    #
    if len(img.shape) == 4:
        img = img[:,:,:3]
    #
    # if color, make it grayscale
    #
    if len(img.shape) > 2:
        img = img.mean(axis=2)
    #
    if img.dtype == 'bool':
        #
        # Already a boolean image.
        # The only issue is that it may be be inverted w.r.t. what we expect, that is, foreground pixels 
        # should be '1' and background '0'.
        # 
        # below we use a heuristic assuming that the background is the most common value.
        #
        if np.mean(img) > 0.5:
            img = ~img
        return img
    else:
        #
        # binarize
        #
        param = args.binarization_param
        param /=255
        method = args.binarization_method.lower()
        if img.dtype != 'bool':
            if method == "otsu":
                thresh = skfilters.threshold_otsu(img)
                return img < thresh
            elif method == "local":
                return img < skfilters.threshold_local(img,block_size=int(param))
            elif method == "threshold":
                return img < param*np.max(img)
        return img


def get_connected_components(img,args,logger):
    #
    # OPTIONAL STEP: remove small blobs
    #
    if args.remove_blobs:
        img = skmorpth.remove_small_objects(img, min_size=args.remove_blobs)
        if args.save_images == "all":
            write_img(f"{args.output}1b_removed_small_objects.{args.image_ext}",~img)
    #
    # now we extract the connected components.
    #
    labels = skmorpth.label(img)
    logger.info(f'Number of connected components found in the image: {np.max(labels)} ')
    if args.save_images == "all":
        write_img(f"{args.output}2_components.{args.image_ext}",labels)
    return labels


def get_borders(labels,args,logger):
    """
    This returns the _inner_ borders of the connected components.
    A regular image borders computation does not work here as it can produce
    "touching" borders. This guarantees that things don't touch by accident.
    """
    aux = skmorpth.erosion(labels)
    return labels != aux


def sample_border_points(borders,args,logger):
    """
    The paper [1] does not specify how to sample the border points.
    We obtain such points by intersecting the borders of the connected components
    with a regular square grid. This is simple but likely not optimal as it doesn't
    take into account the curvature of the borders.
    
    The coarseness of the grid is a parameter that can be adjusted via command line.
    A coarseness of 1 will produce the actual, exact, Area Voronoi Diagram, as no
    subsampling will take place.
    """  

    if args.subsample_method == 'grid':
        grid_width = int(args.subsample_param)
        if grid_width > 1:
            grid = np.zeros(borders.shape,dtype=bool)
            grid[::grid_width,::grid_width] = True
            return borders & grid
        else:
            return borders
    elif args.subsample_method == 'random':
        p = float(args.subsample_param)
        if p < 1:
            rng = np.random.default_rng(args.seed)
            grid = rng.uniform(size=borders.shape) < p            
            return borders & grid
        else:
            return borders
    elif args.subsample_method == 'file':
        fname = args.subsample_param
        if not os.path.exists(fname):
            logger.warning(f'Sample points file {fname} not found.')
            return borders
        else:
            points = np.loadtxt(fname).astype(int)
            points_img = np.zeros(borders.shape)
            for p in points:
                points_img[p[1],p[0]] = 1 # x,y pairs
            return points_img
    else:
        logger.warning(f'Invalid sampling method: {args.subsample_method}')
        return borders


def get_point_voronoi(borders,args,logger):
    """  
    Compute the Voronoi diagram of the border points.
    This is done using Scipy's implementation, which returns a Voronoi object.
     This object describes the Voronoi diagram and several of its properties
     such as the vertices, the ridges, the regions as lists of indices to the input points and
     vertices.
     it is important to understand this list, so we quote the documentation:
    skfilters.
     * points: 
       coordinates of input points as a Numpy array of size (npoints, ndim) and type double
     * vertices: 
       coordinates of the computed vertices as a numpy array of size (nvertices, ndim) and type double
     * ridge_points: 
       This is a numpy array of shape (nridges, 2) and type int where each row lists the indices 
       of the _input points_ (attribute _points_ above) between which each Voronoi ridge lies. 
       This stems from a property of Voronoi diagrams: each ridge in a Voronoi cell corresponds to the
       medial axis between a pair of input points.
     * ridge_vertices:
       This is a numpy array of shape (nridges, 2) and type int where each row lists the indices of
       two _vertices_ (attribute _vertices_ above) that are the endpoints of each Voronoi ridge as a line segment.
     * regions (NOT NEEDED):
       This is a list of tuples of integers. Each list indicates the indices of the _vertices_ that form
       the perimeter of a given region (cell) in the Voronoi diagram.
       The list will contain -1 for vertices that are not associated with a Voronoi region, that is, points that
       occur at the edges of the image.
       We DON'T care about this list. It's of no use to the algorithm.
       Besides, the information becomes useless once we start pruning ridges.
     * point_region (NOT NEEDED):
       This is a numpy array of integers of size (npoints) that indicates the index of the region (cell) in the
       Voronoi diagram that each _input point_ belongs to. If the point is at the edge of the image, the value will be -1.
        We DON'T care about this list. It's of no use to the algorithm.
    """
    borders_x, borders_y = np.where(borders)
    if not len(borders_x):
        return None
    border_points = np.array(list(zip(borders_x,borders_y)))
    return  spatial.Voronoi(border_points)


def eval_redundancy_criterion(points,labels,ridge_points,ridge_vertices):
    """
    " prune ridges that are not separating different connected components
    " this means that the two input points that generate the ridge belong to 
    " the same connected component.
    """
    assert(len(ridge_points) == len(ridge_vertices))
    nridges = len(ridge_points)
    criteria = np.zeros(nridges,dtype=bool)
    for n in range(nridges): # these are _indexes_ to points
        # get the two input points that generate this ridge
        i_1, i_2 = ridge_points[n]
        p_1, p_2 = points[i_1], points[i_2]
        c_1 = labels[p_1[0],p_1[1]]
        c_2 = labels[p_2[0],p_2[1]]
        #
        # keep them if the components are not the same
        #
        criteria[n]  = (c_1 != c_2)
    return criteria


def compute_ridge_features(points,labels,ridge_points,logger):
    """
    " compute the features dE and arE for each ridge E
    """
    #if compiled:
    #    logger.info(' Using compiled C extension.')
    #    vorlib.compute_ridge_features(points,labels,ridge_points)
    #
    #else:
    #    logger.info('Compiled extension not found!! Falling back to Python (much slower).')

    dE = list()
    arE = list()
    #
    # we create a sort of hash for each pair of components
    # so we can find them easily
    #
    component_areas = np.bincount(labels.ravel())
    separating_comp_keys = list()
    separating_comp_dists = dict()
    k = 2**32
    for i_1,i_2 in ridge_points:
        p_1 = points[i_1]
        p_2 = points[i_2]
        c_1 = int(labels[p_1[0],p_1[1]])
        c_2 = int(labels[p_2[0],p_2[1]])
        d = np.linalg.norm(p_1-p_2)
        dE.append(d)
        area_1 = component_areas[c_1]
        area_2 = component_areas[c_2]
        arE.append(max(area_1,area_2)/min(area_1,area_2))
        key = c_1*k+c_2
        separating_comp_keys.append(key)
        if key in separating_comp_dists.keys():
            separating_comp_dists[key] = min(separating_comp_dists[key],d)
        else:
            separating_comp_dists[key] = d
    #
    # now dE is the minimum value of dr among all the ridges that separate
    # the same two components. We compute such value for each pair of separated components
    #
    #for key in separating_comp_dists.keys():
    #    dE = separating_comp_dists[key]
    #    dE = int(round(np.min(dE)))
    #    separating_comp_dists[key] = dE
    #
    # and then overwrite the value of dE for those ridges E which separate the same
    # two  separating components
    #
    dE = np.array([ separating_comp_dists[key] for key in separating_comp_keys],dtype=np.int32)
    arE = np.array(arE,np.float32)
    return dE,arE


def compute_thresholds(dE,args,logger):
    """
    " Compute the distance thresholds t1 and t2, and the area threshold ta
    """
    dist_hist = np.bincount(dE) # discrete valued histogram
    if args.save_images == "all":
        plt.close('all')
        plt.figure(figsize=(10,5))
        plt.plot(dist_hist,lw=1)
        plt.xlim(0,200)
        plt.grid(True)
        plt.xlabel('distance (px)')
        plt.ylabel('frequency')
        plt.legend()
        plt.savefig(f"{args.output}7a_distance_histogram.{args.image_ext}",bbox_inches="tight",dpi=DPI)
        plt.close('all')
    #
    #
    # The smoothing is done using a window averaging with window size 5 (2w+1 with w=2)
    # The parameter w can be modified via the command line; the default is 2, as in [1]
    #
    #
    #" The paper [1] uses a square filter of size 5 (2w+1 with w=2) to smooth the histogram of distances.
    #
    w = args.parameter_w
    k = np.ones(2*w+1)/(2*w+1)
    dist_hist_smoothed = np.convolve(dist_hist,k, mode='same')
    #
    # The next step is to find the 'two largest peaks'
    # of the smoothed distance histogram and then 
    # computing the thresholds from these peaks and the parameter 't'.
    #
    # The problem with the original paper here is that these peaks are not 
    # completely defined!! 
    # 
    # Even with smoothing, depending on the input 
    # and the parameters, the 'two largest peaks' might correspond to the same
    # mode of the histogram if there is some noise in the uphill or downhill
    # enough to make a small dent on either side.
    #
    # We recommend a stronger smoothing or else some other parameter such
    # as a minimum distance betwen peaks, but the goal of this implementation
    # is to adhere to the paper as much as possible.
    #
    peak_indexes, props = signal.find_peaks(dist_hist_smoothed,distance=args.min_peaks_dist) # WARNING!! This might not produce what's 'advertised'
    #
    # WARNING: CHECK THE LOG!!! If d_1 and d_2 are too close, the algorithm will fail.
    #
    if len(peak_indexes) == 0:
        logger.warning(f'\tNo peaks found in the histogram. Bailing out.')
        return -1,-1,-1
    elif len(peak_indexes) == 1:
        logger.warning(f'\tOnly one peak found in the histogram!! Setting v1=v2.')
        v1 = peak_indexes[0]
        h1 = peak_heights[0]
        v2 = v1
        two_largest_peaks = [v1,v1]
        two_largest_heights = [h1,h1]
    else:
        peak_heights = dist_hist_smoothed[peak_indexes]
        two_largest_peak_indexes = np.argsort(peak_heights)[-2:]
        two_largest_peaks = peak_indexes[two_largest_peak_indexes]
        two_largest_heights = peak_heights[two_largest_peak_indexes]
        v2,v1 = max(two_largest_peaks),min(two_largest_peaks)
        logger.info(f'Largest peaks:')
        logger.info(f'\th_1={round(two_largest_heights[0])} at v_1={v1}') 
        logger.info(f'\th_2={round(two_largest_heights[1])} at v_2={v2}') 
        if (np.abs(two_largest_peaks[1]-two_largest_peaks[0])) < 5:
            logger.warning(f'\tTwo largest peaks are way too close!!')

    if args.save_images == "all" or args.save_images == "important":
        plt.close('all')
        plt.figure(figsize=(10,5))
        plt.xlim(0,200)
        plt.grid(True)
        plt.plot(dist_hist_smoothed,lw=1,label='histogram')
        plt.scatter(two_largest_peaks,two_largest_heights,label='peaks',color='red')
        plt.xlabel('distance (px)')
        plt.ylabel('frequency')
        plt.legend()
        plt.savefig(f"{args.output}7b_dist_hist_smoothed.{args.image_ext}",bbox_inches="tight",dpi=DPI)
        plt.close('all')
    #
    # v1 is the smaller distance threshold, v2 is the largest
    #
    #
    # t1 is just v1
    #
    if args.parameter_t1 < 0:
        t1 = v1
    else:
        t1 = args.parameter_t1
    #
    # t2 is such that t2 > v2 and h(t2) = t * h(v2) where h is the distance histogram.
    #
    #
    # the paper proposes a linear interpolation to find this threshold
    # but this is surely overkill.
    # we just find the first point where the histogram is below h2.
    #
    # note: the user may manually override this value for analysis/testing purposes.
    #
    maxd = len(dist_hist_smoothed)
    if args.parameter_t2 < 0:
        h = dist_hist_smoothed # a short alias
        v0 = int(math.floor(v2))
        if v0 >= len(dist_hist_smoothed) - 1:
            t2 = v0
        else:
            # we obtain a linear approximation of h(v2)
            v1 = v0+1
            h0 = h[v0]
            h1 = h[v1]
            h2 = h0 + (v2-v0)*(h1-h0)
            ht2 = args.parameter_t * h2
            # logger.info(f"Target value t * h(v2) = {ht2:7.3f}")
            t0 = v1-1
            while t0 < (maxd-1) and h[t0] > ht2:
                t0 += 1
            if h[t0] > ht2: # bailed out due to histogram; unlikely
                logger.warning(f"Reached max distance  while looking for h(t2) > t * h(v2). Setting t2=maxd={maxd}.")
                t2 = maxd
            else:
                # roll back t0 last value for which h[t0] > ht2
                t0 -= 1
                h0 = h[t0]
                h1 = h[t0+1] 
                # h(t2) = h(t0) + (h(t1)-h(t0))*(t2-t0)/(t1-t0) 
                # h(t2) = h0 + (h1-h0)*(t2-t0) => [h(t2)-h0]/(h1-h0) = t2 - t0
                #       => t2 = t0 + (ht2-h0)/(h1-h0)
                t2 = t0 + (ht2-h0)/(h1-h0)
    else:
        t2 = args.parameter_t2
    logger.info(f'Distance thresholds: t1={t1} and t2={t2}')
    #
    # a third area threshold is fixed to 40 in the paper.
    # this corresponds (according to [1]) to the largest area ratio between 
    # two characters of the same font and size,
    # I guess this would be for example between 
    # a capital M or B or whichever has more 'ink' and a dot (.)
    #
    # logger.info(f'Area threshold: Ta={args.parameter_ta}')
    return t1,t2,args.parameter_ta


def eval_pruning_criteria(dE, arE, t1, t2, ta):
    assert(len(dE) == len(arE))
    N = len(dE)
    satisfies_eq8 = [d< t1 for d in dE]
    satisfies_eq9 = [d/t2 + a/ta < 1 for d,a in zip(dE,arE)]
    return satisfies_eq8, satisfies_eq9


def eval_loop_condition(ridge_vertices):
    v,n = np.unique(ridge_vertices,return_counts=True)
    shared = v[n > 1]
    return [(rv[0] == -1 or rv[0] in shared) and (rv[1] == -1 or rv[1] in shared) for rv in ridge_vertices]


def prune_by_loop_condition(ridge_vertices,ridge_points,logger):
    """
    " prune ridges that do not belong to the frontier between text areas.
    " this means that the ridge either reaches the border of the image or shares a vertex
    " with another frontier.
    """
    #if compiled:
    #    logger.info(' Using compiled C extension.')
    #    vorlib.prune_by_loop_condition(ridge_vertices,ridge_points)
    #else:
    #    logger.warning('Compiled extension NOT found!! Falling back to Python (much slower).')
    iter = 0
    assert(len(ridge_vertices) == len(ridge_points))
    while True:
        keep = eval_loop_condition(ridge_vertices)
        npruned = len(ridge_vertices) - np.sum(keep)
        if not npruned:
            break
        ridge_vertices = ridge_vertices[keep]
        ridge_points = ridge_points[keep]
        iter += 1
        
    return np.array(ridge_vertices),np.array(ridge_points)


def area_voronoi_dla(fname,args):
    """
    " Main algorithm defined in paper [1]
    """
    #
    #------------------------------------------------------------------------- 
    # 0. PREPARATION
    #------------------------------------------------------------------------- 
    #
    logger = get_logger(args.log_file)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    final_output = f"{args.output}9_final_area_voronoi.npz"
    if os.path.exists(final_output) and not args.recompute:
        logger.info(' ALREADY COMPUTED. Use --recompute to force.')
        close_logger(logger)
        return
    
    input_img = read_img(fname)
    if args.save_images == "all":
        plt.imsave(f"{args.output}0_input.png",input_img)
    binary_img = get_binary_image(input_img,args,logger)
    #
    #------------------------------------------------------------------------- 
    # 1. BINARIZATION
    #------------------------------------------------------------------------- 
    #
    # The original paper [1] assumes a binary input. No mention is made of the 
    # binarization process despite the description in the experiments section 
    # (Section 5, page 6) indicates that the authors produced their own dataset 
    # via scanning. 
    #
    if args.save_images == "all" or args.save_images == "important":
        write_img(f"{args.output}1_binarized.{args.image_ext}",~binary_img)

    labels = get_connected_components(binary_img,args,logger)
    NC = np.max(labels)
    if NC < 2:
        logger.warning(f'Less than two connected components in the image. Solutioh is trivial.')
        np.savez(final_output,ridge_points=[],ridge_vertices=[])
        close_logger(logger)
        return
    #
    #------------------------------------------------------------------------- 
    # 2. BORDERS EXTRACTION
    #------------------------------------------------------------------------- 
    #
    # In order to compute the (approximate) area Voronoi diagram, the authors
    # sample the borders of the connected components and construct the ordinary 
    # point Voronoi diagram from there. 
    #
    # The borders are obtained by eroding the connected components, so that
    # the resulting points are inside the corresponding components.
    # 
    borders_img = get_borders(labels,args,logger)
    logger.info(f'Number of border points: {np.sum(borders_img)}.')
    if args.save_images == "all":
        write_img(f"{args.output}3_borders.{args.image_ext}",~borders_img)
    #
    #------------------------------------------------------------------------- 
    # 3. BORDERS SUBSAMPLING
    #------------------------------------------------------------------------- 
    #
    # The paper does not specify how to subsample the borders. 
    # By default we do so randomly with a probability of 0.1 
    #
    border_points_img = sample_border_points(borders_img,args,logger)
    if args.save_images == "all":
        write_img(f"{args.output}4_sampled_borders.{args.image_ext}",~border_points_img)
    logger.info(f'Sampled border points: {np.sum(border_points_img)}.')
    #
    #------------------------------------------------------------------------- 
    # 4. POINT VORONOI DIAGRAM
    #------------------------------------------------------------------------- 
    #
    # Now we compute the Voronoi diagram of the border points.
    #
    if np.sum(border_points_img) < 4:
        logger.warning("Not enough points to construct diagram.")
        return

    pvd = get_point_voronoi(border_points_img,args,logger)
    if pvd is None:
        logger.warning("No borders detected!! Nothing to be done.")
        close_logger(logger)
        return

    if len(pvd.ridge_points) == 1:
        logger.warning("Only two connected components. Solution is the only ridge there is.")
        np.savez(final_output,ridge_points=[],ridge_vertices=[])
        close_logger(logger)
        return
    #
    # we extract the relevant data from the Voronoi object
    #
    ridge_points = np.array(pvd.ridge_points).astype(np.int32)
    points = pvd.points.astype(np.int32)
    ridge_vertices = np.array(pvd.ridge_vertices).astype(np.int32)
    vertices = pvd.vertices.astype(np.int32)
    nridges = len(ridge_points)
    nvertices = len(vertices)
    logger.info(f'Point Voronoi diagram:')
    logger.info(f'\tNumber of ridges: {nridges}.')
    logger.info(f'\tNumber of vertices: {nvertices}.')
    np.savez(f"{args.output}5_point_voronoi.npz",points=points,vertices=vertices,ridge_vertices=ridge_vertices,ridge_points=ridge_points) 

    if args.save_images == "all":
        plotimg = plot_voronoi(input_img, points, vertices, ridge_points, ridge_vertices)
        write_img(f"{args.output}5_point_voronoi.{args.image_ext}",plotimg)
    #
    #------------------------------------------------------------------------- 
    # 4. APPROXIMATE AREA VORONOI DIAGRAM
    #------------------------------------------------------------------------- 
    #
    not_redundant = eval_redundancy_criterion(points,labels,ridge_points,ridge_vertices)
    ridge_points = ridge_points[not_redundant]
    ridge_vertices = ridge_vertices[not_redundant]
    nridges = len(ridge_vertices)
    logger.info(f'Remaining ridges after pruning redundant ones: {nridges}.')
    np.savez(f"{args.output}6_pruned_redundant.npz",ridge_vertices=ridge_vertices,ridge_points=ridge_points) 
    if args.save_images == "all" or args.save_images == "important":
        plotimg = plot_voronoi(input_img, points, vertices, ridge_points, ridge_vertices)
        write_img(f"{args.output}6_pruned_redundant.{args.image_ext}",plotimg)
    #
    #------------------------------------------------------------------------- 
    # 5. PRUNING BY FEATURES
    #------------------------------------------------------------------------- 
    #
    # Now ridges of the Area Voronoi Diagram that separate what is assumed to 
    # be parts of the same text area are removed. This is done using a set of
    # features. These are:
    #
    # a) d(E), the distances of all ridges (E) to their closest connected 
    #    component,
    # b) ar(E) the area ratio a_r(E) between the components divided by ridge E
    #
    dE,arE = compute_ridge_features(points,labels,ridge_points,logger)
    np.savez(f"{args.output}7_ridge_features.npz",dE=dE,arE=arE)
    #
    # The criteria used to remove ridges using the features depend on three
    # thresholds. 
    # T_{d_1} and T_{d_2} are derived from a smoothed version of the 
    # histogram of d(E).
    # A Third threshold, T_a, is a parameter of the algorithm which is fixed to
    # 40 for reasons detailed in the paper.
    #    
    t1,t2,ta  = compute_thresholds(dE,args,logger)
    if t1 < 0: # indicates an error in compute_threshold
        logger.warning("Error computing thresholds!")
        close_logger(logger)
        return
    #
    # Now the actual pruning takes place. 
    # A ridge E is removed if _either_ of the two following conditions hold:
    #
    # i) d(E)/T_{d_1} < 1                   (8)
    # ii) d(E)/T_{d_2} + a_r(E)/T_{a}  < 1  (9)
    #
    # Section 4.3 is ambiguous here, because it says that the ridges pruned
    # are those satisfying eqs. (8) AND (9), but the text right after (8)
    # and (9) indicates that an edge is pruned if EITHER (8) OR (9) is met.
    # As the text is more precise when such criteria were introduced, we 
    # assume that this is an OR, not an AND.
    #
    ta = args.parameter_ta
    eq8, eq9 = eval_pruning_criteria(dE,arE,t1,t2,ta)
    only_eq8 = np.sum(np.logical_and(eq8,np.logical_not(eq9)))
    only_eq9 = np.sum(np.logical_and(eq9,np.logical_not(eq8)))
    eq8_and_9 = np.sum(np.logical_and(eq9,eq8))
    prune     = np.logical_or(eq8,eq9)
    not_prune = np.logical_not(prune)
    logger.info(f'Pruning reason:')
    logger.info(f'\tOnly by eq8: {np.sum(only_eq8)}.')
    logger.info(f'\tOnly by eq9: {np.sum(only_eq9)}.')
    logger.info(f'\tBoth       : {np.sum(eq8_and_9)}.')
    np.savez(f"{args.output}7b_ridge_criteria.npz",eq8=eq8,eq9=eq9)
    #
    # Parenthesis: analysis of the conditions.
    #
    # Here we optionally produce a map where the different pruned ridges are
    # are painted with colors which depend on the conditions used for
    # pruning them. The surviving ridges are also shown.
    #
    # only satisfying eq8 gives color (1,1,0) -> yellow 
    # only satisfying eq9 gives color (1,0,1) -> magenta
    # satisfying bot eq88 and eq9 gives color (1,0,0) -> red
    # 
    
    if args.save_images == "all" or args.save_images == "important":
        ridge_colors = np.outer(np.ones(len(ridge_vertices)),np.array(RIDGE_COLOR)).astype(np.uint8)
        for j in range(len(ridge_vertices)):
            if eq8[j] and eq9[j]: # very pruned
                ridge_colors[j,:] = [255,192,64]
            elif eq8[j]:          # strangely pruned
                ridge_colors[j,:] = [255,0,0]
            elif eq9[j]:          # reasonably pruned
                ridge_colors[j,:] = [0,192,32]
        ridge_colors = list(tuple(r) for r in ridge_colors)
        plotimg = plot_voronoi(input_img, points, vertices, ridge_points, ridge_vertices,ridge_color=ridge_colors)
        cwd = os.path.dirname(__file__)
        legend_img = read_img(os.path.join(cwd,'fig/legend_medium_alpha75.png'))
        alpha = legend_img[:,:,3]
        alpha = alpha/(max(1e-4,np.max(alpha)))
        legend_img = legend_img[:,:,:3]
        hl,wl,_ = legend_img.shape
        hp,wp,_ = plotimg.shape
        #plotimg[0:hl,wp-wl:,:] = legend_img[:,:,0]
        plotimg[0:hl,wp-wl:,0] = (1-alpha)*plotimg[0:hl,wp-wl:,0] + alpha*legend_img[:,:,0]
        plotimg[0:hl,wp-wl:,1] = (1-alpha)*plotimg[0:hl,wp-wl:,1] + alpha*legend_img[:,:,1]
        plotimg[0:hl,wp-wl:,2] = (1-alpha)*plotimg[0:hl,wp-wl:,2] + alpha*legend_img[:,:,2]
        write_img(f"{args.output}8_pruned_by_features.{args.image_ext}",plotimg)
    #
    # Now we remove the ridges that do not satisfy the criteria
    #
    ridge_points= ridge_points[not_prune]
    ridge_vertices = ridge_vertices[not_prune]
    nridges = len(ridge_vertices)
    logger.info(f"Remaining ridges after pruning by criterias (8) and (9): {nridges}")
    np.savez(f"{args.output}8_pruned_by_features.npz",ridge_vertices=ridge_vertices,ridge_points=ridge_points) 
    #
    #------------------------------------------------------------------------- 
    # 6. LOOP CONDITION
    #------------------------------------------------------------------------- 
    #
    # the final step is to prune the ridges that do not belong to frontiers
    # between text areas.  These are identified by the authors as those
    # that satisfy one of two conditions: either they reach the border of the image
    # or share a vertex with another frontier, which may subdivide the image in 
    # other directions.
    #
    # The above condition needs to be checked repeatedly, and thus the authors
    # refer to it as a "loop condition". The name is not very descriptive, as it
    # the actual "loop condition" is met while there are ridges to prune.
    #
    ridge_vertices,ridge_points = prune_by_loop_condition(ridge_vertices,ridge_points,logger)
    nridges = len(ridge_vertices)
    logger.info(f"Number of ridges in final diagram: {nridges}.")
    if args.save_images != "none":
        plotimg = plot_voronoi(input_img, points, vertices, ridge_points, ridge_vertices)
        write_img(f"{args.output}9_final_area_voronoi.{args.image_ext}",plotimg)
        plt.close('all')
    np.savez(f"{args.output}9_final_area_voronoi.npz",ridge_vertices=ridge_vertices,ridge_points=ridge_points) 
    #logger.info("finished.")
    close_logger(logger)
    #
    #------------------------------------------------------------------------- 
    # -- END ---
    #------------------------------------------------------------------------- 
    #


def area_voronoi_dla_mp(rel_fname,args):
    """
    " Helper function for running using multiple processors.
    " Only used when in batch mode.
    """
    args = copy.deepcopy(args)
    fname = os.path.join(args.base_dir,rel_fname)
    output_rel_dir = os.path.splitext(rel_fname)[0]
    args.output = os.path.join(args.output,output_rel_dir) + os.sep
    args.log_file = os.path.join(args.output,'log.txt')
    if not os.path.exists(args.output):
        os.makedirs(args.output,exist_ok=True)
    area_voronoi_dla(fname,args) 


#
#==============================================================================
# MAIN FUNCTION
#==============================================================================
#
if __name__ == "__main__":
    #
    # command line interface via Python's argparse module
    #
    ap = argparse.ArgumentParser(__name__)
    ap.add_argument("-i", "--input-image", default=None, 
                    help="path to the input image")
    ap.add_argument("-d", "--base-dir", default=".", 
                    help="for batch mode, directory where input files are stored. This should give the full path of each input image when concatenated with an entry from the list.")
    ap.add_argument("-L", "--input-list", default=None, 
                    help="path to a list with input image file names (relative to base dir)")
    ap.add_argument("-l","--log-file", default='voronoi.log', 
                    help="Path to output log file.")
    ap.add_argument("-B", "--binarization-method", type=str, default="otsu", 
                    help="Binarization method for non-binary input images.")
    ap.add_argument("-Y", "--binarization-param", type=float, default=0.5, 
                    help="For binarization methods which require a parameter, this is it.")
    ap.add_argument("-b", "--remove-blobs", type=int, default=4, 
                    help="If set to a value larger than zero, this will remove all blobs whose size is smaller than the specified value.")
    ap.add_argument("-v", "--verbose", action="store_true", 
                    help="Print verbose output. Also enables debug mode.")
    ap.add_argument("-o", "--output", default="voronoi", 
                    help="output prefix for all output files. If an input image list is passed, this is assumed to be a directory and output directories will be created below this prefix for each input image.")
    ap.add_argument("-s","--subsample-method", type=str, default="random", 
                    help="How to subsample the borders. grid: intersect with a regular grid. random: intersect with a Bernoulli. file: load points from a text file.")
    ap.add_argument("-r","--subsample-param", type=str,default='0.1', 
                    help="Subsampling param. Spacing for grids, _inverse_ probability for 'random', or file name for 'file'.")
    ap.add_argument("-w","--parameter-w", type=int, default=2, 
                    help="This is the parameter 'w' in the original paper and defines a window of size 2*w+1 over which the histogram is smoothed by averaging.")
    ap.add_argument("-t","--parameter-t", type=float, default=0.34, 
                    help="Parameter 't' in the paper.")
    ap.add_argument("-a","--parameter-ta", type=float, default=40, 
                    help="This is the parameter 'Ta' in the original paper.")
    ap.add_argument("-M","--min-peaks-dist", type=int, default=1, 
                    help="Minimum distance between peaks in the histogram. Faithful implementation to the paper is 1, but this is clearly a bad idea. Use a value of 5 or more if you get a warning saying peaks are too close, or either leave it as is if you want to evaluate the paper as is.")
    ap.add_argument("-S","--save-images", type=str,default="result", 
                    help="Save images showing the differrent stages. Possible values are 'none', 'result','important' and 'all'.")
    ap.add_argument("--image-ext", type=str,default="png", 
                    help="Image extension for saving results.")
    ap.add_argument("-f","--recompute", action="store_true", 
                    help="Force recomputation even if result exists.")
    ap.add_argument("-R","--seed", type=int, default=42,
                    help="Random seed, for reproducibility.")
    ap.add_argument("--parameter-t1", type=float, default=-1,
                    help="Parameter T1 in tne paper. This parameter is set automatically. This allows one to fix it to some arbitrary value for testing purposes..")
    ap.add_argument("--parameter-t2", type=float, default=-1,
                    help="Parameter T2 in tne paper. This parameter is set automatically. This allows one to fix it to some arbitrary value for testing purposes..")
    ap.add_argument("-T","--max-threads", type=int, default=-1,
                    help="IF larger than one, restrict parallel processes to this number. ")
    args = ap.parse_args()


    if args.input_image is None and args.input_list is None:
        print("Must specify either an input image or an input image list!")
        ap.print_help()
        exit(1)
    elif args.input_image is not None and args.input_list is not None:
        print("Cannot specify both an input image and an input image list!")
        ap.print_help()
        exit(1)
    
    logger = logging.getLogger('voronoi.log')
    logger.info('*** Area Voronoi Document Layout Analysis ***')
    logger.info('Parameters:')
    args_dict = vars(args)
    for k in args_dict.keys():
        logger.info(f'\t{k:20}: {args_dict[k]}')

    if args.input_image is not None:
        logger.info('Single input mode. Prefix can be anything.')
        area_voronoi_dla(args.input_image,args)
    else:
        logger.info('Batch mode. Prefix is assumed to be a directory.')
        input_list_file = open(args.input_list,'r')
        input_images = [l.strip() for l in input_list_file]
        input_list_file.close()
        nimages = len(input_images)
        output_base_dir = args.output
        logger.info(f'Number of images to process: {nimages}.')
        if args.max_threads > 0:
            with mp.Pool(args.max_threads) as pool:
                pool.starmap(area_voronoi_dla_mp,[(rel_fname,args) for rel_fname in input_images])
        else:
            with mp.Pool() as pool:
                pool.starmap(area_voronoi_dla_mp,[(rel_fname,args) for rel_fname in input_images])
    
#==============================================================================
# END OF CODE
#==============================================================================
