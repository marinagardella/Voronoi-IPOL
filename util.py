import os
import logging
import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import patches as mpatches
from matplotlib import colors
from PIL import Image, ImageDraw

#===================================================================================================
# Utility functions
#===================================================================================================


def read_img(img_path):
    """
    Read an image from a file, transform to grayscale if eneded and return it as a numpy array with data type float
    """
    with  Image.open(img_path) as IMG:
        return np.array(IMG)


def write_img(img_path, img):
    """
    Utility to write images. We convert them to uint8 before writing.
    """
    if img.dtype == 'bool':
        out_img = (255*img).astype(np.uint8)
    elif img.dtype != np.uint8:
        M = np.max(img)
        if M > 1.0:
            out_img = (255*img/M).astype(np.uint8)
        else:
            out_img = img.astype(np.uint8)
    else:
        out_img = img
    Image.fromarray(out_img).save(img_path)



#===================================================================================================
# PRETTY PLOTTING
#===================================================================================================

color_list = (
    (0.00,(0,0,0,1)),
    (0.09,(0,0,0,1)),
    (0.10,(1,0,0,1)),
    (0.30,(0.5,0.5,0,1)),
    (0.50,(0,0.5,0.5,1)),
    (0.70,(0,0,1,1)),
    (1.00,(0.5,0,0.5,1))
)
vertex_cmap = colors.LinearSegmentedColormap.from_list(name='vertex',colors=color_list)

cidx = np.linspace(0,1,num=7)
color_list = (
    (cidx[0],(0,0,0,1)),
    (cidx[1],(1.0,0.0,0.0,1)),
    (cidx[2],(0.5,0.5,0.0,1)),
    (cidx[3],(0.0,1.0,0.0,1)),
    (cidx[4],(0.0,0.5,0.5,1)),
    (cidx[5],(0.0,0.0,1.0,1)),
    (cidx[6],(0.5,0.0,0.5,1))
)
custom_cmap = colors.LinearSegmentedColormap.from_list(name='vertex',colors=color_list)

fp = FontProperties()
fp.set_family('sans-serif')
fp.set_weight('bold')
fp.set_size(9)


POINT_COLOR  = (255,64,0,128)
VERTEX_COLOR = (255,128,128)
RIDGE_COLOR  = (64,0,192)

def plot_borders(background,borders):
    """
    Plot the borders over the input image with colors, labels, and all sorts of
    fancy stuff.
    """
    if background.dtype != np.uint8:
        background = (255*background).astype(np.uint8)
    im = Image.fromarray(background)
    im = im.convert('RGB')
    draw = ImageDraw.Draw(im)
    for p in borders:
        p_y,p_x = p
        draw.ellipse((p_x-1,p_y-1,p_x+1,p_y+1),fill=POINT_COLOR)
    return np.array(im)



def plot_voronoi(background,
                 points,
                 vertices,
                 ridge_points,
                 ridge_vertices,
                 ridge_color=None,
                 draw_points=False,
                 point_color=None):
    """
    Plot the Voronoi diagram over the input image with colors, labels, and all sorts of
    fancy stuff.
    """
    if background.dtype != np.uint8:
        background = (255*background).astype(np.uint8)
    imvor = Image.fromarray(background)
    imvor = imvor.convert('RGB')
    draw = ImageDraw.Draw(imvor)
    if ridge_color is None:
        ridge_color = [RIDGE_COLOR]*len(ridge_vertices)
    if point_color is None:
        point_color = [POINT_COLOR]*len(ridge_points)
    for i,rv in enumerate(ridge_vertices):
        if rv[0] < 0 or rv[1] < 0:
            continue
        a_i = rv[0]
        b_i = rv[1]
        a_y,a_x = vertices[a_i]
        b_y,b_x = vertices[b_i]
        draw.line((a_x,a_y,b_x,b_y),fill=ridge_color[i])
    if draw_points:
        for i,p in enumerate(ridge_points):
            p_y,p_x = points[p[0]]
            draw.ellipse((p_x-1,p_y-1,p_x+1,p_y+1),fill=POINT_COLOR)
            p_y,p_x = points[p[1]]
            draw.ellipse((p_x-1,p_y-1,p_x+1,p_y+1),fill=POINT_COLOR)
    return np.array(imvor)
