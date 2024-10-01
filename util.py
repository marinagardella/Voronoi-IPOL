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

def timing(func):
    """
    Decorator to measure the time of execution of a function.
    """
    def inner1(*args, **kwargs):
        #logger = get_logger()
        begin = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        #logger.info(f" {func.__name__:50}:{end - begin:8.3f}s")
        return ret
    return inner1


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


POINT_COLOR  = (128,128,255)
VERTEX_COLOR = (255,128,128)
RIDGE_COLOR  = (64,128,192)

def plot_voronoi(background,
                 points,
                 vertices,
                 ridge_points,
                 ridge_vertices,
                 ridge_color=None,
                 draw_points=False,
                 point_color=None,
                 point_labels=False,
                 vertex_labels=False,
                 ridge_labels=False):
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


def plot_voronoi_pyplot(img,borders,points,vertices,ridge_vertices,ridge_points=None,deluxe=False,ridge_color=None):
    """
    Plot the Voronoi diagram over the input image with colors, labels, and all sorts of
    fancy stuff.
    """
    aspect_ratio = img.shape[1]/img.shape[0]
    fig = plt.figure(figsize=(10*aspect_ratio,10),dpi=600)
    ax = fig.add_subplot()
    ax.set_axis_off()
    ax.margins(0)
    ax.set_facecolor((0.0,0.0,0.0))
    ax.patch.set_facecolor((0.0,0.0,0.0))

    ax.imshow((3.0/10.0)*img+(3.0/10.0)*borders,cmap='gray',vmax=1.0,vmin=0.0)
    H,W = img.shape
    
    for i,p in enumerate(points):
        p_y,p_x = p
        ax.add_artist(mpatches.Circle((p_x,p_y),1,color=(0.5,0.5,1.0),alpha=1))
        if deluxe:
            ax.text(p_x,p_y,f'$p_{{{i+1:2}}}$',fontproperties=fp,parse_math=True,color=POINT_COLOR,alpha=0.5)

    if ridge_points is not None:
        for i,rp in enumerate(ridge_points):
            p_y,p_x = points[rp]
            ax.add_artist(mpatches.Circle((p_x,p_y),1,color=POINT_COLOR))
            if deluxe:
                ax.text(p_x,p_y,f'$p_{{{i+1:2}}}$',fontproperties=fp,parse_math=True,color=POINT_COLOR,alpha=0.5)

    for i,rvi in enumerate(ridge_vertices):
        vi_1, vi_2 = rvi
        if vi_1 >= 0:
            p_y,p_x = vertices[vi_1]
            ax.add_artist(mpatches.Circle((p_x,p_y),1,color=VERTEX_COLOR,alpha=0.5))
            if deluxe:
                ax.text(p_x,p_y,f'$v_{{{i+1:2}}}$',fontproperties=fp,parse_math=True,color=VERTEX_COLOR,alpha=0.5)
        if vi_2 >= 0:
            p_y,p_x = vertices[vi_2]
            ax.add_artist(mpatches.Circle((p_x,p_y),1,color=VERTEX_COLOR,alpha=0.5))
            if deluxe:
                ax.text(p_x,p_y,f'$v_{{{i+1:2}}}$',fontproperties=fp,parse_math=True,color=VERTEX_COLOR,alpha=0.5)

    if ridge_color is None:
        ridge_color = [RIDGE_COLOR]*len(ridge_vertices)
    elif len(ridge_color) < len(ridge_vertices):
        assert(len(ridge_color) <= 4)
        ridge_color = [ridge_color]*len(ridge_vertices)
    else:
        assert(len(ridge_color) == len(ridge_vertices))
    for i,rv in enumerate(ridge_vertices):
        if rv[0] < 0 or rv[1] < 0:
            continue
        a_i = rv[0]
        b_i = rv[1]
        a_y,a_x = vertices[a_i]
        b_y,b_x = vertices[b_i]
        e_x = (a_x+b_x)/2
        e_y = (a_y+b_y)/2
        ax.plot((a_x,b_x),(a_y,b_y),lw=1,color=ridge_color[i],alpha=0.5)
        if deluxe:
            ax.text(e_x,e_y,f'$e_{{{i+1:2}}}$',fontproperties=fp,parse_math=True,color=ridge_color[i],alpha=0.5)
    ax.set_xlim(0,W)
    ax.set_ylim(H,0)
    return fig
