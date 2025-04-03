# Voronoi method for Document Layout Analysis

## Copyright information

Voronoi method for Document Layout Analysis
Copyright (C) 2025 Marina Gardella & Ignacio Ramírez 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


## Introduction

Page segmentation is a key task in document processing, providing relevant
structure  information from diverse document types. This paper presents an 
in-depth analysis and faithful implementation of the method proposed 
by Kise et al. [1], a bottom-up approach using area Voronoi diagrams to 
identify spatial relationships between document components. 
    
Our work provides a detailed description of the method, emphasizing clarity, 
reproducibility, and transparency, particularly regarding aspects not fully 
or clearly specified in the original paper. We highlight the impact of various 
implementation choices, such as parameter settings and preprocessing steps,
on the method's performance. 
    
Through extensive testing on diverse document layouts, we demonstrate that
the method can handle a wide range of scenarios but exhibits notable 
sensitivity to specific parameters and document characteristics, 
especially in handling complex elements like lists, drop-caps, and tables.

[1] K. Kise, A. Sato, y M. Iwata, 
    "Segmentation of Page Images Using the Area Voronoi Diagram»,
    Computer Vision and Image Understanding", 
    vol. 70, n.º 3, pp. 370-382, jun. 1998, doi: 10.1006/cviu.1998.0684.

The implementation is made as faithful as possible to the original 
method as described in the paper. Tweaks and potential improvements are 
optionally enabled by the user via command line switches.

Also, we aim at an efficient and self-contained implementation which 
is easily portable for comparison with other methods.

Authors:
* Marina Gardella <marigardella@gmail.com>
* Ignacio Ramírez <nacho@fing.edu.uy>

## Structure of the code and cross-reference with the paper

The code is organized in two files: `voronoi.py` and `util.py`.
A
n additional folder `data` contains sample input data: `test1.png`,
 `dropcaps.jpg` and `test.list` for testing purposes. 
 
 Finally a `fig` folder contains some auxiliary
images that the code uses for creating some of the figures that
it can produce together with the actual output of the algorithm.

The file `util.py`  does not need much explanation; it contains a few
utilty functions such as logging, reading and writing images, etc.

The core file `voronoi.py`,  is also the main entry point for the program, 
that is, the one that should be run (as described un the Usage Section 
below). The functions in  `voronoiy.py` map one-to-one with the
 algorithms described in the paper, with a few exceptions. Below is a detailed
 list of the functions and their mappings, if any. Notice that we do not
 provide nor describe the list of arguments in each case; this is left to the 
 Python code documentation within the file.

* `get_binary_image()`: binarizes the input image
* `get_connected_components()`: obtains the connected components of the binarized input image
* `get_borders()`: extracts the borders of the connected components found in the  binarized input image
* `sample_border_points()`: obtains a subsample of the borders of the connected components found in the binarized input image
* `get_point_voronoi()`: computes the Point Voronoi Diagram from a set of points
* `eval_redundancy_criterion()`: implements Algorithm 2 in the paper
* `compute_ridge_features()`: implements Algorithm 3 in the paper
* `compute_thresholds()`: implements Algorithm 4 in the paper
* `eval_pruning_criteria()`: implements Algorithm 5 in the paper
* `eval_loop_condition(ridge_vertices)`: implements algorithm 6 in the paper
* `prune_by_loop_condition()`: implements Algorithm 7 in the paper
* `area_voronoi_dla()`: implements Algorithm 1 in the paper
* `area_voronoi_dla_mp()`: multi-process dispatcher for batch inputs.

## Installation

The program is written in Python 3.9 and depends on the following additional packages:

* matplotlib
* numpy
* scipy
* scikit-image

The `requirements.txt` package contains a full specification of the dependencies 
together with matching package versions. The dependencies can be installed via 
`pip` using `pip install -r requirements.txt`.

## Usage

The program can be executed on single or multiple (batch) input images.

### Single image, default parameters:

For single inputs one must provide the path to the image with the switch `-i` or `--input`. The default output(s) will have `voronoi` as a filename prefix and will be stored in the current directory. This can be changed with the `--output` switch:

```
python voronoi.py -i data/test1.png
```

### Single image, custom parameters:

Several options are provided in the command line. See `--help` for a list. For example, the following command applies the method using a thresholding by intensity (`-B threshold`) where pixel values below 128 are considered background (`-Y 128`), removes blobs whose size are smaller than 5 pixels (`-b 8`), sub-samples the borders of the components so that 20% (`-r 0.2`) and uses a window of radius 3 to smooth the distance histogram (`-w 3`), and the output is saved with prefix `test1_` to the local folder:
```
python3 voronoi.py -i data/test1.png -B threshold -Y 128 -b 8 -r 0.2 -w 3 -o test1 data/test1

```

### Multiple input images:

If the input is a list file, each file in the list will be processed in sequence. If multiple processors/threads are available, the images will be distributed among them to speed up the process. In this case it is recommended to specify an output directory for the many output files that will be produced. In the example below (which can readily be run), an `output` folder will be created, and subfolders within it will store the output from each image in the list, using their basename as subfolder name.

```
python3 voronoi.py -L data/test.list -d data -o output
```

### List of main arguments

See `--help` for a full list of parameters. Below we provide the most interesting ones.


| Flag           | Value                   | Description                                                                                              |
| :--------------| :-----------------------| :------------------------------------------------------------------------------------------------------- |
| -i             | input image             | input image                                                                                              |
| -L             | list file               | input image list                                                                                         |
| -o             | output path             | output name prefix for single images, output path for multiple inputs                                    |
| -B             | binarization_mode       | binarization method for non-binary inputs                                                                |
| -Y             | binarization_threshold  | binarization threshold (if needed)                                                                       |
| -b             | param_N                 | denoising parameter (remove all blobs whose size is smaller than the specified value)                    |
| -r             | param_rho               | subsample parameter defining the proportion of border points to keep                                     |
| -w             | param_w                 | window size parameter (defines a window of size 2*w+1 over which the histogram is smoothed by averaging) |
| -a             | param_Ta                | area threshold                                                                                           |

  
### Advanced usage: 

The source code has extra parameters not included in the IPOL demo. 
To see the full list run:
```
python3 voronoi.py -h
```

### Example usage

```
python voronoi.py -i data/test.png
```
