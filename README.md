# Voronoi diagrams for page segmentation

Page segmentation is a key task in document processing, enabling effective extraction of structured information from diverse document types. This paper presents an in-depth analysis of the method proposed by Kise et al., a bottom-up approach using area Voronoi diagrams to identify spatial relationships between document components. Our work provides a detailed description of the method, emphasizing clarity, reproducibility, and transparency, particularly regarding aspects not fully specified in the original paper. We highlight the impact of various implementation choices, such as parameter settings and preprocessing steps, on the method's performance. Through extensive testing on diverse document layouts, we demonstrate that the method can handle a wide range of scenarios but exhibits notable sensitivity to specific parameters and document characteristics, especially in handling complex elements like lists, drop-caps, and tables.

## Usage

### To run with default parameters:
```
python voronoi.py -i input.png 
```

### To run with custom parameters:
```
python voronoi.py -i input.png -B binarization_mobe -Y binarization_threshold -b param_N -r param_rho -w param_w -a param_Ta
```

Arguments:

| Flag           | Value                   | Description                                                                                              |
| :--------------| :-----------------------| :------------------------------------------------------------------------------------------------------- |
| -i             | input.png               | input image                                                                                              |
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
python voronoi.py -h
```
