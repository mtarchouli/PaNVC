# PaNVC : Parallel Neural Video coding
PaNVC is a framework for learned video coding, adapted to practical constraints. 
More details about the functionning of this framework are described in the following paper : 
reference to the MHV paper

This project is a collaboration between Ateme and Insa Rennes, as a part of my PhD work.
## Environment  
To set a python environment for this framework, we provide the file env_python.yml, regrouping all the packages necessary for this project. 
To install these packages, you can use the command 
```
conda env create -f env_python.yml
```
Then clone this repository  using : 
```
git clone "lien projet"
```

## Usage 
We provide 8 Encoding and decoding models, corresponding to different lambda values. 4 of them are trained to optimize the MSE metric and the other 4 are trained to optimize MS-SSIM metric.  

The file "config_enc.json", contains information about the video sequence to encode and its configuration.
| Parameter   | Description                                | Constraints                                                                                   |
| :---:       |     :---:                                  | :---:                                                                                         |
| yuv_path    | Path to yuv to encode                      | should be named : Name_WxH_framerate_420.yuv                                                  |
| GOP_type    | Type of encoding All_intra or Inter        | Currently only All_intra is supported                                                         |
| GOP_size    | Number of frames in one GOP                |  1 for All_intra                                                                              |
| N_overlap   | Number of overlapping pixels               |       -                                                                                       |
| patch_w     | Width of the patch                         |       -                                                                                       |
| patch_h     | Height of the patch                        |       -                                                                                       |
| Lambda      | Quality level of the encoder and decoder   | lambda = {420,220,120,64} for MS-SSIM models, lambda = {4096, 3140,2048, 1024} for MSE models |     
| Start_frame | the index of the first frame to encode     |       -                                                                                       |
| nbr_frame   | Number of frames to encode                 |       -                                                                                       |

The file "config_dec.json", contains fixed parameters for encoding and decoding.  
|Parameter          | Description                                    | Constraints                                         |
| :---:             | :---:                                          | :---:                                               |
|N_parallel         | Number of patches to be processed in parallel  | To be adapted according to the available hardware   |     
|model_path_enc     | Path to the encoding model                     | The provided models are in the folder named encoders|
|model_path_dec     | Path to decoding model                         | The provided models are in the folder named decoders|
|flag_md5sum        | To use the md5sum option or not                |                         -                           |
|device             | Which device the coding will run on            | It can be 'cpu' or 'cuda:0'.                        |

To launch encoding , the following command is used : 

```
python encode.py 
```
AFter encoding,  a folder called "bin" is generated. It contains the bitstream of the encoded sequence. 
To launch decoding, we use :
```
python decode.py 
```
The output of the decoder is a reconstructed sequence called "rec420.yuv".

## TO DO  
 - Inprove the code for generating bistream
 - Generalize to other all intra coding models
 - Generalize to other Inter coding models

## References : 
https://github.com/Orange-OpenSource/AIVC
https://github.com/jorge-pessoa/pytorch-gdn
https://github.com/jorge-pessoa/pytorch-msssim

## Licence :
