# Demultipath
![results](./assets/results2.jpg)



![results](./assets/results1.jpg)
## Abstract

Recent research into 3D reconstruction of underwater objects via Neural Radiance Fields (NeRF) has achieved notable progress.
Nevertheless, the NeRF approach overlooks the multi-path effects present in the acquired sonar images, resulting in a reconstruction
of “holes” and “mirages” that do not exist. This study proposes a multi-view consistency metric to eliminate potential multi-path
echoes in images. This metric fuses two transmittance estimations of echoes, which are separately obtained from two multi-view
consensuses. The first consensus is the preliminary NeRF reconstruction and it alone overestimates the transmittance. While,
the second consensus, based on a point cloud of potential occupancy points, underestimates the transmittance. Finally, 3D reconstruction is achieved via the NeRF method based on sonar images free of multi-path echoes. Through simulated experiments,
the proposed method has been shown to successfully eliminate multi-path effects in the sonar images and reconstruct accurate 3D
models without “holes” or “mirages”

## Data and SCUNet
Some sample datasets are available for download here [datasets](https://drive.google.com/drive/folders/1YxgR2I4HUcQujKw1IFR7Te8R4z5kLKbA?usp=drive_link).
You should put the [SCUNet](https://drive.google.com/drive/folders/1EsDpwl9CpIDFMqgzajKi_wwrz3Fi8Zry?usp=drive_link) into `denoise` file. 


## requirements
- Dependencies stored in `requirements.txt`.
- CUDA
- Python 3.10

## Running
- The `run_neusis.py` is for running the Neusis method
- The `pco_reconstruction.py` is for performing PCO reconstruction
- The `run_denoise.py` is for denoising based on a fused metric.
  
After obtaining two metrics by running `run_neusis.py` and `pco_reconstruction.py`, running `run_denoise.py` can remove multi-path effects echoes from sonar images.

## Acknowledgement
Some code snippets are borrowed from neusis and SCUNet. Thanks for these projects!
