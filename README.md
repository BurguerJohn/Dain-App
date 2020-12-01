
# DAIN-APP Application
This is the source code for the video interpolation application **Dain-App**, developed on top of the source code of **DAIN**
[Dain GIT Project](https://github.com/baowenbo/DAIN)


### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Installation](#installation)
1. [Running application with interface](#running-application-with-interface)
1. [Running application with command line](#running-application-with-command-line) 
1. [Slow-motion Generation](#slow-motion-generation)
1. [Training New Models](#training-new-models)
1. [Google Colab Demo](#google-colab-demo)

### Introduction
Dain-App comes with a user interface and a command line script to help new users to start using it with little change to the code. You can also get the Windows binary from the build section.

You can see a few results from those Youtube videos.
[Turning animations to 60FPS](https://youtu.be/IK-Q3EcTnTA).
[Turning Sprite Art to 60FPS](https://youtu.be/q2i6FXVjNT0).
[Turning Stop Motion to 60FPS](https://youtu.be/eAUn7Nvx73s).
[Turning ANIME P1 to 60FPS](https://youtu.be/Auum01OEs8k).
[Turning ANIME P2 to 60FPS](https://youtu.be/x67aYuZ-0YI).


### Citation
If you find the code and datasets useful in your research, please cite:
	
	@article{Dain-App,
		title={Dain-App: Application for Video Interpolations},
		author={Gabriel Poetsch},
		year={2020}
	}
	@inproceedings{DAIN,
        author    = {Bao, Wenbo and Lai, Wei-Sheng and Ma, Chao and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan}, 
        title     = {Depth-Aware Video Frame Interpolation}, 
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
        year      = {2019}
    }
    @article{MEMC-Net,
         title={MEMC-Net: Motion Estimation and Motion Compensation Driven Neural Network for Video Interpolation and Enhancement},
         author={Bao, Wenbo and Lai, Wei-Sheng, and Zhang, Xiaoyun and Gao, Zhiyong and Yang, Ming-Hsuan},
         journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
         doi={10.1109/TPAMI.2019.2941941},
         year={2018}
    }	

### Requirements and Dependencies
- numba=0.51.2
- numpy=1.19.2
- opencv-python=4.4.0.46
- pillow=8.0.1
- pyqt5=5.15.1
- python=3.8.5
- scikit-learn=0.23.2
- scipy=1.5.4
- torch=1.7.0+cu110
- torchvision=0.8.1+cu110
- tqdm=4.51.0
- ffmpeg

### Installation
Check out the Colab code:
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/BurguerJohn/Dain-App/blob/master/Dain_App_Colab.ipynb)

Remember you need to build the .cuda scripts before the app can work.

### Running application with interface
    python my_design.py

### Running application with command line
You can see all commands for CLI using this code:

    python my_design.py -cli -h
A example of a working code:

    python  my_design.py -cli --input "gif/example.gif" -o "example_folder/" -on "interpolated.gif" -m "model_weights/best.pth" -fh 3 --interpolations 2 --depth_awarenes 0 --loop 0 -p 0 --alpha 0 --check_scene_change 10 --png_compress 0 --crf 1 --pixel_upscale_downscale_before 1 --pixel_downscale_upscale_after 1 --pixel_upscale_after 1 --mute_ffmpeg 1 --split_size_x -1 --split_size_y -1 --split_pad 150 --half 0 --step_extract 1 --step_interpolate 1 --batch_size 1 --use_benchmark 0 --force_flow 1 --smooth_flow 0 --downscale -1 --fast_mode 0

### Training New Models
Currently Dain-App training code is broken, to train new models, use the DAIN github and import the models to Dain-App

### Google Colab Demo
[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](https://colab.research.google.com/github/BurguerJohn/Dain-App/blob/master/Dain_App_Colab.ipynb)

### Contact
[Gabriel Poetsch](mailto:griskai.yt@gmail.com)

### License
See [MIT License](https://github.com/BurguerJohn/Dain-App/blob/master/LICENSE)