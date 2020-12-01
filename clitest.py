import argparse
import warnings
import RenderData
import cv2
import os

parser = argparse.ArgumentParser(description='DAIN')
parser.add_argument('--cli', '-c', type = bool, default=False, help = 'Execute application in CLI mode.')
parser.add_argument('--input', '-i', type = str, default="", help = 'Path to the input video')
parser.add_argument('--output', '-o', type = str, default="", help = 'Output Path to generate the folder with all the files.')
parser.add_argument('--output_name', '-on', type = str, default="video.mp4", help = 'Name and extension of the videos that will be rendered. [mp4, webm, gif, apng]')

parser.add_argument('--pallete', '-p', type = int, default=0, help = 'Generate a version of the file limiting the pallete.')
parser.add_argument('--downscale', '-d', type = int, default=-1, help = 'Downscale the input resolution. (-1 Turn off)')
parser.add_argument('--loop', '-l', type = int, default=0, help = 'Turn on if the animation do a perfect loop.')
parser.add_argument('--interpolations', '-in', type = int, default=2, choices=[2,4,8], help = 'How much new frames will be created.')
parser.add_argument('--downsample_fps', '-ds', type = int, default=-1, help = 'Create a version of the movie with a smaller fps. (-1 Turn off)')
parser.add_argument('--frame_handling', '-fh', type = int, default=1, choices=[1,2,3,4], help = 'Interpolation Modes: Mode 1 - Default ; Mode 2 - Default, remove duplicates ; Mode 3 - Adaptative Timestamp ; 4 - Static timestamp')
parser.add_argument('--depth_awarenes', '-da', type = int, default=1, help = 'Should depth be calculated in interpolations?')
parser.add_argument('--split_size_x', '-ssx', type = int, default=-1, help = 'How much division are made in the X axis of each frame. (-1 Turn off ; 1 = No divisions in this axis)')
parser.add_argument('--split_size_y', '-ssy', type = int, default=-1, help = 'How much division are made in the Y axis of each frame. (-1 Turn off ; 1 = No divisions in this axis)')
parser.add_argument('--split_pad', '-sp', type = int, default=150, help = 'Split frames using this values as pixel padding in width and height for each side of the frame.')
parser.add_argument('--alpha', '-a', type = int, default=0, choices=[0, 1], help = 'Calculate transparency in interpolations. 0: Off, 1: Fast')
parser.add_argument('--check_scene_change', '-csc', type = int, default=-1, help = 'Sensitivity for scene change detection, skip interpolation if detect it as true. (-1 Turn off)')
parser.add_argument('--audio_version', '-av', type = int, default=0, help = 'Generate a version with audio.')
parser.add_argument('--interpolation_algo', '-ia', type = int, default=0, choices=[0, 1], help = '0: Default   1: Experimental')
parser.add_argument('--interpolate_missing_original', '-imo', type = int, default=0, choices=[0, 1], help = 'Create interpolation of any missing files in the original_folders')

parser.add_argument('--clear_original_folder', '-cof', type = int, default=1,  help = 'Clean all files in the original frames folder before starting.')
parser.add_argument('--clear_interpolated_folder', '-cif', type = int, default=1,  help = 'Clean all files in the interpolation folder before starting. If turned on, it skip already interpolated frames.')

#parser.add_argument('--input_type', type = int, default=1,  help = 'Type of input: [1 = Video ; 2 = PNG Sequence ; 3 = Resume Render]')

parser.add_argument('--step_extract', '-se', type = int, default=1,  help = 'Do the step of extracting all frames from the original video into the folder.')
parser.add_argument('--step_interpolate', '-si', type = int, default=1,  help = 'Do the step of interpolating all the original frames.')
parser.add_argument('--step_render', '-sr', type = int, default=1,  help = 'Do the step of creating the video from all the interpolated frames.')

parser.add_argument('--model', '-m', type = str, default="./model_weights/best.pth",  help = 'Path of the model to be used.')
parser.add_argument('--half', '-ha', type = int, default=0,  help = 'Use half precision float points.')

parser.add_argument('--clean_cache', '-cc', type = int, default=1,  help = 'Clean the CUDA cache between frames.')
parser.add_argument('--quiet', '-qu', type = int, default=0,  help = "Don't print messages in the console.")

parser.add_argument('--only_originals', '-doo', type = int, default=0,  help = "Debug: Only interpolate missing frames.")


parser.add_argument('--png_compress', '-pngc', type = int, default=6,  help = "")
parser.add_argument('--crf', type = int, default=17,  help = "CRF value for output video.")
parser.add_argument('--pixel_upscale_downscale_before', '-pudb', type = int, default=1,  help = "Multiple that will upscale and then downscale the image. 1 = Disabled")
parser.add_argument('--pixel_downscale_upscale_after', '-pdua', type = int, default=1,  help = "Multiple that will downscale and then upscale the image. 1 = Disabled")
parser.add_argument('--pixel_upscale_after', '-pua', type = int, default=1,  help = "Multiple that will upscale the image. 1 = Disabled")
parser.add_argument('--pixel_bg_color', '-pbgc', type = int, nargs=3, default=[255, 0 , 127],  help = "")
parser.add_argument('--mute_ffmpeg', type = int, default=1,  help = "Dont show ffmpeg messages.")
parser.add_argument('--use_benchmark', type = int, default=1,  help = "Use Cunnd benchmark.")
parser.add_argument('--batch_size', type = int, default=1,  help = "Size of each batch.")
parser.add_argument('--selected_device', type = int, default=0,  help = "Set the device to render the interpolation.")

parser.add_argument('--share_flow', type = int, default=0,  help = "Share the same flow for the inverse movement flow.")
parser.add_argument('--smooth_flow', type = int, default=0,  help = "Smooth the motion flow.")
parser.add_argument('--force_flow', type = int, default=12,  help = "The force of the flow.")

parser.add_argument('--fast_mode', type = int, default=0,  help = "Use fast interpolation mode.")

parser.add_argument('--multiprocessing-fork', type = bool, default=False,  help = "For internal use")
parser.add_argument('--parent_pid', type = int, default=0,  help = "For internal use")
parser.add_argument('--pipe_handle', type = int, default=0,  help = "For internal use")



args = parser.parse_args()



def Execute():
    warnings.filterwarnings("ignore")

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    myRenderData = RenderData.RenderData()

    myRenderData.crf = args.crf
    myRenderData.pngcompress = args.png_compress
    myRenderData.pixelUpscaleDowscaleBefore = args.pixel_upscale_downscale_before
    myRenderData.pixelDownscaleUpscaleAfter = args.pixel_downscale_upscale_after
    myRenderData.pixelUpscaleAfter = args.pixel_upscale_after
    myRenderData.pixelBgColor = tuple(args.pixel_bg_color)
    myRenderData.mute_ffmpeg = args.mute_ffmpeg

    myRenderData.sel_process = args.selected_device

    myRenderData.video = args.input
    myRenderData.outFolder = os.path.normpath(args.output) + "/"
    myRenderData.outStr = os.path.join(args.output, args.output_name)

    myRenderData.uploadBar = None
    myRenderData.useWatermark = 0

    myRenderData.batch_size = args.batch_size

    myRenderData.palette = int(args.pallete)
    myRenderData.loop = int(args.loop)
    
    myRenderData.framerateConf = args.interpolations
    myRenderData.fastMode = args.fast_mode
    
    myRenderData.use60C1 = 0
    myRenderData.use60C2 = 0
    myRenderData.interpolationMethod = args.frame_handling - 1
    myRenderData.exportPng = 0

    myRenderData.flowForce = args.force_flow
    myRenderData.ShareFlow = bool(args.share_flow)
    myRenderData.SmoothFlow = args.smooth_flow

    myRenderData.useBenchmark = args.use_benchmark 

    if args.depth_awarenes:
        myRenderData.useAnimationMethod = 0
    else:
        myRenderData.useAnimationMethod = 1
    
    myRenderData.alphaMethod = args.alpha
    myRenderData.inputMethod = 0

    myRenderData.audioVersion = int(args.audio_version)

    myRenderData.cleanOriginal = int(args.clear_original_folder)
    myRenderData.cleanInterpol = int(args.clear_interpolated_folder)

    myRenderData.doOriginal = int(args.step_extract)
    myRenderData.doIntepolation = int(args.step_interpolate)
    myRenderData.doVideo = int(args.step_render)

    myRenderData.cleanCudaCache = int(args.clean_cache)
    myRenderData.quiet = int(args.quiet)
    myRenderData.model = args.model
    myRenderData.onlyRenderMissing = args.only_originals
    myRenderData.use_half = args.half

    #myRenderData.useMultiProcess = args.use_multiprocess
    #myRenderData.processes = args.n_processes

    myRenderData.version = RenderData.GetVersion()

    myRenderData.interpolationAlgorithm = int(args.interpolation_algo)
    

    if args.downsample_fps == -1:
        myRenderData.use60 = 0
    else:
        myRenderData.use60 = 1
        myRenderData.use60RealFps = args.downsample_fps

    if os.path.exists(args.input):
        cam = cv2.VideoCapture(args.input)
        fps = cam.get(cv2.CAP_PROP_FPS)
        myRenderData.fps = float(fps)
    else:
        myRenderData.fps = 25

    if args.downscale == -1:
        myRenderData.resc = 0
        myRenderData.maxResc = 0
    else:
        myRenderData.resc = 1
        myRenderData.maxResc = args.downscale

    if args.split_size_x == -1:
        myRenderData.splitFrames = 0
        myRenderData.splitSizeX = 0
        myRenderData.splitSizeY = 0
        myRenderData.splitPad = 0
    else:
        myRenderData.splitFrames = 1
        myRenderData.splitSizeX = args.split_size_x
        myRenderData.splitSizeY = args.split_size_y
        myRenderData.splitPad = args.split_pad

    if args.check_scene_change == -1:
        myRenderData.checkSceneChanges = 0
        myRenderData.sceneChangeSensibility = 0
    else:
        myRenderData.checkSceneChanges = 1
        myRenderData.sceneChangeSensibility = args.check_scene_change

    myRenderData.fillMissingOriginal = args.interpolate_missing_original

    return myRenderData

    #dain_class = DainClass()
    #dain_class.RenderVideo(myRenderData)

if __name__ == "__main__":
    Execute()


