import time
import os
from torch.autograd import Variable
import math
import torch
import random
import numpy as np
import numpy
import networks
import cv2
import shutil
import PIL
import PIL.Image
from tqdm import tqdm
import time
import psnr
#from torch.cuda.amp import autocast
import setting
from DainDataset import DainDataset
#from torch.multiprocessing import Pool, set_start_method, Queue, Process, cpu_count
from shutil import copyfile
from RenderData import FrameCollection, FrameData
import RenderData
from tqdm import tqdm

DEBUG_TEST_NO_CUDA = False
SHOW_TIMING = False

USE_CUDA = True
TIMER = None
 
depad_value = []

import json
from json import JSONEncoder


def NumpyResultAsList(data, pad):
  outputList = []

  size = data[0].shape[0]

  for x in range(0, size):
    for i in range(0, len(data)):
      if pad == None:
        yy_ = torch.round(255.0 * torch.clamp(data[i][x].float(), 0.0, 1.0)[:, :, :].permute(1, 2, 0)).detach().cpu().numpy().astype(numpy.uint8)
      else:
        yy_ = torch.round(255.0 * torch.clamp(data[i][x].float(), 0.0, 1.0)[:, pad[2]:pad[3], pad[0]:pad[1]].permute(1, 2, 0)).detach().cpu().numpy().astype(numpy.uint8)
      outputList.append(yy_)

  return outputList


def lerp(a, b, f):
  return int(a + f * (b - a))

def SetTiming(tag):
  global TIMER, SHOW_TIMING
  if TIMER == None:
    TIMER = time.time()
    if SHOW_TIMING:
      print("\nStarting Timer: " + tag) 
  if SHOW_TIMING:
    print("\n"+tag + ": --- %s seconds ---" % (time.time() - TIMER))
    TIMER = time.time()

def CalculatePad(intWidth, intHeight):
  if intWidth != ((intWidth >> 7) << 7):
    intWidth_pad = (((intWidth >> 7) + 1) << 7)
    intPaddingLeft =int(( intWidth_pad - intWidth)/2)
    intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
  else:
    intWidth_pad = intWidth
    intPaddingLeft = 32
    intPaddingRight= 32

  if intHeight != ((intHeight >> 7) << 7):
    intHeight_pad = (((intHeight >> 7) + 1) << 7) 
    intPaddingTop = int((intHeight_pad - intHeight) / 2)
    intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
  else:
    intHeight_pad = intHeight
    intPaddingTop = 32
    intPaddingBottom = 32

  return (intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom,0,0)




def Configure(owner, myRenderData):
  global DEBUG_TEST_NO_CUDA

  useAnimationMethod = myRenderData.useAnimationMethod

  (iWidth, iHeight) = owner.GetInputSize()
  padding_value = CalculatePad(iWidth, iHeight)
  setting.SetPad(padding_value)

  global depad_value
  depad_value = setting.GetPad()
  depad_value = (padding_value[0], padding_value[0] +iWidth, padding_value[2], padding_value[2]  +iHeight)

  print("Using Model: " + myRenderData.model)

  if DEBUG_TEST_NO_CUDA:
    return

  torch.cuda.empty_cache()


  if myRenderData.interpolationAlgorithm == 0:
    gl_model = networks.__dict__["DAIN"](channel=3,
                                filter_size = 4 ,
                                timestep= 0.5,
                                training=False,
                                useAnimationMethod = useAnimationMethod,
                                shareMemory = False,
                                cuda = USE_CUDA, iSize = [iWidth, iHeight],
                                batch_size= myRenderData.batch_size,
                                padding = padding_value, depadding = depad_value,
                                is_half = bool(myRenderData.use_half))

    #gl_model = torch.jit.script(gl_model)

  else:
    #Broken
    gl_model = networks.__dict__["DAIN_slowmotion"](channel=3,
                                filter_size = 4,
                                timestep= 0.5,
                                training=False,
                                useAnimationMethod = useAnimationMethod)

  
  if USE_CUDA:
    gl_model = gl_model.cuda()


  SAVED_MODEL = myRenderData.model

  
  
  if os.path.exists(SAVED_MODEL):
    #self.LogPrint("The testing model weight is: " + SAVED_MODEL)
    if not USE_CUDA:
      pretrained_dict = torch.load(SAVED_MODEL, map_location=lambda storage, loc: storage)
      # model.load_state_dict(torch.load(SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
      pretrained_dict = torch.load(SAVED_MODEL)
      # model.load_state_dict(torch.load(SAVED_MODEL))

    model_dict = gl_model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    gl_model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
  else:
    raise Exception("Model file not found. Try to start the app on Admin-mode, if the error persists try to download again the last version of the app.")
    #self.LogPrint("*****************************************************************")
    #self.LogPrint("**** We don't load any trained weights **************************")
    #self.LogPrint("*****************************************************************")


  #import PWCNet
  gl_model = gl_model.eval() # deploy mode

  return gl_model



def interpolate_(gl_model, myRenderData, X0, X1, convert):
  use_cuda=True
  save_which=1
  dtype = torch.cuda.FloatTensor

  if myRenderData.cleanCudaCache == 1:
    SetTiming("Start Cache Clean")
    torch.cuda.empty_cache()
    SetTiming("End Cache Clean")


  if DEBUG_TEST_NO_CUDA:
    return X0
    X0 = X0.data.cpu().numpy()
    X0 = [np.transpose(255.0 * x.clip(0,1.0), (1, 2, 0)) for x in X0]
    X0 = [np.round(x).astype(numpy.uint8) for x in X0]

    imgList = []
    for item in X0:
      imgList.append(np.round(item).astype(numpy.uint8))
    return imgList

  padding = [0,0 ,0,0]

  SetTiming("Calling Model")
  
  with torch.no_grad():
    y_s,offset,filter = gl_model(X0, X1, padding, myRenderData.flowForce, myRenderData.SmoothFlow, myRenderData.ShareFlow, convert, not bool(myRenderData.fastMode))

  torch.cuda.synchronize()
  SetTiming("Ending Model Call")
  y_ = y_s[save_which]
  
  #It used to run everything under this, not anymore
  if True:
    return y_

  imgList = []
  if not isinstance(y_, list):
    yy_ = [torch.round(255.0 * torch.clamp(y_, 0.0, 1.0)[:, :, :].permute(1, 2, 0)) for yy_ in y_]
    SetTiming("End Changing Images")
    return yy_

  else:
    return imgList

def SavePNG(png_compress, pil, filepath):
  pil.save(filepath, optimize = False, compress_level = int(png_compress))


class DainClass():
  panic = ""


  myRenderData = None
  # 2 / 4 / 8

  mainFolder =  ""
  originalFrames = ""
  interpolatedFrames = ""
  renderFolder = ""
  configFile = ""
  lastFrameMs = 0

  save_which = None


  useMask = False
  mask = []

  model = None

  def GetInterCounter(self):
    
    filename = self.mainFolder + "/resume.txt"
    if not os.path.exists(filename):
      return 0
    myfile = open(filename, 'r')
    data = myfile.read()
    myfile.close()
    obj = json.loads(data)

    obj["index"] =  int(obj["index"])
    obj["counter"] =  int(obj["counter"])
    return obj


    
  def SetInterpolCounter(self, index, counter):

    obj = {"index": index , "counter": counter}

    js = json.dumps(obj, cls=json.JSONEncoder)
    filename = self.mainFolder + "/resume.txt"
    myfile = open(filename, 'w')
    myfile.write(js)
    myfile.close()


    

  def GetInputSize(self):
    for image in sorted(os.listdir(self.originalFrames)):
      if image.lower().endswith(".png"):
        return PIL.Image.open( os.path.join(self.originalFrames, image)).size

  def PilResizeFFMPEG(self, pil, size = 1):
    width, height = pil.size
    w = round(width * size)
    h = round(height * size)
    return "scale={}:{}:flags=neighbor".format(w, h)

  def PilResize(self, pil, size = 1):
    width, height = pil.size
    w = round(width * size)
    h = round(height * size)
    return pil.resize((w, h), resample = PIL.Image.NEAREST)


  def extract_frames(self, video, outDir):

    if self.myRenderData.inputType == 2:
      index = 0
      src_files = self._make_video_dataset(video)
      for file_name in src_files:
        if file_name.endswith("png"):
          name = str(index).zfill(10) + ".png"
          img = PIL.Image.open(file_name)
          if self.myRenderData.resc == 1 and img.size[1] > self.myRenderData.maxResc:
            img.thumbnail((99999, self.myRenderData.maxResc) , PIL.Image.ANTIALIAS)
          SavePNG(self.myRenderData.pngcompress, img, outDir + "/" + name)
          index += 1

      return


    myRenderData = self.myRenderData
    vid = cv2.VideoCapture(video)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    vfArr = []

    error = ""

    if myRenderData.interpolationMethod != 0 and myRenderData.debugKeepDuplicates == 0:
      vfArr.append("mpdecimate")

    vfArr.append("scale=-1:"+str(myRenderData.maxResc))

    if myRenderData.outStr.endswith(".mp4"):
      vfArr.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")


    vrStr = ",".join(vfArr)

    if vrStr != "":
      vrStr = '-vf "{}"'.format(vrStr)

    
    if myRenderData.interpolationMethod == 0:
      retn = os.system('{} -i "{}" -vsync 0 {} {} -qscale:v 1  "{}%10d.png"'.format(os.path.join("", "ffmpeg"), video, self.panic, vrStr,  outDir))
    if myRenderData.interpolationMethod == 1:
      retn = os.system('{} -i "{}" -vsync 0 {} {} -qscale:v 1  "{}%10d.png"'.format(os.path.join("", "ffmpeg"), video, self.panic, vrStr, outDir))
    if myRenderData.interpolationMethod == 2 or myRenderData.interpolationMethod == 3:
      retn = os.system('{} -vsync 0 -i "{}" -copyts -r 1000  {} -frame_pts true {} -qscale:v 1 "{}%10d.png"'.format(os.path.join("", "ffmpeg"), video, self.panic, vrStr, outDir))

    if self.myRenderData.palette == 1:
      os.system('ffmpeg -y -i "{}"  {}  -filter_complex palettegen=reserve_transparent=1 "{}"'.format(video, self.panic, os.path.join(self.mainFolder, "palette.png") ))

    
    list = sorted(os.listdir(outDir))
    first = list[0]
    second = list[1]    
    last = list[len(list)-1]

    if myRenderData.interpolationMethod == 2 or myRenderData.interpolationMethod == 3:
      if video.endswith("gif"):
        self.lastFrameMs = self.filename2timestamp(last) + self.LastFrameTiming(video)
      else:
        self.lastFrameMs = self.filename2timestamp(last) + self.filename2timestamp(second)
    else:
      self.lastFrameMs = int(self.filename2timestamp(last)) + 1


    if int(self.myRenderData.loop) == 1:
      formated = str(self.lastFrameMs).zfill(10) + ".png"

      copyfile(os.path.join(outDir, sorted(os.listdir(outDir))[0]), os.path.join(outDir, formated))
      if myRenderData.interpolationMethod == 2 or myRenderData.interpolationMethod == 3:
        if video.endswith("gif"):
          self.lastFrameMs = self.filename2timestamp(last) + self.LastFrameTiming(video) + self.filename2timestamp(second)
        else:
          self.lastFrameMs = self.filename2timestamp(last) + self.filename2timestamp(second) + self.filename2timestamp(second)
      else:
        self.lastFrameMs = int(self.filename2timestamp(last)) + 2

    fList = sorted(os.listdir(outDir))

    frames = FrameCollection()
    for i in range(0, len(fList)):
      f = FrameData()
      f.frameName = fList[i]
      f.tsStart = self.filename2timestamp(fList[i])
      if i != len(fList) -1:
        f.tsEnd = self.filename2timestamp(fList[i+1])
      else:
        f.tsEnd = self.lastFrameMs
      frames.AddFrame(f)
    
    frames.ToJsonFile(self.framesFile)


  def LastFrameTiming(self, file):
    cam = cv2.VideoCapture(file)
    frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cam.get(cv2.CAP_PROP_FPS)

    p = PIL.Image.open(file)
    p.seek(int(frames)-1)
    return int(p.info['duration'])
  
  def export_png(self, dir):
    path = os.path.dirname(self.myRenderData.outStr) + "/png_frames"

    if not os.path.isdir(path):
      os.makedirs(path)

    src_files = os.listdir(dir)
    for file_name in src_files:
      full_file_name = os.path.join(dir, file_name)
      if os.path.isfile(full_file_name):
        if self.myRenderData.palette == 1:
          os.system('ffmpeg -y -i "{}" -i "{}" {} -filter_complex "[0:v][1:v] paletteuse" "{}"'.format(full_file_name, os.path.join(self.mainFolder, "palette.png"), self.panic, path + "/" + file_name))
        else:
          shutil.copy(full_file_name, path)
  
  def RenameSeqFolder(self, src):
    ext = ".png"
    files = self._make_video_dataset(src)
    for i,filename in enumerate(files):
      if filename.endswith(ext):
        strOut = src + str(i+1).zfill(15)+ext
        if filename != strOut:
          self.LogPrint("Renaming " + filename +" to " + strOut)
          os.rename(filename, strOut)

  def GetLastTimingFromGif(path):
    cam = cv2.VideoCapture(path)
    fps = cam.get(cv2.CAP_PROP_FRAME_COUNT)

    p = PIL.Image.open(path)
    p.seek(int(fps)-1)
    return int(p.info['duration'])

  def filesToPil(self, list, original):
    pal = PIL.Image.open(original).convert("P")
    pal.load()
    pilList = []
    for i in range(0, len(list)):
      f = PIL.Image.open(list[i]).convert("P", None, 0, pal.im)
      pilList.append(f)
    return pilList

  def filename2timestamp(self, filename):
    name = os.path.basename(filename)
    name_no_ext = name.split(".")[0]
    return int(name_no_ext)

  def GeneratePILGif(self, outname, fps):
    self.gifFolder = os.path.join(self.mainFolder, "gif_frames/")

    pallFolder = os.path.join(self.mainFolder, "palette.png")

    if not os.path.isfile(pallFolder):
      raise Exception("pallet.png not found. Did you run step 1 with 'Limit Colors' on?")
    
    if not os.path.isdir(self.gifFolder):
      os.makedirs(self.gifFolder)

    files = self._make_video_dataset(self.interpolatedFrames)

    pilImages = []
    pilDurations = []

    std =  int(math.ceil(1 / fps * 1000/10)*10)

    for i in range(0, len(files)):
      filename = os.path.join(self.gifFolder, str(i).zfill(15) + ".gif")
      os.system('ffmpeg -y -i "{}" -i  "{}" {}  -lavfi paletteuse=dither=0  "{}"'.format(files[i], pallFolder, self.panic,  filename))

      if(i != len(files) - 1):
        ts = self.filename2timestamp(files[i+1])
      else:
        ts = self.filename2timestamp(files[i]) + std

      if i == 0:
        lastTs = 0
      else:
        lastTs = self.filename2timestamp(files[i])
      
      img = PIL.Image.open(filename)

      if self.myRenderData.pixelUpscaleDowscaleBefore != 1:
        img = self.PilResize(img, 1 / self.myRenderData.pixelUpscaleDowscaleBefore)

      if self.myRenderData.pixelDownscaleUpscaleAfter != 1:
        img = self.PilResize(img, 1 / self.myRenderData.pixelDownscaleUpscaleAfter)
        img = self.PilResize(img, self.myRenderData.pixelDownscaleUpscaleAfter)

      if self.myRenderData.pixelUpscaleAfter != 1:
        img = self.PilResize(img, self.myRenderData.pixelUpscaleAfter)

      pilImages.append( img)

      if self.myRenderData.interpolationMethod == 2 or self.myRenderData.interpolationMethod == 3:
        pilDurations.append( math.ceil(ts/10)*10-math.ceil(lastTs/10)*10 )
      else:
        pilDurations.append(std)

    pilImages[0].save(
      outname,
      save_all = True,
      append_images = pilImages[1:],
      duration = pilDurations,
      loop = 0,
      optimize=False,
      disposal=2
    )

  def create_video2(self, dir, outSub, speed):
    myRenderData = self.myRenderData

    interpolationMethod = myRenderData.interpolationMethod

    error = ""
    scaleF = ""
    palArg = ''
    
    if interpolationMethod == 0 or interpolationMethod == 1:
      self.RenameSeqFolder(dir)
    
    name = os.path.basename(self.myRenderData.outStr)
    path = os.path.dirname(self.myRenderData.outStr)

    name_base_no_ext = name.split(".")[0]

    originalF = self._make_video_dataset(self.originalFrames)
    interpolF = self._make_video_dataset(self.interpolatedFrames)

    imgOri = PIL.Image.open(originalF[0])
    imgInt = PIL.Image.open(interpolF[0])



    if myRenderData.onlyRenderMissing == 1:
      fps = self.myRenderData.fps
      fpsStr = str(int(fps))
    elif interpolationMethod == 2 or interpolationMethod == 3:
      fps = self.myRenderData.fps * speed
      fpsStr = str(int(fps))
    else:
      originalSize = max(1, len(originalF))
      self.LogPrint("Original FPS: " + str(self.myRenderData.fps))
      self.LogPrint("Total frames Original: " + str(len(originalF)))
      self.LogPrint("Total frames interpolated: " + str(len(interpolF)))
      self.LogPrint("Final FPS: " + str(self.myRenderData.fps * (len(interpolF) / originalSize)))
      fps = self.myRenderData.fps * (len(interpolF) / originalSize)
      fpsStr = str(int(fps))

    vf = []
    
    if self.myRenderData.pixelUpscaleDowscaleBefore != 1:
      vf.append(self.PilResizeFFMPEG(imgInt, 1 / self.myRenderData.pixelUpscaleDowscaleBefore))

    if self.myRenderData.pixelDownscaleUpscaleAfter != 1:
      vf.append(self.PilResizeFFMPEG(imgOri, 1 / self.myRenderData.pixelDownscaleUpscaleAfter))
      vf.append(self.PilResizeFFMPEG(imgOri, 1))

    if self.myRenderData.pixelUpscaleAfter != 1:
      vf.append(self.PilResizeFFMPEG(imgOri, self.myRenderData.pixelUpscaleAfter))

    vfStr = ",".join(vf)

    vf.append("paletteuse=dither=0")
    vfPixelStr = ",".join(vf)
    if vfStr != "":
      vfStr = '-lavfi "{}"'.format(vfStr)
    vfPixelStr = '-lavfi "{}"'.format(vfPixelStr)

    outFile =  outSub+ fpsStr + "fps_" + name
    outFile60 = outSub +str(myRenderData.use60RealFps)+ "fps_" + name
    outFileSmooth60 = outSub +str(myRenderData.use60RealFps)+ "fps_Smooth_" + name
    outFileSharp60 = outSub +str(myRenderData.use60RealFps)+ "fps_Sharp_" + name
    outFilePallet = outSub + fpsStr + "fps_limitPallete_" + name
    outFilePalletPIL = outSub + fpsStr + "fps_limitPallete_PIL_" + name_base_no_ext + ".gif"

    outFileAudio =  outSub+ fpsStr + "fps_audio_" + name
    outFile60Audio = outSub +str(myRenderData.use60RealFps)+ "fps_audio_" + name
    outFileSmooth60Audio = outSub +str(myRenderData.use60RealFps)+ "fps_audio_Smooth_" + name
    outFileSharp60Audio = outSub +str(myRenderData.use60RealFps)+ "fps_audio_Sharp_" + name


    self.LogPrint("Interpolation Folder: " + dir)
    self.LogPrint("Output Video: " + outFile)

    muxer = ""
    pixFmt = "-pix_fmt yuv420p"
    if outFile.endswith(".mp4"):
      muxer = "-c:v libx264 -crf " + str(self.myRenderData.crf)

    if outFile.endswith(".gif"):
      pixFmt = ""
    if outFile.endswith(".apng"):
      pixFmt = "-pix_fmt rgb32"
      

    if interpolationMethod == 0 or interpolationMethod == 1:
      retn = os.system('ffmpeg -y -framerate {fps} -i "{input}%15d.png" {panic}  -qscale:v 0 {muxer} {fmt} {vf} "{out}"'.format(fps=fps, input=dir, panic=self.panic, muxer=muxer, fmt=pixFmt, vf=vfStr, out=outFile))

      if self.myRenderData.palette == 1:
        palArg = '-framerate "{fps}" -i "{input}" {vf}'.format(fps=fps, input=os.path.join(self.mainFolder, "palette.png"), vf=vfPixelStr)
        f = 'ffmpeg -y -framerate "{fps}" -i "{input}%15d.png"  {pallet} {panic} -qscale:v 0 {muxer} {fmt} "{out}"'.format(fps=fps, input=dir, pallet=palArg, panic=self.panic, muxer=muxer, fmt=pixFmt, out=outFilePallet)
        os.system(f)
        self.GeneratePILGif(outFilePalletPIL, fps)
    
    else:
      files = []
      files = self._make_video_dataset(dir)
      strFile = ""
      pilDurations = []

      for i in range(0, len(files)-1):
        name = os.path.basename(files[i])
        name_no_ext = name.split(".")[0]
        duration = float(name_no_ext) / 1000 

        name2 = os.path.basename(files[i+1])
        name_no_ext2 = name2.split(".")[0]
        duration2 = float(name_no_ext2) / 1000 

        pilDurations.append(float(name_no_ext2) - float(name_no_ext))

        dur = (duration2-duration)
        strFile += "file '"+ name+"'\nduration "+ "{:.5f}".format(dur) + "\n"

      framesCol = FrameCollection()
      framesCol.FromJsonFile(self.framesFile)

      if len(framesCol.frames) != 0:
        file = files[len(files)-1]
        name = os.path.basename(file)
        lastF = framesCol.frames[len(framesCol.frames) - 1]

      file = files[len(files)-1]
      name = os.path.basename(file)
      strFile += "file '"+name+"'\nduration "+ "{:.5f}".format(dur) + "\n"

      dir_path = os.path.dirname(os.path.realpath(__file__))
      filename = dir + "/my_file.txt"
      myfile = open(filename, 'w')
      myfile.write(strFile)
      myfile.close()


      txtPath= os.path.join(dir, "my_file.txt")
      palletPath = os.path.join(self.mainFolder, "palette.png")
      
      retn = os.system('ffmpeg -y -vsync 1 -f concat -i "{txtPath}" {panic} {muxer} {fmt} -qscale:v 1 {vf} -r {fps} "{out}"'.format(txtPath=txtPath, fps=fps, speed=speed, panic=self.panic, muxer=muxer, fmt=pixFmt, vf=vfStr, out=outFile))
      if self.myRenderData.palette == 1:
        retn = os.system('ffmpeg -y -vsync 0 -f concat -i "{txtPath}"  -i "{palletPath}"  {panic} {muxer} {fmt} -qscale:v 1 {vf} -r {fps} "{out}"'.format(txtPath=txtPath, palletPath=palletPath, fps=fps, panic=self.panic, muxer=muxer, fmt=pixFmt, vf=vfPixelStr, out=outFilePallet))
        self.GeneratePILGif(outFilePalletPIL, fps)
        

    if myRenderData.audioVersion == 1:
      self.LogPrint("Creating audio version")
      self.AddAudio(outFile, myRenderData.video, outFileAudio)

    if fps > myRenderData.use60RealFps:
      if myRenderData.use60:
        self.SetTo60(outFile, outFile60, myRenderData.use60RealFps)
        if myRenderData.audioVersion == 1:
          self.AddAudio(outFile60, myRenderData.video, outFile60Audio)

      if myRenderData.use60C1:
        self.SetTo60Smooth(outFile, outFileSmooth60, myRenderData.use60RealFps)
        if myRenderData.audioVersion == 1:
          self.AddAudio(outFileSmooth60, myRenderData.video, outFileSmooth60Audio)

      if myRenderData.use60C2:
        self.SetTo60Shart(outFile, outFileSharp60, myRenderData.use60RealFps)
        if myRenderData.audioVersion == 1:
          self.AddAudio(outFileSharp60, myRenderData.video, outFileSharp60Audio)
    
    if retn:
      error = "Error creating output video. Exiting."
    return error
    
  def SetTo60(self, inFile, outFile, fps):
    os.system('ffmpeg -y -i "{}" {} -filter:v fps=fps={} -crf {} -qscale:v 0 "{}"'.format(inFile, self.panic, fps, str(self.myRenderData.crf), outFile))
  def SetTo60Smooth(self, inFile, outFile, fps):
    os.system('ffmpeg -y -i "{}" {} -vf "minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" -crf {} -qscale:v 0 "{}"'.format(inFile, self.panic, fps, str(self.myRenderData.crf), outFile))
  def SetTo60Shart(self, inFile, outFile, fps):
    os.system('ffmpeg -y -i "{}" {} -vf "minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=0" -crf {} -qscale:v 0 "{}"'.format(inFile, self.panic, fps, str(self.myRenderData.crf), outFile))
  
  def AddAudio(self, inFile, audioSource, outFile):
    v = cv2.VideoCapture(audioSource)
    fps = max(1, v.get(cv2.CAP_PROP_FPS))      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
    vDuration = frame_count/fps

    a = cv2.VideoCapture(inFile)
    fps = max(1, a.get(cv2.CAP_PROP_FPS))      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(a.get(cv2.CAP_PROP_FRAME_COUNT))
    aDuration = frame_count/fps

    if aDuration == 0:
      atempo = 0
    else:
      atempo = vDuration / aDuration
    os.system('ffmpeg -y -i "{}" -i "{}" {} -filter_complex "[0:v:0]setpts=PTS[v];[1:a:0]atempo={}[a]" -crf {} -qscale:v 0  -map "[v]" -map "[a]" "{}"'.format(inFile, audioSource, self.panic, atempo, str(self.myRenderData.crf), outFile))


  def _make_video_dataset(self, dir):
    framesPath = []
    # Find and loop over all the frames in root `dir`.
    for image in sorted(os.listdir(dir)):
      # Add path to list.
      if image.lower().endswith(".png"):
        framesPath.append(os.path.join(dir, image))
    return framesPath

  def FitFrame(self, frame):
    return (frame.clamp(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0).astype(numpy.uint8)

  def easySin(self, t, repeats):
    inc = int(t * repeats) 
    incDiv = inc / repeats
    t2 = t * repeats
    cos = 0.5 * (1 - math.cos((t2 - int(t2)) * math.pi))
    return cos / repeats + incDiv


  def Splitter(self, img, img2, ww, hh, padding):
    areas1 = []
    areas2 = []
    areaPoints = []

    debug = False

    indexW = 3
    indexH = 2

    wSource = img.shape[indexW]
    hSource = img.shape[indexH]
    width  = math.ceil(wSource / ww)
    height = math.ceil(hSource / hh)

    if not self.didSplitterPrinted:
      print("\n--Splits:")
      print("Using X_Param:{x} and Y_Param:{y}".format(x=ww, y=hh))

    splitCounter = 1
    for x in range(0, img.shape[indexW], width):
      for y in range(0, img.shape[indexH], height):
        left = max(0, x - padding)
        right = min(img.shape[indexW], x + width + padding)
        top = max(0, y - padding)
        down = min(img.shape[indexH], y + height + padding)

        if right - left % 2 == 1 and right != 0:
          left -= 1

        xIni = max(0, x)
        xEnd = min(img.shape[indexW], x + width)
        yIni = max(0, y)
        yEnd = min(img.shape[indexH], y + height)

        if not self.didSplitterPrinted:
          print("Split {c}: No-Padded frame: From_X:{xIni:4} To_X:{xEnd:4} From_Y:{yIni:4} To_Y: {yEnd:4}".format(
            xIni=xIni, xEnd=xEnd, yIni=yIni, yEnd=yEnd, c=splitCounter))
          print("Split {c}: Padded frame   : From_X:{xIni:4} To_X:{xEnd:4} From_Y:{yIni:4} To_Y: {yEnd:4}".format(
            xIni=left, xEnd=right, yIni=top, yEnd=down, c=splitCounter))


        if(debug):
          xEnd -= 1
          yEnd -= 1

        points = []
        points.append(xIni)
        points.append(xEnd)
        points.append(yIni)
        points.append(yEnd)

        points.append(xIni - left)
        points.append(xEnd - x)
        points.append(yIni - top)
        points.append(yEnd - y)

        areaPoints.append(points)
        areas1.append(img[:, :, top:down, left:right])
        areas2.append(img2[:, :, top:down, left:right])
        splitCounter += 1

    self.didSplitterPrinted = True


    
    
    areaInter = []
    for i in range(0, len(areas1)):
      loopW = areas1[i].shape[indexW]
      loopH = areas1[i].shape[indexH]

      pad = CalculatePad(loopW, loopH)

      areas1[i] = torch.nn.functional.pad(areas1[i].float(), (pad[0], pad[1] , pad[2], pad[3]), mode='replicate', value=0)
      areas2[i] = torch.nn.functional.pad(areas2[i].float(), (pad[0], pad[1] , pad[2], pad[3]), mode='replicate', value=0)

      depad = (pad[0], pad[1] +loopW, pad[2], pad[3]  +loopH)
      self.model.depadding = depad

      newArea = interpolate_(self.model, self.myRenderData, areas1[i], areas2[i], True)
      areaInter.append(newArea.cpu())

    frames = len(areaInter[0])


    sewed = torch.empty((frames, 3, img.shape[indexH], img.shape[indexW]))
    for f in range(0, frames):
      for i in range(0, len(areaInter)):
        p = areaPoints[i]
        sewed[f, :, p[2]:p[3], p[0]:p[1]] = areaInter[i][f][:, p[6]:p[7] + p[6], p[4]:p[5] + p[4]]

    return sewed
    

  def DoInterpolation(self, x0, x1, loops):
    myRenderData = self.myRenderData

    if myRenderData.splitFrames == 0:
      image = interpolate_(self.model, self.myRenderData, x0, x1, False)
    else:
      image = self.Splitter(x0, x1, myRenderData.splitSizeX, myRenderData.splitSizeY, myRenderData.splitPad)

    if myRenderData.interpolationAlgorithm == 1:
      if isinstance(image, list):
        return image

    if loops - 1 > 0:
      imB = self.DoInterpolation(x0, image, loops-1)
      imA = self.DoInterpolation(image, x1, loops-1)
      return imB + [image] + imA
    return [image]

  def TakeMedium(self, f1, f2):
    currFilename = os.path.basename(f1)
    prevFilename = os.path.basename(f2)

    fC = int(currFilename.split(".")[0])
    fP = int(prevFilename.split(".")[0])
    return fC
    return (fC + fP) / 2

  def dirtHack(self, i, val):
    temp = i
    while temp < val:
      temp = (temp * 2) + 1
      i += 1
    return i - 1

  def FillMissingOriginalFrames(self):
    toFill = []
    list = self._make_video_dataset(self.originalFrames)

    last_name = os.path.basename(list[len(list)-1])
    last_index = int(last_name.split('.')[0])
    lastF = 1
    for i in range(1, last_index):
      if os.path.exists(self.originalFrames + str(i).zfill(10) + ".png"):
        print("Exist : " + str(i).zfill(10))
        lastF = i
      else:
        print("Not : " + str(i).zfill(10))

  def GetChannel(self, _filePath, type):
    conversions = []

    for _, filePath in enumerate(_filePath):
      myRenderData = self.myRenderData
      img = PIL.Image.open(filePath)

      if self.myRenderData.pixelUpscaleDowscaleBefore != 1:
          img = self.PilResize(img, self.myRenderData.pixelUpscaleDowscaleBefore)

      if myRenderData.alphaMethod == 0:
        conversions.append(img.convert('RGB'))
        continue


      red, green, blue, alpha = img.convert('RGBA').split()

      alphaRGB = alpha.convert("RGB")


      red = PIL.ImageChops.multiply(red, alphaRGB.split()[0])
      green = PIL.ImageChops.multiply(green, alphaRGB.split()[1])
      blue = PIL.ImageChops.multiply(blue, alphaRGB.split()[2])

      if myRenderData.alphaMethod == 1:
        return PIL.Image.merge("RGBA", (red, green, blue, alpha))

      if type == 0:
        conversions.append(PIL.Image.merge("RGB", (red, green, blue)))
      if type == 1:
        conversions.append(PIL.Image.merge("RGB", (alpha, alpha, alpha)))
      if type == 2:
        conversions.append(PIL.Image.merge("RGB", (red, green, alpha)))
    return conversions


  

  def LogPrint(self, str):
    if self.myRenderData.quiet == 0:
      print(str)

  def CheckAllScenes(self, myRenderData, diff):
    self.SetFolders(myRenderData)
    files = self._make_video_dataset(self.originalFrames)

    diffs = []
    for i in range(1, len(files)):
      if psnr.IsDiffScenes(files[i-1], files[i], diff):
        diffs.append(files[i-1])
        diffs.append(files[i])

    return diffs
    

  def SetFolders(self, myRenderData):
    name = os.path.basename(myRenderData.outStr)

    self.mainFolder =  myRenderData.outFolder
    self.originalFrames = self.mainFolder + "original_frames/"
    self.interpolatedFrames = self.mainFolder + "interpolated_frames/"
    self.renderFolder = self.mainFolder + "output_videos/"
    self.configFile = self.mainFolder + "config.json"
    self.framesFile = self.mainFolder + "frames.json"

    

    self.LogPrint("Main Folder: " + self.mainFolder)

    if not os.path.isdir(self.mainFolder):
      os.makedirs(self.mainFolder)

    if myRenderData.inputType != 3:
      myRenderData.ToJsonFile(self.configFile)
    else:
      do = myRenderData.doOriginal
      di = myRenderData.doIntepolation
      dv = myRenderData.doVideo
      myRenderData.FromJsonFile(self.configFile)
      myRenderData.doOriginal = do
      myRenderData.doIntepolation = di
      myRenderData.doVideo = dv
      myRenderData.uploadBar = None
      myRenderData.cleanOriginal = 0
      myRenderData.cleanInterpol = 0
      myRenderData.inputType = 3

  def StepExtractFrames(self, myRenderData):
    self.LogPrint("Starting PNG frames extraction!")

    
    if myRenderData.cleanOriginal == 1:
      if os.path.isdir(self.originalFrames):
        shutil.rmtree(self.originalFrames)
        time.sleep(2)

      if not os.path.isdir(self.originalFrames):
        os.makedirs(self.originalFrames)


      self.extract_frames(myRenderData.video, self.originalFrames)

    self.LogPrint("Finished PNG frames extraction!")

  def StepRenderInterpolation(self, myRenderData):
    self.LogPrint("Starting Interpolation!")

    files = self._make_video_dataset(self.originalFrames)
    if myRenderData.cleanInterpol == 1:
      if os.path.isdir(self.interpolatedFrames):
        shutil.rmtree(self.interpolatedFrames)
        #Try to avoid Denied access
        time.sleep(2)
    
    if not os.path.isdir(self.interpolatedFrames):
      os.makedirs(self.interpolatedFrames)

    if myRenderData.onlyRenderMissing == 1:
      framesCol = FrameCollection()
      framesCol.FromJsonFile(self.framesFile)

      fList = framesCol.frames
      missingInterpolations = []
      lastIndex =  0
      newList = []
      for i in range(0, len(fList)):
        framepath = self.originalFrames + fList[i]["frameName"]
        if os.path.isfile(framepath):
          newList.append(framepath)
          index = len(newList)- 1
          missingInterpolations.append(0)
          lastIndex = index
        else:
          missingInterpolations[lastIndex] += 1
      files = newList


    if myRenderData.interpolationMethod:
      myRenderData.batch_size = 1


    sceneDiff = -1
    if myRenderData.checkSceneChanges == 1:
      sceneDiff = myRenderData.sceneChangeSensibility

    frameFormat = "RGB"
    if myRenderData.alphaMethod == 1:
      frameFormat = "RGBA"

    dSet = DainDataset(files, setting.GetPad(), sceneDiff,
    frameFormat, myRenderData.splitFrames == 0, myRenderData.use_half)

    loader = torch.utils.data.DataLoader(
        dSet, batch_size = myRenderData.batch_size, num_workers= 0,
        pin_memory=True)

    startInterpolating = 0
    intFrames = 1

    if myRenderData.inputType == 3:
      resumeData = self.GetInterCounter()
      startInterpolating = resumeData['counter']
      intFrames = resumeData['index']

    qtList = tqdm(files, ncols=80)
      
      

    #It seen to be a memory leak in the DataLoader or in the loop?
    for i, (combo, X1, X2) in enumerate(loader):

      qtList.update(X1.size(0))
      qtList.set_postfix(file=combo['f1'][0][-14:], refresh=False)
      if i < startInterpolating:
        continue

      SetTiming("Starting loop")
      alphaList = []

      #Used to be able to skip frames, now it should be handled inside DataLoader
      skipInterpolation = False
      if skipInterpolation:
        qtList.set_postfix(file=str(i-1) + " Jumping intepolation", refresh=False)
        self.LogPrint("\n" + str(i-1) + " Jumping intepolation")
      else:

        if myRenderData.framerateConf == 2:
          loops = 1
        if myRenderData.framerateConf == 4:
          loops = 2
        if myRenderData.framerateConf == 8:
          loops = 3

        
        if myRenderData.interpolationMethod == 2 :

          fP = self.filename2timestamp(combo['f1'][0])
          fC = self.filename2timestamp(combo['f2'][0])

          loopExtra = max(round( ((fC - fP) / 1000 ) / (1 / (myRenderData.fps * self.myRenderData.framerateConf))), 1)

          loopCalculated = self.dirtHack(loops, loopExtra)
          #If we want to do less frames instead of more, remove the +1
          loopCalculated += 1
          loops = loopCalculated
          

        if myRenderData.interpolationAlgorithm == 0:
          totalFrames = 2 ** loops
          
          SetTiming("Starting Interpolation V0")
          
          images = self.DoInterpolation(X1, X2, loops)
          
          if myRenderData.splitFrames == 0:
            images = NumpyResultAsList(images, depad_value)
          else:
            images = NumpyResultAsList(images, None)

          SetTiming("Finishin Interpolation V0")

        if myRenderData.uploadBar != None:
          myRenderData.uploadBar(i/(len(files)))


      SetTiming("Starting  Image Loop")

      total = 0
      cleanFiles = []

      #Loop between all selected frames, horrible solution for the new Dataset Method
      for tt in range(0, len(combo['i'])):
        loops = myRenderData.framerateConf - 1
        if myRenderData.batch_size == 1:
          loops = len(images)


        start = self.filename2timestamp(combo['f1'][tt])
        end = self.filename2timestamp(combo['f2'][tt])

        cleanFiles.append( {"data" : combo['original'][tt].numpy(), 'timestamp' : start})

        for zz in range(0, loops):
          index = lerp(start, end, (total+1) / (len(images) + 1))
          cleanFiles.append({"data" : images[total], 'timestamp' : index})
          total += 1

      loopLen = len(cleanFiles)

      for zz in range(0, len(cleanFiles)):
        sel_file = cleanFiles[zz]["data"] 
        #If Alpha is calculated separated (slow method)        
        if myRenderData.alphaMethod == 2:
          image = PIL.Image.fromarray(sel_file).convert("RGB")
          alpha = PIL.Image.fromarray(sel_file).convert("RGB").split()[2]
          image.putalpha(alpha)
        else:
          image = PIL.Image.fromarray(sel_file).convert(frameFormat)
        
        #If mode is 2 or 3 (Keeping timestamp)
        if myRenderData.interpolationMethod == 2 or myRenderData.interpolationMethod == 3:
          SavePNG(self.myRenderData.pngcompress, image, self.interpolatedFrames + '/' + str(cleanFiles[zz]["timestamp"]).zfill(15) + '.png')
        else:
          SavePNG(self.myRenderData.pngcompress, image, self.interpolatedFrames + '/' + str(intFrames).zfill(15) + '.png')

        intFrames += 1

      self.SetInterpolCounter(intFrames, i+1)
      
      SetTiming("Finishin Loop")
    

    qtList.close()
   
    self.LogPrint("Ending Interpolation!")


  def StepCreateVideo(self, myRenderData):
    self.LogPrint("Now creating the video!")

    if not os.path.isdir(self.renderFolder):
      os.makedirs(self.renderFolder)

    self.create_video2(self.interpolatedFrames, self.renderFolder, self.myRenderData.framerateConf)
    self.LogPrint("Video finished!")


  def RenderVideo(self, renderVideoData):
    self.myRenderData = renderVideoData
    
    torch.set_flush_denormal(True)

    setting.AddCounter("RenderVideo")  
    
    fps = self.myRenderData.fps

    self.didSplitterPrinted = False

    if self.myRenderData.mute_ffmpeg == 1:
      self.panic = "-hide_banner -loglevel panic"


    self.SetFolders(self.myRenderData)

    torch.cuda.set_device(self.myRenderData.sel_process)

    self.myRenderData.optimizer = 0

    print("Use Half is: " + str(renderVideoData.use_half))

    
    if renderVideoData.useBenchmark == 1:
      torch.backends.cudnn.benchmark = True
    else:
      torch.backends.cudnn.benchmark = False

    if self.myRenderData.fillMissingOriginal == 1:
      self.FillMissingOriginalFrames()
      return

    if self.myRenderData.doOriginal:
      self.StepExtractFrames(self.myRenderData)

    if self.myRenderData.use_half:
      torch.set_default_tensor_type(torch.HalfTensor)

    with torch.cuda.amp.autocast(bool(self.myRenderData.use_half)):
      if self.myRenderData.doIntepolation:
        self.model = Configure(self, self.myRenderData)
        self.StepRenderInterpolation(self.myRenderData)


    if self.myRenderData.doVideo:
      self.StepCreateVideo(self.myRenderData)

    if self.myRenderData.uploadBar != None:
      self.myRenderData.uploadBar(1)




if __name__ == "__main__":
  import warnings
  warnings.filterwarnings("ignore")
