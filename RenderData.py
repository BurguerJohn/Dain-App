import json
from json import JSONEncoder
import inspect
import os
import time


__APPVERSION = "1.00"
__TIMER = 0

def GetVersion():
	return __APPVERSION

def PrintTime(tag):
	global __TIMER
	return
	print(tag +": " + str(time.process_time() - __TIMER))
	__TIMER = time.process_time()



class ObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "to_json"):
            return self.default(obj.to_json())
        elif hasattr(obj, "__dict__"):
            d = dict(
                (key, value)
                for key, value in inspect.getmembers(obj)
                if not key.startswith("__")
                and not inspect.isabstract(value)
                and not inspect.isbuiltin(value)
                and not inspect.isfunction(value)
                and not inspect.isgenerator(value)
                and not inspect.isgeneratorfunction(value)
                and not inspect.ismethod(value)
                and not inspect.ismethoddescriptor(value)
                and not inspect.isroutine(value)
            )
            return self.default(d)
        return obj

class MyEncoder(JSONEncoder):
	def default(self, o):
	    return o.__dict__


class FrameCollection():
  def __init__(self):
  	self.frames = []
  def AddFrame(self, frame):
    self.frames.append(frame)
  def duration(self, frame):
  	return int(frame["tsEnd"]) - int(frame["tsStart"])
  def ToJson(self):
    return json.dumps(self, cls=ObjectEncoder)
  def FromJson(self, data):
    self.__dict__ = json.loads(data)
  def ToJsonFile(self, path):
  	if os.path.exists(path):
  		os.remove(path)
  	with open(path, 'w') as outfile:
  		outfile.write(str(self.ToJson()))
  def FromJsonFile(self, path):
  	if os.path.exists(path):
	    with open(path, 'r') as f:
	      self.__dict__ = json.loads(f.read())

class FrameData():
  frameName = 0
  tsStart = 0
  tsEnd = 0


class RenderData:
	def __init__(self):
		self.video = ""
		self.outStr = ""
		self.outFolder = ""
		self.fps = 5
		self.palette = 0
		self.resc = 0
		self.maxResc = 0
		self.loop = 0
		self.uploadBar = None
		self.useWatermark = 0
		self.framerateConf = 0
		self.use60RealFps = 60
		self.use60 = 0
		self.use60C1 = 0
		self.use60C2 = 0
		self.interpolationMethod = 0
		self.exportPng = 0
		self.useAnimationMethod = 0
		self.splitFrames = 0
		self.splitSizeX = 2
		self.splitSizeY = 2
		self.splitPad = 150
		self.alphaMethod = 0
		self.inputMethod = 1
		self.cleanOriginal = 1
		self.cleanInterpol = 0
		self.doOriginal = 1
		self.doIntepolation = 1
		self.doVideo = 1
		self.checkSceneChanges = 0
		self.sceneChangeSensibility = 10
		self.inputType = 0
		self.audioVersion = 0
		self.interpolationAlgorithm = 0
		self.cleanCudaCache = 1
		self.quiet = 0
		self.fillMissingOriginal = 0
		self.model = "./model_weights/best.pth"
		self.onlyRenderMissing = 0
		self.debugKeepDuplicates = 0

		self.use_half = 0
		self.version  = ""

		self.crf = 17
		self.pngcompress = 6
		#Pixelart
		self.pixelUpscaleDowscaleBefore = 1
		self.pixelDownscaleUpscaleAfter = 1
		self.pixelUpscaleAfter = 1
		self.pixelBgColor = (255,0,127)

		self.mute_ffmpeg = 1

		self.useBenchmark = 1

		self.sel_process = 0

		self.ShareFlow = False
		self.SmoothFlow = 0
		self.flowForce = 20
		self.batch_size = 1
		self.fastMode = 0


	def ToJson(self):
		return json.dumps(self, cls=ObjectEncoder)
	def FromJson(self, data):
		self.__dict__ = json.loads(data)
	def ToJsonFile(self, path):
		if os.path.exists(path):
			os.remove(path)
		with open(path, 'w') as outfile:
			outfile.write(str(self.ToJson()))
	def FromJsonFile(self, path):
		with open(path, 'r') as f:
			self.__dict__ = json.loads(f.read())
