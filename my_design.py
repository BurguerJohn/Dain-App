import multiprocessing
import torch.multiprocessing
multiprocessing.freeze_support()
torch.multiprocessing.freeze_support()

import os

import my_DAIN_class

import cv2
import webbrowser
import warnings
import traceback
import RenderData
import clitest
import torch
import json
from my_client import CallClient


import design3
import my_imageUI
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QWidget
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt, QSettings, QStandardPaths
from design3 import Ui_Dialog

import setting


def debug_pickle(instance):
  attribute = None

  for k, v in instance.__dict__.items():
      try:
          cPickle.dumps(v)
      except:
          attribute = k
          break

  return attribute


class Worker(QtCore.QThread):
    
	def __init__(self, parent=None, window = None, renderList=None, doOriginal = 1, doIntepolation = 1, doVideo = 1, device = 0):
		QtCore.QThread.__init__(self, parent)
		self.renderList = renderList
		self.window = window
		self.doOriginal = doOriginal
		self.doIntepolation = doIntepolation
		self.doVideo = doVideo
		self._device = device

		
		

	def run(self):
		isSuccess = False
		renderList = self.renderList

		for i in range(0, len(renderList)):
			myRenderData = renderList[i]
			myRenderData.doOriginal = self.doOriginal
			myRenderData.doIntepolation = self.doIntepolation
			myRenderData.doVideo = self.doVideo


			try:
				dain_class = my_DAIN_class.DainClass()
				dain_class.RenderVideo(myRenderData)
				isSuccess = True
			except RuntimeError as e:
				tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
				logf = open("crash_log.txt", "w")
				logf.write("\nApp Crash Log "+str("".join(tb_str))+"\n")
				print("".join(tb_str))
				if "CUDA out of memory" in str(e):
					self.window.messageSignal.emit("Error", "Your CUDA ran out of memory, the video resolution is to big for your card to handle it.\nCheck out the 'Fix OutOfMemory Options' tab for solutions.\n\n" + str(e))
				else:
					self.window.messageSignal.emit("Error", "Crash [Saved on crash_log.txt]:<br />Check out Discord for possible solutions:<br /><a href=\"http://discordapp.com/channels/668789174257844225/674943474142937129/\">Check Errors fix on Discord.</a>  <br /><br />Error: " + str(e))

			except Exception as e:
				tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
				logf = open("crash_log.txt", "w")
				logf.write("\nApp Crash Log "+str("".join(tb_str))+"\n")
				print("".join(tb_str))
				self.window.messageSignal.emit("Error", "Crash [Saved on crash_log.txt]:<br />Check out Discord for possible solutions:<br /><a href=\"http://discordapp.com/channels/668789174257844225/674943474142937129/\">Check Errors fix on Discord.</a>  <br /><br />Error: " + str(e))
			
		if isSuccess:
			print("Your render is complete.\nCheck out the output folder for the result.")



class My_Ui_Dialog(Ui_Dialog):
	
	progressSignal = QtCore.pyqtSignal(int)
	messageSignal = QtCore.pyqtSignal(str, str)

	model_path = "model_weights/"

	def __getstate__(self):
		#print("Im pickle rick")
		return None

	def setupUi(self, Dialog):
		super().setupUi(Dialog)

		localAppData = QStandardPaths.writableLocation(QStandardPaths.AppLocalDataLocation)
		self.prefabsFolder = os.path.join(localAppData, "prefabs")

		self.selectFiles = ""
		self.selectInputSeq = ""
		self.selectResume = ""
		self.selectedOutFolder = ""
		self.framerateConf = 1
		self.inputType = 0
		self.Dialog = None
		self._device = 0
		
		self.progressSignal.connect(self.UpdateLoadingProp)
		self.messageSignal.connect(self.MessageBox)

		self.Dialog = Dialog

		self.inputFileBtn.clicked.connect(self.OpenInputFile)
		self.inputFolderBtn.clicked.connect(self.OpenInputSequence)
		self.inputResumeFolder.clicked.connect(self.OpenInputResume)

		self.label.setText("DAIN-APP "+ RenderData.GetVersion() +" #DAINAPP")

		self.whitePalette = self.Dialog.palette()
		self.settings = QSettings("DainApp", "default")

		if self.settings.value("pallet") == "dark":
			self.SetDarkMode()
		else:
			self.SetWhiteMode()


		self.tabWidget.setCurrentIndex(0)

		self.verifyScenes.clicked.connect(self.OnVerifyScenes)

		self.outputFolderBtn.clicked.connect(self.OpenOutputFolder)
		self.renderBtn.clicked.connect(self.render)

		self.renderFrames.clicked.connect(self.renderO)
		self.renderInterpolation.clicked.connect(self.renderI)
		self.renderVideo.clicked.connect(self.renderV)

		self.patreonBtn.clicked.connect(self.OpenPatreon)
		self.discBtn.clicked.connect(self.OpenDiscord)
		self.credits.clicked.connect(self.OpenCredits)
		self.pushButton.clicked.connect(self.OpenPatrons)

		self.interpolationLevel.currentIndexChanged.connect(self.onInterpolationChange)
		self.interpolMethod.currentIndexChanged.connect(self.SetInterpolMethodText)
		self.intAlgo.currentIndexChanged.connect(self.OnInterAlgoChange)

		self.brightBtn.clicked.connect(self.SetWhiteMode)
		self.darkBtn.clicked.connect(self.SetDarkMode)

		self.radioInputVideos.toggled.connect(self.radio_clicked)
		self.radioInputPNG.toggled.connect(self.radio_clicked)
		self.radioResumeRender.toggled.connect(self.radio_clicked)

		self.savePlace.clicked.connect(self.OnSavePrefab)
		self.loadPlace.clicked.connect(self.OnLoadPrefab)
		self.deletePlace.clicked.connect(self.OnDeletePrefab)


		self.SetLayoutVisibility()
		self.SetInterpolMethodText()
		self.GetDevices()
		self.SetPrefabList()

		self.SetModels()


		if not torch.cuda.is_available():
			self.MessageBox("Error", "WARNING!\nNo available CUDA detected.\nYou will not be able to create interpolations with this application.\nOnly Graphic Cards that have Hardware compatible with CUDA 5.0 or above can be used.")
		elif float(torch.version.cuda) < 5:
			self.MessageBox("Error", "WARNING!\nYour CUDA version seen to be outdated.\nYou will not be able to create interpolations with this application.\nOnly Graphic Cards that have Hardware compatible with CUDA 5.0 or above can be used.")

	def OnInterAlgoChange(self):
		for i in reversed(range(0, self.interpolationLevel.count())):
			self.interpolationLevel.removeItem(i)
		method = self.intAlgo.currentIndex()

		if method == 0:
			self.interpolationLevel.addItem("Interpolate 2X")
			self.interpolationLevel.addItem("Interpolate 4X")
			self.interpolationLevel.addItem("Interpolate 8X")
			self.interpolationLevel.setCurrentIndex(0)
		if method == 1:
			self.interpolationLevel.addItem("Interpolate 2X")
			self.interpolationLevel.addItem("Interpolate 3X")
			self.interpolationLevel.addItem("Interpolate 4X")
			self.interpolationLevel.addItem("Interpolate 5X")
			self.interpolationLevel.addItem("Interpolate 6X")
			self.interpolationLevel.addItem("Interpolate 7X")
			self.interpolationLevel.addItem("Interpolate 8X")
			self.interpolationLevel.setCurrentIndex(0)


	def SetInterpolMethodText(self):
		interpolMethod =  self.interpolMethod.currentIndex()

		text = ''
		if interpolMethod == 0:
			text = 'In this mode:\n Each frame will appear for the exatly the same time in the output.\nIf the animation have variable framerate (gifs) it can alter the animation speed and timing.'
		if interpolMethod == 1:
			text = 'In this mode:\n The app will try to remove duplicated frames and\neach frame will appear for the exatly the same time in the output.\nIf the animation have variable framerate (gifs) it can alter the animation speed and timing.'
		if interpolMethod == 2:
			text = 'In this mode:\n The app will try to remove duplicated frames and\neach frame will be kep at the same timestamp it was before.\nIt then will try to guess how much new frames it need to create between\neach original frame to generate the desired output fps.'
		if interpolMethod == 3:
			text = 'In this mode:\n The app will try to remove duplicated frames and\neach frame will be kep at the same timestamp it was before.\nIt then will create exacly the number of selected new frames [2X/3X/4X/5X/etc] between\neach original frame to generate the desired output fps.'
		#print(text)
		self.modeDesc.setText(text)

	def OnSavePrefab(self):
		name = self.placeholderName.text()
		self.SavePrefab(name)
		self.placeholderName.setText("")

	def OnLoadPrefab(self):
		index = self.placeholderList.currentIndex()
		if index >= 0:
			data = self.prefabList[index]
			self.LoadPrefab(data)

	def OnDeletePrefab(self):
		if self.placeholderList.currentIndex() >= 0 :
			self.ConfirmBox("Caution", "This will remove the selected prefab, continue?", self.DeletePrefab)


	def SavePrefab(self, name = "No-Name"):
		vr = ["interpolMethod","animMethod","intAlgo","fpsInput","interpolationLevel","pngCompress",
"crfVal","dontInterpolateScenes","lineEdit_2","cleanInterpol","perfectLoop","audioVersion",
"fpsLimit","to60","to60C1","to60C2","limitPalette","pixelUpscaleDowscaleBefore",
"pixelDownscaleUpscaleAfter","pixelUpscaleAfter","useResize","widthValue","useSplit",
"splitSizeX","splitSizeY","splitPad", "useHalf", "doBenchmark", "deviceList", "fastMode"]
		#savePlace

		vList = [("name", name)]



		for i in range(0, len(vr)):
			inst = getattr(self, vr[i])
			if isinstance(inst, QtWidgets.QComboBox):
				vList.append( (vr[i], inst.currentIndex()) )
			elif isinstance(inst, QtWidgets.QLineEdit):
				vList.append( (vr[i], inst.text()) )
			elif isinstance(inst, QtWidgets.QCheckBox):
				vList.append( (vr[i], int(inst.isChecked())  ))
			else:
				print("Atribute not found " + st(inst))

		
		file = os.path.join(self.prefabsFolder, "config_{}.json".format(name))

		if not os.path.exists(self.prefabsFolder):
			os.makedirs(self.prefabsFolder)

		print(vList)

		with open(file, 'w') as outfile:
			outfile.write(str(json.dumps(vList)))

		self.SetPrefabList()



	def LoadPrefab(self, data):
		for key in data:
			item = data[key]
			try:
				inst = getattr(self, key)
			except AttributeError as ae:
				#print(ae)
				continue


			if isinstance(inst, QtWidgets.QComboBox):
				inst.setCurrentIndex(item)
			elif isinstance(inst, QtWidgets.QLineEdit):
				inst.setText(item)
			elif isinstance(inst, QtWidgets.QCheckBox):
				inst.setChecked(bool(item))
			else:
				print("Atribute not found " + st(inst))


		self.OnInterAlgoChange()

	def DeletePrefab(self, ret):
		index = self.placeholderList.currentIndex()
		data = self.prefabList[index]
		name = ""
		for key in data:
			item = data[key]
			if key == "name":
				name = item
				break

		file = os.path.join(self.prefabsFolder, "config_{}.json".format(name))
		if os.path.exists(self.prefabsFolder):
			os.remove(file)

		self.SetPrefabList()


	def SetPrefabList(self):
		if not os.path.exists(self.prefabsFolder):
			return

		prefabs = sorted(os.listdir(self.prefabsFolder))

		vList = []

		for i in reversed(range(0, self.placeholderList.count())):
			self.placeholderList.removeItem(i)

		

		for i in range(0, len(prefabs)):
			keyname = {}
			with open(os.path.join(self.prefabsFolder, prefabs[i]), 'r') as f:
				vrs = json.loads(f.read())
			
			for x in range(0, len(vrs)):
				keyname[vrs[x][0]] = vrs[x][1]

			vList.append( keyname )
			self.placeholderList.addItem(keyname['name'])

		self.prefabList = vList


	def SetModels(self):
		for i in reversed(range(0, self.flowModel.count())):
			self.flowModel.removeItem(i)

		for model in sorted(os.listdir(self.model_path)):
			self.flowModel.addItem(model)



	def radio_clicked(self, enabled):
		self.SetLayoutVisibility()

	def SetLayoutVisibility(self):
		input = self.GetInputType()

		self.inputType = input
		self.SetResumeOpts()       

		self.inputVideosLayout.setVisible(input == 1)
		self.inputSequenceLayout_2.setVisible(input == 2)
		self.inputResumeLayout_2.setVisible(input == 3)

	def GetInputType(self):
		rA = self.radioInputVideos.isChecked()
		rB = self.radioInputPNG.isChecked()
		rC = self.radioResumeRender.isChecked()

		if rA:
			return 1
		if rB:
			return 2
		if rC:
			return 3

	def SetResumeOpts(self):
		mode = self.inputType != 3
		self.exportType.setEnabled(mode)
		self.outputFolderBtn.setEnabled(mode)
		self.animMethod.setEnabled(mode)
		self.interpolMethod.setEnabled(mode)
		self.alphaOpt.setEnabled(mode)
		self.fpsInput.setEnabled(mode)

		self.audioVersion.setEnabled(mode)

		self.useSplit.setEnabled(mode)
		self.splitSizeX.setEnabled(mode)
		self.splitSizeY.setEnabled(mode)
		self.splitPad.setEnabled(mode)

		self.useResize.setEnabled(mode)
		self.widthValue.setEnabled(mode)

		self.dontInterpolateScenes.setEnabled(mode)
		self.lineEdit_2.setEnabled(mode)

		self.limitPalette.setEnabled(mode)
		self.perfectLoop.setEnabled(mode)
		self.cleanInterpol.setEnabled(mode)

		self.fpsLimit.setEnabled(mode)
		self.to60.setEnabled(mode)
		self.to60C1.setEnabled(mode)
		self.to60C2.setEnabled(mode)


	def OnVerifyScenes(self, Dialog):
		data = self.CreateRenderData()
		if len(data) == 0:
			return

		diff = int(self.lineEdit_2.text())

		myRenderData = data[0]
		myRenderData.doOriginal = 1
		myRenderData.doIntepolation = 0
		myRenderData.doVideo = 0
		myRenderData.cleanOriginal = 0

		dain_class = my_DAIN_class.DainClass()
		dain_class.RenderVideo(myRenderData)
		diffs = dain_class.CheckAllScenes(myRenderData, diff)

		if len(diffs) != 0:
			ui = my_imageUI.My_Ui_Dialog()
			dia = QtWidgets.QDialog()
			dia.ui = ui
			dia.ui.setupUi(dia)
			dia.setAttribute(QtCore.Qt.WA_DeleteOnClose)

			for i in range(0, len(diffs), 2):
				ui.AddComp(diffs[i], diffs[i+1])

			dia.exec_()
		else:
			self.MessageBox("Message", "No scene changes found with this setting.")


	def OpenCredits(self, Dialog):
		#ui = QtWidgets.QApplication(self.Dialog)
		dia = QtWidgets.QDialog()
		#dia.ui = ui
		
		c = [
		"<b>Dain-App Credits:</b>",
		"",
		"&nbsp;&nbsp;Dain Creator:",
		"&nbsp;&nbsp;&nbsp;&nbsp;Wenbo Bao",
		"<br />",
		"&nbsp;&nbsp;Dain-App Development:",
		"&nbsp;&nbsp;&nbsp;Gabriel Poetsch",
		"&nbsp;&nbsp;&nbsp;Wenbo Bao",
		"<br />", 
		"&nbsp;&nbsp;App Testers:",
		"&nbsp;&nbsp;&nbsp;Mikael Poetsch",
		"&nbsp;&nbsp;&nbsp;Estevan Sanchez-Wentz",
		"&nbsp;&nbsp;&nbsp;Akira Baes",
		"&nbsp;&nbsp;&nbsp;Anh Minh Pho",
		"&nbsp;&nbsp;&nbsp;Hubert Sontowski",
		"&nbsp;&nbsp;&nbsp;Mr. Anon",
		"&nbsp;&nbsp;&nbsp;Aury",
		"<br />",
		"&nbsp;&nbsp;Dream Maker Patreon Supporters:",
		"&nbsp;&nbsp;&nbsp;June"
		]

		dia.setObjectName("Dialog")
		dia.resize(500, 600)
		gridLayout = QtWidgets.QGridLayout(dia)
		gridLayout.setObjectName("gridLayout")

		
		verticalLayout = QtWidgets.QVBoxLayout()
		verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
		verticalLayout.setObjectName("verticalLayout")
		label = QtWidgets.QLabel(dia)
		font = QtGui.QFont()
		font.setPointSize(16)
		font.setBold(False)
		font.setUnderline(False)
		font.setWeight(30)
		label.setFont(font)
		label.setScaledContents(False)
		label.setAlignment(QtCore.Qt.AlignLeft)
		label.setObjectName("label")
		label.setText("<br />".join(c))
		verticalLayout.addWidget(label)
		gridLayout.addLayout(verticalLayout, 0, 0, 1, 1)

		dia.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		dia.exec_()

	def OpenPatrons(self, Dialog):
		dia = QtWidgets.QDialog()
		
		f = open("patrons.txt", "r", encoding="utf-8")
		c = f.read()

		dia.setObjectName("Dialog")
		dia.resize(500, 600)
		gridLayout = QtWidgets.QGridLayout(dia)
		gridLayout.setObjectName("gridLayout")


		verticalLayout = QtWidgets.QVBoxLayout()
		verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
		verticalLayout.setObjectName("verticalLayout")

		label = QtWidgets.QTextEdit(dia)
		font = QtGui.QFont()
		font.setPointSize(16)
		font.setBold(False),  
		font.setUnderline(False)
		font.setWeight(30)
		label.setFont(font)
		label.setReadOnly(True)
		label.setAlignment(QtCore.Qt.AlignLeft)
		label.setObjectName("label")
		label.setText(c)
		verticalLayout.addWidget(label)
		gridLayout.addLayout(verticalLayout, 0, 0, 1, 1)

		dia.setAttribute(QtCore.Qt.WA_DeleteOnClose)
		dia.exec_()


	def OpenInputFile(self, Dialog):
		                                                                            
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.selectFiles, _ = QFileDialog.getOpenFileNames()
		if self.selectFiles:
		    print(self.selectFiles)
		

		if len(self.selectFiles) == 1:
			self.fpsInput.setReadOnly(False)
			ftp = self.GetFPS(self.selectFiles[0])
			self.fpsInput.setText(str(ftp))
			self.onInterpolationChange()
			self.inputFileLabel.setText(" ".join(self.selectFiles))
		else:
			self.fpsInput.setReadOnly(True)
			self.fpsInput.setText("Multiple Files selected.")
			self.onInterpolationChange()
			self.inputFileLabel.setText("Multiple files selected.")


	def OpenInputSequence(self, Dialog):
		self.fpsInput.setReadOnly(False)                     
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.selectInputSeq = QFileDialog.getExistingDirectory()
		if self.selectInputSeq:
		    print(self.selectInputSeq)
		    self.selectFiles = [self.selectInputSeq]
		self.inputFolderLabel.setText(self.selectInputSeq)
		self.fpsInput.setText(str(0))
		self.onInterpolationChange()

	def OpenInputResume(self, Dialog):                                                                                  
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.selectFiles, _ = QFileDialog.getOpenFileNames(QFileDialog(), "", "", "json(*.json)")
		if len(self.selectFiles) == 1:
		    print(self.selectFiles[0])
		    self.selectedOutFolder = os.path.dirname(self.selectFiles[0]) + "/"
		    
		self.inputResumeLabel.setText(self.selectResume)
		self.fpsInput.setText("0")
		self.onInterpolationChange()
		if len(self.selectFiles) == 1:
			self.render()



	def OpenOutputFolder(self, Dialog):                                                                                  
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		self.selectedOutFolder = QFileDialog.getExistingDirectory()
		if self.selectedOutFolder:
		    print(self.selectedOutFolder)
		self.outputFolderLabel.setText(self.selectedOutFolder)

	def renderO(self, Dialog = None):
		self.StartRender(1,0,0)
	def renderI(self, Dialog = None):
		self.StartRender(0,1,0)
	def renderV(self, Dialog = None):
		self.ConfirmBox("Caution", "This may rename a few files inside the interpolated_frames folder, are you sure?", self.renderVConfir)
		
	def renderVConfir(self, answer):
		if answer:
			self.StartRender(0,0,1)

	def render(self, Dialog = None):
		self.StartRender(1,1,1)

	def LoadConfigToUI(renderData):
		resize = self.useResize.isChecked()
		resizeWidth = int(self.widthValue.text())
		interAlgo = self.intAlgo.currentIndex()
		interpolCode = self.framerateConf
		interpolMethod =  self.interpolMethod.currentIndex()
		alphaOpt = self.alphaOpt.currentIndex()
		audioVersion = int(self.audioVersion.isChecked())

		to60 = int(self.to60.isChecked())
		to60Smooth = int(self.to60C1.isChecked())
		to60Sharp = int(self.to60C2.isChecked())

		exportExt = self.exportType.currentIndex()
		animMethod = self.animMethod.currentIndex()

		splitFrames = int(self.useSplit.isChecked())
		cleanInterpol = int(self.cleanInterpol.isChecked())
		fpsLimit = int(self.fpsLimit.text())
		checkScenes = int(self.dontInterpolateScenes.isChecked())

		debugKeepDuplicates = int(self.debugKeepDuplicates.isChecked())
		onlyInterpolateMissing = int(self.onlyInterpolateMissing.isChecked())


		checkScenes = int(self.dontInterpolateScenes.isChecked())
		checkScenes = int(self.dontInterpolateScenes.isChecked())


		sceneSenvility =  int(self.lineEdit_2.text())

		useHalf = int(self.useHalf.isChecked())

		
		
		splitSizeX = int(self.splitSizeX.text())
		splitSizeY = int(self.splitSizeY.text())
		splitPad = int(self.splitPad.text())

		myRenderData.palette = int(self.limitPalette.isChecked())
		myRenderData.resc = int(resize)
		myRenderData.maxResc = resizeWidth
		myRenderData.loop = int(self.perfectLoop.isChecked())
	

	def CreateRenderData(self):
		if self.selectFiles == "":
			self.MessageBox("Error", "Select a file before hiting Render.")
			return []
		if self.selectedOutFolder == "":
			self.MessageBox("Error", "Select a output folder before hiting Render.")
			return []

		warnings.filterwarnings("ignore")

		renderList = []
		for i in range(0, len(self.selectFiles)):

			selectFile = self.selectFiles[i]
			ftp = self.fpsInput.text()
			path = os.path.dirname(selectFile)
			name = os.path.basename(selectFile)
			name_no_ext = name.split(".")[0]

			if self.inputType != 3:
				outFolder = self.selectedOutFolder +"/"+name_no_ext+"/"
			else:
				outFolder = self.selectedOutFolder

			resize = self.useResize.isChecked()
			resizeWidth = 0
			if resize:
				resizeWidth = int(self.widthValue.text())

			useWatermark = 0

			interAlgo = self.intAlgo.currentIndex()
			interpolCode = self.framerateConf
			interpolMethod =  self.interpolMethod.currentIndex()
			alphaOpt = self.alphaOpt.currentIndex()
			audioVersion = int(self.audioVersion.isChecked())

			to60 = int(self.to60.isChecked())
			to60Smooth = int(self.to60C1.isChecked())
			to60Sharp = int(self.to60C2.isChecked())

			exportExt = self.exportType.currentIndex()
			animMethod = self.animMethod.currentIndex()

			splitFrames = int(self.useSplit.isChecked())
			splitSizeX = 0
			splitSizeY = 0
			splitPad = 0

			cleanInterpol = int(self.cleanInterpol.isChecked())
			fpsLimit = int(self.fpsLimit.text())
			checkScenes = int(self.dontInterpolateScenes.isChecked())

			debugKeepDuplicates = int(self.debugKeepDuplicates.isChecked())
			onlyInterpolateMissing = int(self.onlyInterpolateMissing.isChecked())

			flowPath = "./"+ self.model_path + self.flowModel.currentText()

			device = int(self.deviceList.currentIndex())
		


			checkScenes = int(self.dontInterpolateScenes.isChecked())
			checkScenes = int(self.dontInterpolateScenes.isChecked())

			sf = self.smoothFlow.text()
			ff = self.flowForce.text()
			if sf == "":
				sf = 0
			if ff == "":
				ff = 20
			flowForce = int(ff)
			SmoothFlow = int(sf)

			sceneSenvility =  int(self.lineEdit_2.text())

			useHalf = int(self.useHalf.isChecked())


			if splitFrames == 1:
				splitSizeX = int(self.splitSizeX.text())
				splitSizeY = int(self.splitSizeY.text())
				splitPad = int(self.splitPad.text())

				if splitSizeX == 0 or splitSizeY == 0:
					self.MessageBox("Error", "Division value should not be zero. If you do not want divisions in a axis, use the value 1.")
					return []

			
			batchSize =  int(self.batchSize.text())

			ext = "mp4"
			if exportExt == 0:
				ext = "mp4"
			elif exportExt == 1:
				ext = "webm"
			elif exportExt == 2:
				ext = "gif"
			elif exportExt == 3:
				ext = "apng"

			if len(self.selectFiles) > 1:
				ftp = self.GetFPS(self.selectFiles[i])

			if self.inputType != 3:
				if float(ftp) == 0:
					self.MessageBox("Error", "Input FPS cannot be zero.")
					return []


			myRenderData = RenderData.RenderData()
			myRenderData.inputType = self.inputType
			myRenderData.video = selectFile
			myRenderData.outStr = outFolder+"/"+name_no_ext+"."+ext
			myRenderData.fps = float(ftp)
			myRenderData.palette = int(self.limitPalette.isChecked())
			myRenderData.resc = int(resize)
			myRenderData.maxResc = resizeWidth
			myRenderData.loop = int(self.perfectLoop.isChecked())
			myRenderData.uploadBar = self.UpdateLoading
			myRenderData.useWatermark = useWatermark
			myRenderData.interpolationAlgorithm = interAlgo
			myRenderData.framerateConf = interpolCode
			myRenderData.use60 = to60
			myRenderData.use60C1 = to60Smooth
			myRenderData.use60C2 = to60Sharp
			myRenderData.interpolationMethod = interpolMethod
			myRenderData.exportPng = 0
			myRenderData.useAnimationMethod = animMethod
			myRenderData.splitFrames = splitFrames
			myRenderData.splitSizeX = splitSizeX
			myRenderData.splitSizeY = splitSizeY
			myRenderData.splitPad = splitPad
			myRenderData.alphaMethod = alphaOpt
			myRenderData.checkSceneChanges = checkScenes
			myRenderData.sceneChangeSensibility = sceneSenvility
			myRenderData.model = flowPath
			myRenderData.sel_process = device

			myRenderData.use60RealFps = fpsLimit
			myRenderData.inputMethod = self.GetInputType()
			myRenderData.outFolder = outFolder
			myRenderData.cleanInterpol = cleanInterpol
			myRenderData.audioVersion = audioVersion
			myRenderData.onlyRenderMissing = onlyInterpolateMissing
			myRenderData.debugKeepDuplicates = debugKeepDuplicates
			myRenderData.use_half = useHalf
			myRenderData.fastMode = int(self.fastMode.isChecked())

			myRenderData.version = RenderData.GetVersion()
			myRenderData.quiet = 0

			myRenderData.pixelBgColor = (255,0,127)

			myRenderData.crf = int(self.crfVal.text())
			myRenderData.pngcompress = int(self.pngCompress.text())

			myRenderData.pixelUpscaleDowscaleBefore = self.pixelUpscaleDowscaleBefore.currentIndex() + 1
			myRenderData.pixelDownscaleUpscaleAfter = self.pixelDownscaleUpscaleAfter.currentIndex() + 1
			myRenderData.pixelUpscaleAfter = self.pixelUpscaleAfter.currentIndex() + 1

			myRenderData.mute_ffmpeg = 1 - int(self.ffmpegPrint.isChecked())
			myRenderData.cleanCudaCache = 1 - int(self.dontCleanCache.isChecked())
			myRenderData.useBenchmark = int(self.doBenchmark.isChecked())


			myRenderData.batch_size = batchSize

			myRenderData.flowForce = flowForce
			myRenderData.SmoothFlow = SmoothFlow

			renderList.append(myRenderData)
		return renderList

	def StartRender(self, doOriginal, doIntepolation, doVideo):
		if self.selectFiles == "":
			self.MessageBox("Error", "Select a file before hiting Render.")
			return

		if self.inputType != 3:
			if self.selectedOutFolder == "":
				self.MessageBox("Error", "Select a output folder before hiting Render.")
				return

		warnings.filterwarnings("ignore")

		renderList = self.CreateRenderData()

		t = Worker(parent = Dialog, window = self, renderList = renderList, doOriginal = doOriginal, doIntepolation = doIntepolation, doVideo = doVideo, device = self._device)
		t.start()


	def GetFPS(self, pathVideo):
		cam = cv2.VideoCapture(pathVideo)
		print("Input FPS: " + str(cam.get(cv2.CAP_PROP_FPS)))
		return cam.get(cv2.CAP_PROP_FPS)
	def UpdateLoading(self, value):
		intVal = int(value * 100)
		self.progressSignal.emit(intVal)


	def UpdateLoadingProp(self, intVal):
		self.progressBar.setProperty("value", intVal)
		
	def CheckScenesDiff():
		self.render()


	def MessageBox(self, title, message):
		win = QWidget(self.Dialog, )
		QMessageBox.about(win, title, message)
		win.show()
	def ConfirmBox(self, title, message, callback):
		win = QWidget()
		buttonReply = QMessageBox.question(win, title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
		if buttonReply == QMessageBox.Yes:
			callback(True)
		else:
			callback(False)
		win.show()
	def onInterpolationChange(self):
		inputType = self.GetInputType();


		if len(self.selectFiles) > 1 and inputType == 1:
			self.outputFps.setText("Multiple Files selected")
			return

		try:
			iinput = float(self.fpsInput.text())
		except ValueError:
			iinput = 0

		algorith = self.intAlgo.currentIndex()
		index = self.interpolationLevel.currentIndex()
		multi = 0

		if algorith == 0:
			if index == 0:
				multi = 2
			elif index == 1:
				multi = 4
			elif index == 2:
				multi = 8
		if algorith == 1:
			multi = 2 + index


		self.framerateConf = multi
		self.outputFps.setText(str(multi * iinput))
		
	def OpenPatreon(self, Dialog):
		webbrowser.open('https://www.patreon.com/DAINAPP')
	def OpenDiscord(self, Dialog):
		webbrowser.open('https://discord.gg/fF7rcgS')
	def SetDarkMode(self):
		C1 = 22

		palette = QPalette()
		palette.setColor(QPalette.Window, QColor(53, 53, 53))
		palette.setColor(QPalette.WindowText, Qt.white)
		palette.setColor(QPalette.Base, QColor(25, 25, 25))
		palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
		palette.setColor(QPalette.ToolTipBase, Qt.white)
		palette.setColor(QPalette.ToolTipText, Qt.white)
		palette.setColor(QPalette.Text, Qt.white)
		palette.setColor(QPalette.Button, QColor(53, 53, 53))
		palette.setColor(QPalette.ButtonText, Qt.white)
		palette.setColor(QPalette.BrightText, Qt.red)
		palette.setColor(QPalette.Link, QColor(42, 130, 218))
		palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
		palette.setColor(QPalette.HighlightedText, Qt.black)

		self.Dialog.setPalette( palette )
		self.settings.setValue("pallet", "dark")
	def SetWhiteMode(self):
		self.Dialog.setPalette( self.whitePalette )
		self.settings.setValue("pallet", "white")

	def GetDevices(self):
		count = torch.cuda.device_count()

		for i in reversed(range(0, self.deviceList.count())):
			self.deviceList.removeItem(i)
			
		for i in range(0, count):
			name = torch.cuda.get_device_name(i)
			self.deviceList.addItem(name)

		if self.deviceList.count() > 0:
			self.deviceList.setCurrentIndex(torch.cuda.current_device())


import sys

if __name__ == "__main__":
	print("Starting...")
	
	setting.AddCounter("Starting")	
	if clitest.args.cli:
		CallClient()
	else:
		app = QtWidgets.QApplication(sys.argv)
		app.setStyle("Fusion")
		app.setApplicationName("dainapp")
		Dialog = QtWidgets.QDialog()
		ui = My_Ui_Dialog()
		ui.setupUi(Dialog)

		Dialog.setWindowFlags(Dialog.windowFlags() |
        QtCore.Qt.WindowMinimizeButtonHint |
        QtCore.Qt.WindowSystemMenuHint)


		Dialog.show()
		sys.exit(app.exec_())
	
	
