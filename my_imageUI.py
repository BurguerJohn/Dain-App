from imageUI import Ui_Dialog

from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel, QFileDialog, QMessageBox, QWidget
from PyQt5.QtGui import QPixmap

class My_Ui_Dialog(Ui_Dialog):
	myDialog = None
	def setupUi(self, Dialog):
		super().setupUi(Dialog)
		self.myDialog = Dialog
		self.picWindowLayout.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
		self.picWindowLayout.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)


	def AddComp(self, im1, im2):
		box = self.CreateBox(self.myDialog, im1, im2)
		self.verticalLayout_2.addLayout(box)
		

	def CreateBox(self, Dialog, im1, im2):
		boxLayout = QtWidgets.QHBoxLayout()

		l1 = QLabel(Dialog)
		pixmap = QPixmap(im1)
		#l1.setPixmap(pixmap)
		#l1.setScaledContents(True);
		l1.setPixmap(pixmap.scaled(200,100,Qt.KeepAspectRatio))
		boxLayout.addWidget(l1)

		l2 = QLabel(Dialog)
		pixmap = QPixmap(im2)
		#l2.setPixmap(pixmap)
		#l2.setScaledContents(True);
		l2.setPixmap(pixmap.scaled(200,100,Qt.KeepAspectRatio))
		boxLayout.addWidget(l2)

		return boxLayout

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = My_Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    ui.Test(Dialog)
    sys.exit(app.exec_())