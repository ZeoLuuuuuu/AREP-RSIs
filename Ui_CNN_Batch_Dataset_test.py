from PyQt5.QtGui import QIcon
from PyQt5.Qt import QApplication, QWidget, QPushButton,QThread,QMutex
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QButtonGroup
import numpy as np
import time
import torch
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.switch_backend('TKAgg')
from tqdm import trange,tqdm
import matplotlib as mpl
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

class Ui_CNN_dataset_test(object):
    def setupUi(self, CNN_dataset_test):
        CNN_dataset_test.setObjectName("CNN_dataset_test")
        CNN_dataset_test.setStyleSheet("background-color: rgb(246, 250, 255);")
        CNN_dataset_test.resize(1755, 745)
        font = QtGui.QFont()
        font.setPointSize(10)
        CNN_dataset_test.setFont(font)
        self.label_2 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_2.setGeometry(QtCore.QRect(30, 20, 221, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(CNN_dataset_test)
        self.pushButton.setGeometry(QtCore.QRect(260, 390, 161, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_21 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_21.setGeometry(QtCore.QRect(30, 138, 261, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.label_image_show = QtWidgets.QLabel(CNN_dataset_test)
        self.label_image_show.setGeometry(QtCore.QRect(92, 460, 231, 231))
        self.label_image_show.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_image_show.setText("")
        self.label_image_show.setObjectName("label_image_show")
        self.label = QtWidgets.QLabel(CNN_dataset_test)
        self.label.setGeometry(QtCore.QRect(106, 699, 199, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.label_image_show_2 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_image_show_2.setGeometry(QtCore.QRect(720, 10, 511, 461))
        self.label_image_show_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_image_show_2.setText("")
        self.label_image_show_2.setObjectName("label_image_show_2")
        self.label_3 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_3.setGeometry(QtCore.QRect(750, 480, 444, 25))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.lineEdit_CNN_classification_imagetest_model_address = QtWidgets.QLineEdit(CNN_dataset_test)
        self.lineEdit_CNN_classification_imagetest_model_address.setGeometry(QtCore.QRect(80, 81, 531, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.lineEdit_CNN_classification_imagetest_model_address.setFont(font)
        self.lineEdit_CNN_classification_imagetest_model_address.setStyleSheet("background-color: rgb(243, 243, 243);")
        self.lineEdit_CNN_classification_imagetest_model_address.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_CNN_classification_imagetest_model_address.setObjectName("lineEdit_CNN_classification_imagetest_model_address")
        self.pushButton_CNN_classification_imagetest_model_address = QtWidgets.QPushButton(CNN_dataset_test)
        self.pushButton_CNN_classification_imagetest_model_address.setGeometry(QtCore.QRect(611, 81, 31, 31))
        self.pushButton_CNN_classification_imagetest_model_address.setStyleSheet("background-color: rgb(98, 98, 98);")
        self.pushButton_CNN_classification_imagetest_model_address.setObjectName("pushButton_CNN_classification_imagetest_model_address")
        self.radioButton_2 = QtWidgets.QRadioButton(CNN_dataset_test)
        self.radioButton_2.setGeometry(QtCore.QRect(280, 210, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_2.setFont(font)
        self.radioButton_2.setObjectName("radioButton_2")
        self.radioButton_4 = QtWidgets.QRadioButton(CNN_dataset_test)
        self.radioButton_4.setGeometry(QtCore.QRect(70, 210, 89, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_4.setFont(font)
        self.radioButton_4.setChecked(True)
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_3 = QtWidgets.QRadioButton(CNN_dataset_test)
        self.radioButton_3.setGeometry(QtCore.QRect(160, 210, 89, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton = QtWidgets.QRadioButton(CNN_dataset_test)
        self.radioButton.setGeometry(QtCore.QRect(404, 210, 151, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton.setFont(font)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_5 = QtWidgets.QRadioButton(CNN_dataset_test)
        self.radioButton_5.setGeometry(QtCore.QRect(557, 210, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_5.setFont(font)
        self.radioButton_5.setObjectName("radioButton_5")
        self.frame = QtWidgets.QFrame(CNN_dataset_test)
        self.frame.setGeometry(QtCore.QRect(38, 317, 651, 61))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.radioButton_8 = QtWidgets.QRadioButton(self.frame)
        self.radioButton_8.setGeometry(QtCore.QRect(121, 3, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_8.setFont(font)
        self.radioButton_8.setObjectName("radioButton_8")
        self.radioButton_6 = QtWidgets.QRadioButton(self.frame)
        self.radioButton_6.setGeometry(QtCore.QRect(28, 9, 89, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_6.setFont(font)
        self.radioButton_6.setObjectName("radioButton_6")
        self.radioButton_7 = QtWidgets.QRadioButton(self.frame)
        self.radioButton_7.setGeometry(QtCore.QRect(242, 11, 89, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_7.setFont(font)
        self.radioButton_7.setChecked(True)
        self.radioButton_7.setObjectName("radioButton_7")
        self.radioButton_9 = QtWidgets.QRadioButton(self.frame)
        self.radioButton_9.setGeometry(QtCore.QRect(362, 9, 135, 23))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_9.setFont(font)
        self.radioButton_9.setObjectName("radioButton_9")
        self.radioButton_10 = QtWidgets.QRadioButton(self.frame)
        self.radioButton_10.setGeometry(QtCore.QRect(513, 11, 89, 19))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_10.setFont(font)
        self.radioButton_10.setChecked(False)
        self.radioButton_10.setObjectName("radioButton_10")
        self.label_4 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_4.setGeometry(QtCore.QRect(34, 261, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_image_show_3 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_image_show_3.setGeometry(QtCore.QRect(390, 460, 231, 231))
        self.label_image_show_3.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_image_show_3.setText("")
        self.label_image_show_3.setObjectName("label_image_show_3")
        self.label_5 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_5.setGeometry(QtCore.QRect(368, 699, 271, 20))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_image_show_4 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_image_show_4.setGeometry(QtCore.QRect(1237, 11, 511, 461))
        self.label_image_show_4.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_image_show_4.setText("")
        self.label_image_show_4.setObjectName("label_image_show_4")
        self.label_6 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_6.setGeometry(QtCore.QRect(1250, 480, 492, 25))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(CNN_dataset_test)
        self.label_7.setGeometry(QtCore.QRect(730, 530, 1021, 202))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.label_7.setFont(font)
        self.label_7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.label_7.setObjectName("label_7")
        self.retranslateUi(CNN_dataset_test)
        QtCore.QMetaObject.connectSlotsByName(CNN_dataset_test)

        btngroup1 = QButtonGroup(CNN_dataset_test)
        btngroup1.addButton(self.radioButton)
        btngroup1.addButton(self.radioButton_2)
        btngroup1.addButton(self.radioButton_3)
        btngroup1.addButton(self.radioButton_4)
        btngroup1.addButton(self.radioButton_5)

        btngroup2 = QButtonGroup(CNN_dataset_test)
        btngroup2.addButton(self.radioButton_6)
        btngroup2.addButton(self.radioButton_9)
        btngroup2.addButton(self.radioButton_7)
        btngroup2.addButton(self.radioButton_8)
        btngroup2.addButton(self.radioButton_10)

        self.dataset_name = 'uc'
        self.clean_img = "Data\Scene\\UCMerced_LandUse\\val"
        self.adv_img = r"D:\Data\Adv_Fool_Img\Adversarial Scene Dataset\UC_adv_fool\PGD"


    def retranslateUi(self, CNN_dataset_test):
        _translate = QtCore.QCoreApplication.translate
        CNN_dataset_test.setWindowTitle(_translate("CNN_dataset_test", "Recognition Test of Remote Sensing Images in Batch"))
        CNN_dataset_test.setWindowIcon(QIcon("./1.ico"))
        self.label_2.setText(_translate("CNN_dataset_test", "Model Address:"))
        self.pushButton.setText(_translate("CNN_dataset_test", "Start to Test"))
        self.label_21.setText(_translate("CNN_dataset_test", "Select the Dataset："))
        self.label.setText(_translate("CNN_dataset_test", "Example of Benign Data"))
        self.label_3.setText(_translate("CNN_dataset_test", "Classification Confusion Matrix of Benign Data"))
        self.lineEdit_CNN_classification_imagetest_model_address.setText(_translate("CNN_dataset_test", "pkl_save/uc_resnet18.pkl"))
        self.pushButton_CNN_classification_imagetest_model_address.setText(_translate("CNN_dataset_test", "…"))
        self.radioButton_2.setText(_translate("CNN_dataset_test", " FGSC-23"))
        self.radioButton_4.setText(_translate("CNN_dataset_test", "UC"))
        self.radioButton_3.setText(_translate("CNN_dataset_test", "MSTAR"))
        self.radioButton.setText(_translate("CNN_dataset_test", " FUSAR-Ship"))
        self.radioButton_5.setText(_translate("CNN_dataset_test", "Sorted-Cars"))
        self.radioButton_8.setText(_translate("CNN_dataset_test", " Deepfool"))
        self.radioButton_6.setText(_translate("CNN_dataset_test", " PGD"))
        self.radioButton_7.setText(_translate("CNN_dataset_test", "  C&W"))
        self.radioButton_9.setText(_translate("CNN_dataset_test", "HopSkipJump"))
        self.radioButton_10.setText(_translate("CNN_dataset_test", " UAP"))
        self.label_4.setText(_translate("CNN_dataset_test", "Adversarial Attacks："))
        self.label_5.setText(_translate("CNN_dataset_test", "Example of Adversarial Data"))
        self.label_6.setText(_translate("CNN_dataset_test", "Classification Confusion Matrix of Adversarial Data"))
        self.label_7.setText(_translate("CNN_dataset_test", "Output Results:"))
        self.radioButton_4.setChecked(True)
        self.radioButton_6.setChecked(True)
        self.openimage_ori("Data\Scene\\UCMerced_LandUse\\val")
        self.openimage_adv("Data\Scene\\UCMerced_LandUse\对抗欺骗样本_New\PGD")

    def caolianjie(self):
        # self.pushButton_CNN_classification_imagetest_image_address.clicked.connect(self.choose_CNN_dataset_test_data_dir)
        self.radioButton.toggled.connect(self.dataset_attack_select_set_visible)
        self.radioButton_2.toggled.connect(self.dataset_attack_select_set_visible)
        self.radioButton_3.toggled.connect(self.dataset_attack_select_set_visible)
        self.radioButton_4.toggled.connect(self.dataset_attack_select_set_visible)
        self.radioButton_5.toggled.connect(self.dataset_attack_select_set_visible)
        self.radioButton_6.toggled.connect(self.dataset_attack_select_set_visible)
        self.radioButton_7.toggled.connect(self.dataset_attack_select_set_visible)
        self.radioButton_8.toggled.connect(self.dataset_attack_select_set_visible)
        self.radioButton_9.toggled.connect(self.dataset_attack_select_set_visible)
        self.radioButton_10.toggled.connect(self.dataset_attack_select_set_visible)

        self.pushButton_CNN_classification_imagetest_model_address.clicked.connect(
            self.choose_CNN_dataset_test_model_dir)
        self.pushButton.clicked.connect(self.out_setting)
        self.pushButton.clicked.connect(
            lambda: self.CNN_dataset_test(self.clean_img, self.adv_img,self.dataset_name,
                                          self.lineEdit_CNN_classification_imagetest_model_address.text(),
                                          self.attack_name))

    def out_setting(self):
        if self.radioButton.isChecked():
            self.dataset_name = 'fusarship'
        elif self.radioButton_2.isChecked():
            self.dataset_name = 'fgsc23'
        elif self.radioButton_3.isChecked():
            self.dataset_name = 'mstar'
        elif self.radioButton_4.isChecked():
            self.dataset_name = 'uc'
        elif self.radioButton_5.isChecked():
            self.dataset_name = 'sortedcars'
        if self.radioButton_6.isChecked():
            self.attack_name = "PGD"
        elif self.radioButton_7.isChecked():
            self.attack_name = "CW"
        elif self.radioButton_9.isChecked():
            self.attack_name = "HopSkipJump"
        elif self.radioButton_8.isChecked():
            self.attack_name = "Deepfool"
        elif self.radioButton_10.isChecked():
            self.attack_name = "UAP"

    def dataset_attack_select_set_visible(self):
        if self.radioButton_6.isChecked():
            if self.radioButton.isChecked():
                self.clean_img = "Data\Targets\FUSAR_Ship_jpg\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FUSAR_adv_fool\PGD"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_2.isChecked():
                # data_dir = "Data\目标\\FGSC-23\\val_PGD.npy"
                self.clean_img = "Data\Targets\FGSC-23\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FGSC_adv_fool\PGD"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_3.isChecked():
                # data_dir = "Data\目标\\MSTAR-10\\val_PGD.npy"
                self.clean_img = "Data\Targets\MSTAR-10\\val"
                self.adv_img = r'D:\Data\Targets\MSTAR-10\Mstar_对抗欺骗样本\PGD_L2_0.5'
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_4.isChecked():
                # data_dir = "Data\场景\\UCMerced_LandUse\\val_PGD.npy"
                self.clean_img = "Data\Scene\\UCMerced_LandUse\\All"
                self.adv_img = r"D:\Data\Adv_Fool_Img\Adversarial Scene Dataset\UC_adv_fool\PGD"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_5.isChecked():
                data_dir = "Data\Targets\\Sorted_Cars\\val_PGD.npy"

        elif self.radioButton_7.isChecked():
            self.attack = "CW"
            if self.radioButton.isChecked():
                # data_dir = "Data\目标\\FUSAR_Ship\\val_CW.npy"
                self.clean_img = "Data\Targets\FUSAR_Ship_jpg\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FUSAR_adv_fool\CW"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_2.isChecked():
                self.clean_img = "Data\Targets\FGSC-23\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FGSC_adv_fool\CW"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_3.isChecked():
                self.clean_img = "Data\Targets\MSTAR-10\\val"
                self.adv_img = r'D:\Data\Targets\MSTAR-10\Mstar_对抗欺骗样本\CW8'
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_4.isChecked():
                self.clean_img = "Data\Scene\\UCMerced_LandUse\\All"
                self.adv_img = r"D:\Data\Scene\UCMerced_LandUse\对抗欺骗样本\UC_CW_Adv"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_11.isChecked():
                data_dir = "Data\Targets\\Sorted_Cars\\val_CW.npy"

        elif self.radioButton_9.isChecked():
            self.attack = "HopSkipJump"
            if self.radioButton.isChecked():
                self.clean_img = "Data\Targets\FUSAR_Ship_jpg\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FUSAR_adv_fool\HopSkipJump"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_2.isChecked():
                self.clean_img = "Data\Targets\FGSC-23\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FGSC_adv_fool\HopSkipJump"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_3.isChecked():
                self.clean_img = "Data\Targets\MSTAR-10\\val"
                self.adv_img = r'D:\Data\Targets\MSTAR-10\Mstar_对抗欺骗样本\HopSkipJump8'
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_4.isChecked():
                self.clean_img = "Data\Scene\\UCMerced_LandUse\\All"
                self.adv_img = r"D:\Data\Adv_Fool_Img\Adversarial Scene Dataset\UC_adv_fool\HopSkipJump"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_11.isChecked():
                data_dir = "Data\Targets\\Sorted_Cars\\val_HopSkipJump.npy"

        elif self.radioButton_8.isChecked():
            self.attack = "Deepfool"
            if self.radioButton.isChecked():
                self.clean_img = "Data\Targets\FUSAR_Ship_jpg\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FUSAR_adv_fool\Deepfool"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_2.isChecked():
                self.clean_img = "Data\Targets\FGSC-23\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FGSC_adv_fool\Deepfool"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_3.isChecked():
                self.clean_img = "Data\Targets\MSTAR-10\\val"
                self.adv_img = r'D:\Data\Targets\MSTAR-10\Mstar_对抗欺骗样本\Deepfool'
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_4.isChecked():
                self.clean_img = "Data\Scene\\UCMerced_LandUse\\All"
                self.adv_img = r"D:\Data\Scene\UCMerced_LandUse\对抗欺骗样本\UC_Deepfool_Adv"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_11.isChecked():
                data_dir = "Data\Targets\\Sorted_Cars\\val_Deepfool.npy"

        elif self.radioButton_10.isChecked():
            self.attack = "UAP"     # (except UC, all is from CW)
            if self.radioButton.isChecked():
                self.clean_img = "Data\Targets\FUSAR_Ship_jpg\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FUSAR_adv_fool\FGSM"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_2.isChecked():
                self.clean_img = "Data\Targets\FGSC-23\\All"
                self.adv_img = r"D:\Data\adv_fool_pic\FGSC_adv_fool\FGSM"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_3.isChecked():
                self.clean_img = "Data\Targets\MSTAR-10\\val"
                self.adv_img =  r'D:\Data\adv_fool_pic\MSTAR_adv_fool\FGSM'
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_4.isChecked():
                self.clean_img = "Data\Scene\\UCMerced_LandUse\\All"
                self.adv_img =  r"D:\Data\Adv_Fool_Img\Adversarial Scene Dataset\UC_adv_fool\FGSM"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)
            elif self.radioButton_5.isChecked():
                self.clean_img = "Data/Targets/Sorted_Cars/All"
                self.adv_img = "Data/Targets/Sorted_Cars/对抗欺骗样本/FGSM"
                self.openimage_ori(self.clean_img)
                self.openimage_adv(self.adv_img)

    def CNN_dataset_test(self,clean_img_dir,adv_img_dir,dataset_name,model_address,attack_method):
        self.thread_image_test = Thread_CNN_dataset_test(clean_img_dir,adv_img_dir,dataset_name,model_address,attack_method)
        self.thread_image_test.update_datasettest.connect(self.get_CNN_dataset_test_result)
        self.thread_image_test.update_datasettest.connect(self.draw_confusion_matrix_clean)
        self.thread_image_test.update_datasettest.connect(self.draw_confusion_matrix_adv)
        self.thread_image_test.start()
        self.thread_image_test.exec()

    def get_CNN_dataset_test_result(self, test_result):
        self.label_7.setText(test_result[0])
        self.label_7.repaint()

    def draw_confusion_matrix_clean(self,test_result):
        test_cm_clean = test_result[1]
        sub_dir = test_result[3]
        plt.imshow(test_cm_clean, cmap=plt.cm.Reds)
        indices = range(len(test_cm_clean))
        plt.xticks(indices, sub_dir, rotation=270)
        plt.yticks(indices, sub_dir)
        # plt.colorbar()

        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('原始样本混淆矩阵')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        for first_index in range(len(test_cm_clean)):  # 第几行
            for second_index in range(len(test_cm_clean[first_index])):  # 第几列
                plt.text(first_index, second_index, test_cm_clean[second_index][first_index], verticalalignment='center',
                         horizontalalignment='center')
        plt.savefig('result_image/CNN_cm.jpg')
        plt.close('all')
        self.openimage_cm_clean('result_image/CNN_cm.jpg')

    def draw_confusion_matrix_adv(self,test_result):
        test_cm_adv = test_result[2]
        sub_dir = test_result[3]
        plt.imshow(test_cm_adv, cmap=plt.cm.Reds)
        indices = range(len(test_cm_adv))
        plt.xticks(indices, sub_dir, rotation=270)
        plt.yticks(indices, sub_dir)
        # plt.colorbar()

        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('对抗样本混淆矩阵')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        for first_index in range(len(test_cm_adv)):  # 第几行
            for second_index in range(len(test_cm_adv[first_index])):  # 第几列
                plt.text(first_index, second_index, test_cm_adv[second_index][first_index],
                         verticalalignment='center',
                         horizontalalignment='center')
        plt.savefig('result_image/CNN_cm_adv.jpg')
        plt.close('all')
        self.openimage_cm_adv('result_image/CNN_cm_adv.jpg')

    def choose_CNN_dataset_test_model_dir(self):
        model_dir = QFileDialog.getOpenFileName(None, 'Choose data File', '')
        self.lineEdit_CNN_classification_imagetest_model_address.setText(model_dir[0])

    def openimage_cm_clean(self, imgName_1):
        imgName_1 = imgName_1.replace('\\', '/')
        jpg = QtGui.QPixmap(imgName_1).scaled(self.label_image_show_2.width(), self.label_image_show_2.height())
        self.label_image_show_2.setPixmap(jpg)

    def openimage_cm_adv(self, imgName_1):
        imgName_1 = imgName_1.replace('\\', '/')
        jpg = QtGui.QPixmap(imgName_1).scaled(self.label_image_show_4.width(), self.label_image_show_4.height())
        self.label_image_show_4.setPixmap(jpg)

    def openimage_ori(self, dataset_Name):
        dataset_Name = dataset_Name.replace('\\', '/')
        list = os.listdir(dataset_Name)
        name = os.listdir(dataset_Name + '/' + list[0])
        imgName = dataset_Name + '/' + list[0] + '/' + name[0]
        jpg = QtGui.QPixmap(imgName).scaled(self.label_image_show.width(), self.label_image_show.height())
        self.label_image_show.setPixmap(jpg)

    def openimage_adv(self, dataset_Name):
        dataset_Name = dataset_Name.replace('\\', '/')
        list = os.listdir(dataset_Name)
        name = os.listdir(dataset_Name + '/' + list[0])
        imgName = dataset_Name + '/' + list[0] + '/' + name[0]
        jpg = QtGui.QPixmap(imgName).scaled(self.label_image_show_3.width(), self.label_image_show_3.height())
        self.label_image_show_3.setPixmap(jpg)

class Thread_CNN_dataset_test(QThread):
    update_datasettest = pyqtSignal(dict)


    def __init__(self, clean_img_dir,adv_img_dir,dataset_name,model_address,attack_method):
        super().__init__()
        self.out = ['Done', '\n', 'Dataset Address of Benign Data：', clean_img_dir,'\n',
                    'Dataset Address of Adversarial Data：', adv_img_dir,'\n','Model Address：',model_address,'\n','Attack Methods: ', attack_method ]
        # if channel_num=='3':
        #     self.out.extend(['mean0,mean1,mean2:',mean0,',',mean1,',',mean2,' std0,std1,std2:',std0,',',std1,',',std2, '\n'])
        # elif channel_num=='1':
        #     self.out.extend(['mean0:',mean0,'   std0:',std0])
        self.clean_img_address, self.adv_img_address, self.model_address, self.dataset_name=\
        clean_img_dir.replace('\\', '/'), adv_img_dir.replace('\\', '/'),model_address.replace('\\', '/'),dataset_name

    def run(self):
        test_result = self.test()
        self.update_datasettest.emit(test_result)
    def test(self):

        sub_dir = os.listdir(self.clean_img_address)
        since = time.time()
        if self.dataset_name == 'uc':
            data_transforms = transforms.Compose([transforms.Resize(224),
                                                  transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.44979182, 0.48921227, 0.48212156), (0.19673954, 0.20322968, 0.21901236))])

        elif self.dataset_name == 'mstar':
            data_transforms =transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.184,), (0.119,))])
        elif self.dataset_name == 'fusarship':
            data_transforms = transforms.Compose([
                transforms.Grayscale(1),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ])
        elif self.dataset_name == 'fgsc23':
            data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.352978, 0.373653, 0.359517), (0.4979, 0.4846, 0.4829))])
        elif self.dataset_name == 'sortedcars':
            data_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.500798, 0.487246, 0.446257), (0.3841, 0.3740,  0.3781))])

        image_datasets_clean = datasets.ImageFolder(self.clean_img_address,data_transforms)
        image_datasets_adv = datasets.ImageFolder(self.adv_img_address,data_transforms)
        dataloders_clean = torch.utils.data.DataLoader(image_datasets_clean,
                                                     batch_size=16,
                                                     shuffle=False,
                                                     num_workers=0)
        dataloders_adv = torch.utils.data.DataLoader(image_datasets_adv,
                                                     batch_size=16,
                                                     shuffle=False,
                                                     num_workers=0)
        dataset_sizes_clean = len(image_datasets_clean)
        dataset_sizes_adv = len(image_datasets_adv)
        model = torch.load(self.model_address)
        use_gpu = torch.cuda.is_available()

        model.eval()
        running_corrects_clean = 0.0
        running_corrects_adv = 0.0

        # Iterate over data.
        preds_all_clean = np.zeros(dataset_sizes_clean)
        labels_all_clean = np.zeros(dataset_sizes_clean)
        i = 0
        for data in tqdm(dataloders_clean):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            # forward
            outputs = model(inputs)
            # print (outputs.shape)
            _, preds = torch.max(outputs.data, 1)
            # statistics
            running_corrects_clean += torch.sum(preds == labels.data).to(torch.float32)
            preds_all_clean[i*preds.shape[0]:(i+1)*preds.shape[0]] = preds.cpu().detach().numpy()
            labels_all_clean[i * preds.shape[0]: (i + 1) * preds.shape[0]] = labels.cpu().detach().numpy()
            i+=1
        print(preds_all_clean)
        print(labels_all_clean)
        test_cm = confusion_matrix(labels_all_clean, preds_all_clean)
        epoch_acc_clean = running_corrects_clean / dataset_sizes_clean


        # Iterate over data.
        preds_all_adv = np.zeros(dataset_sizes_adv)
        labels_all_adv = np.zeros(dataset_sizes_adv)
        i = 0
        for data in tqdm(dataloders_adv):
            # get the inputs
            inputs, labels = data
            # wrap them in Variable
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            # forward
            outputs = model(inputs)
            # print (outputs.shape)
            _, preds = torch.max(outputs.data, 1)
            # statistics
            running_corrects_adv += torch.sum(preds == labels.data).to(torch.float32)
            preds_all_adv[i*preds.shape[0]:(i+1)*preds.shape[0]] = preds.cpu().detach().numpy()
            labels_all_adv[i * preds.shape[0]: (i + 1) * preds.shape[0]] = labels.cpu().detach().numpy()
            i+=1
        print(preds_all_adv)
        print(labels_all_adv)
        test_cm_adv = confusion_matrix(labels_all_adv, preds_all_adv)
        epoch_acc_adv = running_corrects_adv / dataset_sizes_adv
        jiagu_acc = ((dataset_sizes_clean/(dataset_sizes_clean + dataset_sizes_adv))*epoch_acc_clean) + ((dataset_sizes_adv/(dataset_sizes_clean + dataset_sizes_adv))*epoch_acc_adv)
        time_elapsed = time.time() - since
        self.out.extend(['\n','Time used：',time_elapsed,'s','\n','Classification accuracy on benign data： ',float(epoch_acc_clean*100),'%',
                         '\n','Classification accuracy on adversarial data： ',float(epoch_acc_adv*100),'%','\n',' Average accuracy:',float(jiagu_acc*100),'%'])
        out = [str(i) for i in self.out]
        out = "".join(out)
        test_out_log = out.split('\n')
        print(test_out_log)
        # write_excel_xlsx('log.xlsx', test_out_log)
        return ({0:out,1:test_cm, 2:test_cm_adv, 3:sub_dir})

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    train = QtWidgets.QDialog()
    ui = Ui_CNN_dataset_test()
    ui.setupUi(train)
    ui.caolianjie()
    train.show()
    sys.exit(app.exec_())