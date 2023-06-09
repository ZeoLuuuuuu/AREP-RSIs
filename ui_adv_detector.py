# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_adv_detector.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5.Qt import pyqtSignal
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QButtonGroup, QFileDialog
from PyQt5.QtGui import QIcon
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from Adv_Detect.common.util import *
from Adv_Detect.setup_paths import *
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sklearn.metrics import accuracy_score, precision_score, recall_score
from Adv_Detect.fs.datasets.datasets_utils import *
from Adv_Detect.fs.utils.squeeze import *
from Adv_Detect.fs.utils.output import write_to_csv
from Adv_Detect.fs.robustness import evaluate_robustness
from Adv_Detect.fs.detections.base import DetectionEvaluator, evalulate_detection_test, get_tpr_fpr


class Ui_CNN_Detector(object):
    def setupUi(self, CNN_Detector):
        CNN_Detector.setObjectName("CNN_Detector")
        CNN_Detector.resize(1258, 830)
        font = QtGui.QFont()
        font.setPointSize(10)
        CNN_Detector.setFont(font)
        self.label_Dataset = QtWidgets.QLabel(CNN_Detector)
        self.label_Dataset.setGeometry(QtCore.QRect(43, 174, 271, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_Dataset.setFont(font)
        self.label_Dataset.setObjectName("label_Dataset")
        self.pushButton = QtWidgets.QPushButton(CNN_Detector)
        self.pushButton.setGeometry(QtCore.QRect(520, 524, 211, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.training_show = QtWidgets.QLabel(CNN_Detector)
        self.training_show.setGeometry(QtCore.QRect(0, 580, 1251, 246))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.training_show.setFont(font)
        self.training_show.setStyleSheet("background-color: rgb(127, 130, 136);\n"
"")
        self.training_show.setFrameShadow(QtWidgets.QFrame.Raised)
        self.training_show.setLineWidth(3)
        self.training_show.setText("")
        self.training_show.setScaledContents(False)
        self.training_show.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.training_show.setObjectName("training_show")
        self.label_image_show = QtWidgets.QLabel(CNN_Detector)
        self.label_image_show.setGeometry(QtCore.QRect(715, 146, 241, 241))
        self.label_image_show.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_image_show.setText("")
        self.label_image_show.setObjectName("label_image_show")
        self.label = QtWidgets.QLabel(CNN_Detector)
        self.label.setGeometry(QtCore.QRect(691, 393, 291, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.radioButton_UC = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_UC.setGeometry(QtCore.QRect(52, 234, 89, 21))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_UC.setFont(font)
        self.radioButton_UC.setChecked(False)
        self.radioButton_UC.setObjectName("radioButton_UC")
        self.lineEdit_CNN_classification_imagetest_image_address = QtWidgets.QLineEdit(CNN_Detector)
        self.lineEdit_CNN_classification_imagetest_image_address.setGeometry(QtCore.QRect(250, 680, 351, 21))
        self.lineEdit_CNN_classification_imagetest_image_address.setObjectName("lineEdit_CNN_classification_imagetest_image_address")
        self.label_image_show_2 = QtWidgets.QLabel(CNN_Detector)
        self.label_image_show_2.setGeometry(QtCore.QRect(989, 146, 241, 241))
        self.label_image_show_2.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_image_show_2.setText("")
        self.label_image_show_2.setObjectName("label_image_show_2")
        self.label_3 = QtWidgets.QLabel(CNN_Detector)
        self.label_3.setGeometry(QtCore.QRect(961, 393, 291, 31))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.lineEdit_3 = QtWidgets.QLineEdit(CNN_Detector)
        self.lineEdit_3.setGeometry(QtCore.QRect(370, 690, 113, 20))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.radioButton_FGSC = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_FGSC.setGeometry(QtCore.QRect(165, 224, 121, 40))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_FGSC.setFont(font)
        self.radioButton_FGSC.setObjectName("radioButton_FGSC")
        self.radioButton_MSTAR = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_MSTAR.setGeometry(QtCore.QRect(294, 224, 121, 40))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_MSTAR.setFont(font)
        self.radioButton_MSTAR.setObjectName("radioButton_MSTAR")
        self.radioButton_FUSAR = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_FUSAR.setGeometry(QtCore.QRect(418, 225, 131, 40))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_FUSAR.setFont(font)
        self.radioButton_FUSAR.setObjectName("radioButton_FUSAR")
        self.radioButton_AID = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_AID.setGeometry(QtCore.QRect(574, 226, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_AID.setFont(font)
        self.radioButton_AID.setObjectName("radioButton_AID")
        self.label_Attack = QtWidgets.QLabel(CNN_Detector)
        self.label_Attack.setGeometry(QtCore.QRect(43, 287, 221, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_Attack.setFont(font)
        self.label_Attack.setObjectName("label_Attack")
        self.radioButton_BIM = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_BIM.setGeometry(QtCore.QRect(165, 333, 121, 40))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_BIM.setFont(font)
        self.radioButton_BIM.setObjectName("radioButton_BIM")
        self.radioButton_DF = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_DF.setGeometry(QtCore.QRect(290, 333, 121, 40))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_DF.setFont(font)
        self.radioButton_DF.setObjectName("radioButton_DF")
        self.radioButton_PGD = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_PGD.setGeometry(QtCore.QRect(53, 333, 121, 40))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_PGD.setFont(font)
        self.radioButton_PGD.setChecked(False)
        self.radioButton_PGD.setObjectName("radioButton_PGD")
        self.radioButton_CW = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_CW.setGeometry(QtCore.QRect(416, 333, 121, 40))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_CW.setFont(font)
        self.radioButton_CW.setObjectName("radioButton_CW")
        self.radioButton_FGSM = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_FGSM.setGeometry(QtCore.QRect(512, 343, 171, 21))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_FGSM.setFont(font)
        self.radioButton_FGSM.setChecked(False)
        self.radioButton_FGSM.setObjectName("radioButton_FGSM")
        self.lineEdit_CNN_classification_imagetest_model_address = QtWidgets.QLineEdit(CNN_Detector)
        self.lineEdit_CNN_classification_imagetest_model_address.setGeometry(QtCore.QRect(101, 107, 561, 21))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.lineEdit_CNN_classification_imagetest_model_address.setFont(font)
        self.lineEdit_CNN_classification_imagetest_model_address.setStyleSheet("background-color: rgb(243, 243, 243);")
        self.lineEdit_CNN_classification_imagetest_model_address.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_CNN_classification_imagetest_model_address.setObjectName("lineEdit_CNN_classification_imagetest_model_address")
        self.pushButton_CNN_classification_imagetest_model_address = QtWidgets.QPushButton(CNN_Detector)
        self.pushButton_CNN_classification_imagetest_model_address.setGeometry(QtCore.QRect(662, 107, 31, 21))
        self.pushButton_CNN_classification_imagetest_model_address.setStyleSheet("background-color: rgb(98, 98, 98);")
        self.pushButton_CNN_classification_imagetest_model_address.setObjectName("pushButton_CNN_classification_imagetest_model_address")
        self.label_2 = QtWidgets.QLabel(CNN_Detector)
        self.label_2.setGeometry(QtCore.QRect(44, 55, 141, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_Attack_2 = QtWidgets.QLabel(CNN_Detector)
        self.label_Attack_2.setGeometry(QtCore.QRect(44, 396, 241, 41))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(13)
        self.label_Attack_2.setFont(font)
        self.label_Attack_2.setObjectName("label_Attack_2")
        self.radioButton_FS = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_FS.setGeometry(QtCore.QRect(52, 447, 191, 45))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_FS.setFont(font)
        self.radioButton_FS.setChecked(False)
        self.radioButton_FS.setObjectName("radioButton_FS")
        self.radioButton_LID = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_LID.setGeometry(QtCore.QRect(263, 447, 71, 45))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_LID.setFont(font)
        self.radioButton_LID.setChecked(False)
        self.radioButton_LID.setObjectName("radioButton_LID")
        self.radioButton_IM = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_IM.setGeometry(QtCore.QRect(362, 447, 191, 45))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_IM.setFont(font)
        self.radioButton_IM.setChecked(False)
        self.radioButton_IM.setObjectName("radioButton_IM")
        self.radioButton_LM = QtWidgets.QRadioButton(CNN_Detector)
        self.radioButton_LM.setGeometry(QtCore.QRect(516, 447, 211, 45))
        font = QtGui.QFont()
        font.setFamily("Dubai")
        font.setPointSize(12)
        self.radioButton_LM.setFont(font)
        self.radioButton_LM.setChecked(False)
        self.radioButton_LM.setObjectName("radioButton_LM")

        self.lineEdit_3.raise_()
        self.label_Dataset.raise_()
        self.pushButton.raise_()
        self.label_image_show.raise_()
        self.label.raise_()
        self.radioButton_UC.raise_()
        self.lineEdit_CNN_classification_imagetest_image_address.raise_()
        self.training_show.raise_()
        self.label_image_show_2.raise_()
        self.label_3.raise_()
        self.radioButton_FGSC.raise_()
        self.radioButton_MSTAR.raise_()
        self.radioButton_FUSAR.raise_()
        self.radioButton_AID.raise_()
        self.label_Attack.raise_()
        self.radioButton_BIM.raise_()
        self.radioButton_DF.raise_()
        self.radioButton_PGD.raise_()
        self.radioButton_CW.raise_()
        self.radioButton_FGSM.raise_()
        self.lineEdit_CNN_classification_imagetest_model_address.raise_()
        self.pushButton_CNN_classification_imagetest_model_address.raise_()
        self.label_2.raise_()
        self.label_Attack_2.raise_()
        self.radioButton_FS.raise_()
        self.radioButton_LID.raise_()
        self.radioButton_IM.raise_()
        self.radioButton_LM.raise_()
        # self.label_image_show_3.raise_()
        # self.label_4.raise_()

        btngroup1 = QButtonGroup(CNN_Detector)
        btngroup1.addButton(self.radioButton_MSTAR)
        btngroup1.addButton(self.radioButton_FGSC)
        btngroup1.addButton(self.radioButton_AID)
        btngroup1.addButton(self.radioButton_UC)
        btngroup1.addButton(self.radioButton_FUSAR)

        btngroup2 = QButtonGroup(CNN_Detector)
        btngroup2.addButton(self.radioButton_PGD)
        btngroup2.addButton(self.radioButton_BIM)
        btngroup2.addButton(self.radioButton_FGSM)
        btngroup2.addButton(self.radioButton_DF)
        btngroup2.addButton(self.radioButton_CW)

        btngroup3 = QButtonGroup(CNN_Detector)
        btngroup3.addButton(self.radioButton_LM)
        btngroup3.addButton(self.radioButton_IM)
        btngroup3.addButton(self.radioButton_LID)
        btngroup3.addButton(self.radioButton_FS)

        self.attack_method = "PGD"
        self.dataset_name = "UC"
        self.classifier = "FS"

        self.retranslateUi(CNN_Detector)
        QtCore.QMetaObject.connectSlotsByName(CNN_Detector)

    def retranslateUi(self, CNN_Detector):
        _translate = QtCore.QCoreApplication.translate
        CNN_Detector.setWindowTitle(_translate("CNN_Detector", "Reactive Defense"))
        CNN_Detector.setWindowIcon(QIcon("./1.ico"))
        self.label_Dataset.setText(_translate("CNN_Detector", "Datasets:"))
        self.pushButton.setText(_translate("CNN_Detector", "Start"))
        self.label.setText(_translate("CNN_Detector", "Feature Extracted from benign data"))
        self.radioButton_UC.setText(_translate("CNN_Detector", "  UC"))
        self.label_3.setText(_translate("CNN_Detector", "Feature Extracted from adversarial data"))
        self.radioButton_FGSC.setText(_translate("CNN_Detector", " FGSC-23"))
        self.radioButton_MSTAR.setText(_translate("CNN_Detector", " MSTAR"))
        self.radioButton_FUSAR.setText(_translate("CNN_Detector", " FUSAR-Ship"))
        self.radioButton_AID.setText(_translate("CNN_Detector", " AID"))
        self.label_Attack.setText(_translate("CNN_Detector", "Attack Methods："))
        self.radioButton_BIM.setText(_translate("CNN_Detector", "  FGSM"))
        self.radioButton_DF.setText(_translate("CNN_Detector", " DeepFool"))
        self.radioButton_PGD.setText(_translate("CNN_Detector", " PGD"))
        self.radioButton_CW.setText(_translate("CNN_Detector", " C&W"))
        self.radioButton_FGSM.setText(_translate("CNN_Detector", "  HopSkipJump"))
        self.lineEdit_CNN_classification_imagetest_model_address.setText(_translate("CNN_Detector", "pkl_save/resnet18_uc.h5"))
        self.pushButton_CNN_classification_imagetest_model_address.setText(_translate("CNN_Detector", "…"))
        self.label_2.setText(_translate("CNN_Detector", "Model Address:"))
        self.label_Attack_2.setText(_translate("CNN_Detector", "Detection Methods："))
        self.radioButton_FS.setText(_translate("CNN_Detector", "Feature Squeezing"))
        self.radioButton_LID.setText(_translate("CNN_Detector", "LID"))
        self.radioButton_IM.setText(_translate("CNN_Detector", "InputMFS"))
        self.radioButton_LM.setText(_translate("CNN_Detector", "LayerMFS"))
        self.radioButton_PGD.setChecked(True)
        self.radioButton_UC.setChecked(True)
        self.radioButton_FS.setChecked(True)

    def caolianjie(self):
        self.radioButton_UC.toggled.connect(self.dataset_select_set_visible)
        self.radioButton_FUSAR.toggled.connect(self.dataset_select_set_visible)
        self.radioButton_AID.toggled.connect(self.dataset_select_set_visible)
        self.radioButton_FGSC.toggled.connect(self.dataset_select_set_visible)
        self.radioButton_MSTAR.toggled.connect(self.dataset_select_set_visible)

        self.radioButton_PGD.toggled.connect(self.attack_select_set_visible)
        self.radioButton_FGSM.toggled.connect(self.attack_select_set_visible)
        self.radioButton_BIM.toggled.connect(self.attack_select_set_visible)
        self.radioButton_DF.toggled.connect(self.attack_select_set_visible)
        self.radioButton_CW.toggled.connect(self.attack_select_set_visible)

        self.radioButton_FS.toggled.connect(self.classifier_select_set_visible)
        self.radioButton_LID.toggled.connect(self.classifier_select_set_visible)
        self.radioButton_IM.toggled.connect(self.classifier_select_set_visible)
        self.radioButton_LM.toggled.connect(self.classifier_select_set_visible)

        self.pushButton.clicked.connect(self.start)

        self.pushButton.clicked.connect(
            lambda: self.CNN_detect_test(self.attack_method, self.lineEdit_CNN_classification_imagetest_model_address.text(),
                                          self.dataset_name,
                                          self.classifier))

    def dataset_select_set_visible(self):
        if self.radioButton_MSTAR.isChecked():
            self.dataset_name = "MSTAR"
            print(self.dataset_name)
        elif self.radioButton_UC.isChecked():
            self.dataset_name = "UC"
            print(self.dataset_name)
        elif self.radioButton_FGSC.isChecked():
            self.dataset_name = "FGSC-23"
            print(self.dataset_name)
        elif self.radioButton_FUSAR.isChecked():
            self.dataset_name = "FUSAR-Ship"
            print(self.dataset_name)
        elif self.radioButton_AID.isChecked():
            self.dataset_name = "AID"
            print(self.dataset_name)


    def attack_select_set_visible(self):
        if self.radioButton_PGD.isChecked():
            self.attack_method = "PGD"
            print(self.attack_method)
        elif self.radioButton_BIM.isChecked():
            self.attack_method = "FGSM"
            print(self.attack_method)
        elif self.radioButton_FGSM.isChecked():
            self.attack_method = "HopSkipJump"
            print(self.attack_method)
        elif self.radioButton_DF.isChecked():
            self.attack_method = "Deepfool"
            print(self.attack_method)
        elif self.radioButton_CW.isChecked():
            self.attack_method = "CW"
            print(self.attack_method)

    def classifier_select_set_visible(self):
        if self.radioButton_FS.isChecked():
            self.classifier = "FS"
            print(self.classifier)
        elif self.radioButton_LID.isChecked():
            self.classifier = "LID"
            print(self.classifier)
        elif self.radioButton_IM.isChecked():
            self.classifier = "InputMFS"
            print(self.classifier)
        elif self.radioButton_LM.isChecked():
            self.classifier = "LayerMFS"
            print(self.classifier)

    def start(self):
        self.training_show.setText("Start")

    def CNN_detect_test(self, attack_method,model,dataset_name,classifier):
        self.thread_image_test = Thread_CNN_Adv_Detect(attack_method,model,dataset_name,classifier)
        self.thread_image_test.update_datasettest.connect(self.get_CNN_dataset_test_result)
        self.thread_image_test.start()
        self.thread_image_test.exec()

    def choose_CNN_dataset_test_model_dir(self):
        model_dir = QFileDialog.getOpenFileName(None, 'Choose data File', '')
        self.lineEdit_CNN_classification_imagetest_model_address.setText(model_dir[0])

    def get_CNN_dataset_test_result(self,test_result):
        self.training_show.setText(test_result[0])
        self.training_show.repaint()

        plt.plot(test_result[1],test_result[2],label='ROC_of_{}_{}_Detection'.format(test_result[4],test_result[3]))
        plt.plot([0,1],[0,1],color='navy',linewidth=3,linestyle='--')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve of {} of {} Attack Detection with {}'.format(test_result[4],test_result[3],test_result[5]))
        plt.legend(loc="lower right")
        self.openimage_example(self.label_image_show_3,"result_image/roc1.png")

        if self.classifier == "FS":
            squeezed_image = test_result[6]
            original_image = test_result[7]
            squeezed_image[:, :, 0] = squeezed_image[:, :, 0] * 0.19673954 + 0.44979182
            squeezed_image[:, :, 1] = squeezed_image[:, :, 1] * 0.20322968 + 0.48921227
            squeezed_image[:, :, 2] = squeezed_image[:, :, 2] * 0.21901236 + 0.48212156
            original_image[:, :, 0] = original_image[:, :, 0] * 0.19673954 + 0.44979182
            original_image[:, :, 1] = original_image[:, :, 1] * 0.20322968 + 0.48921227
            original_image[:, :, 2] = original_image[:, :, 2] * 0.21901236 + 0.48212156
            plt.imshow(squeezed_image)
            plt.savefig('result_image/squeezed_bit.jpg')
            self.openimage_example(self.label_image_show_2, 'result_image/squeezed_bit.jpg')
            plt.imshow(original_image)
            plt.savefig('result_image/original.jpg')
            self.openimage_example(self.label_image_show, 'result_image/original.jpg')

        elif self.classifier == "InputMFS":
            self.openimage_example(self.label_image_show, "result_image\\feature_show\\InputMFS\\1.jpg")
            self.openimage_example(self.label_image_show_2, "result_image\\feature_show\\InputMFS\\2.jpg")
        elif self.classifier == "LayerMFS":
            self.openimage_example(self.label_image_show, "result_image\\feature_show\LayerMFS\\3.jpg")
            self.openimage_example(self.label_image_show_2, "result_image\\feature_show\LayerMFS\\4.jpg")

    def openimage_example(self,image_show,img_address):
        imgName = img_address
        jpg = QtGui.QPixmap(imgName).scaled(image_show.width(), image_show.height())
        image_show.setPixmap(jpg)

class Thread_CNN_Adv_Detect(QThread):
    update_datasettest = pyqtSignal(dict)
    def __init__(self, attack_method, model_address, dataset_name, classifier):
        super().__init__()
        self.out = ['Adversarial Detection：', '\n', 'Dataset：', dataset_name, '\n', 'Attack:', attack_method,'\n',"Detection Method：",classifier]
        self.attack_method, self.model_address, self.dataset_name, self.classifier = attack_method, model_address.replace(
            '\\', '/'), dataset_name, classifier

    def run(self):
        test_result = self.test()
        self.update_datasettest.emit(test_result)

    def get_distance(self, model, dataset, X1):
        X1_pred = model.predict(X1)
        vals_squeezed = []

        X1_seqeezed_bit = bit_depth_py(X1, 5)
        vals_squeezed.append(model.predict(X1_seqeezed_bit))
        X1_seqeezed_filter_median = median_filter_py(X1, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_median))
        X1_seqeezed_filter_local = non_local_means_color_py(X1, 13, 3, 2)
        vals_squeezed.append(model.predict(X1_seqeezed_filter_local))

        dist_array = []
        for val_squeezed in vals_squeezed:
            dist = np.sum(np.abs(X1_pred - val_squeezed), axis=tuple(range(len(X1_pred.shape))[1:]))
            dist_array.append(dist)

        dist_array = np.array(dist_array)
        return np.max(dist_array, axis=0)

    def detect_test(self, model, dataset, X, threshold):
        distances = self.get_distance(model, dataset, X)
        Y_pred = distances > threshold
        return Y_pred, distances

    def test(self):
        since = time.time()
        if self.classifier == "FS":
            dataset = self.dataset_name
            print('Loading the data and model...')
            # Load the model
            if self.dataset_name == "UC":
                from Adv_Detect.baselineCNN.cnn.cnn_uc import UC10CNN as myModel


            model_class = myModel(mode='load', filename=self.model_address)
            model = model_class.model
            sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

            # Load the dataset
            X_test_all = model_class.X_test_all
            Y_test_all = model_class.Y_test_all
            X_test_all = X_test_all.transpose(0, 2, 3, 1)
            print("X_test shape:", X_test_all.shape)
            # -----------------------------------------------------------------
            # Evaluate the trained model.
            # Refine the normal and adversarial sets to only include samples for
            # which the original version was correctly classified by the model
            print("Evaluating the pre-trained model...")
            Y_pred_all = model.predict(X_test_all)
            Y_test_all = to_categorical(Y_test_all, 21)
            accuracy_all = calculate_accuracy(Y_pred_all, Y_test_all)
            print('Test accuracy on raw legitimate examples %.4f' % (accuracy_all))
            inds_correct = np.where(Y_pred_all.argmax(axis=1) == Y_test_all.argmax(axis=1))[0]
            X_test = X_test_all[inds_correct]
            indx_train = random.sample(range(len(X_test)), int(len(X_test) / 2))
            indx_test = list(set(range(0, len(X_test))) - set(indx_train))
            print("Number of correctly predict images: %s" % (len(inds_correct)))
            x_test = X_test

            file = open("Adv_Detect/Threshold.txt", 'r')
            file_data = file.readline()
            threshold = np.float(file_data)
            print("Threshold:",threshold)
            if self.dataset_name == "UC":
                data_transforms = transforms.Compose([transforms.Resize(224),
                                                      transforms.CenterCrop(224),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.44979182, 0.48921227, 0.48212156),
                                                                           (0.19673954, 0.20322968, 0.21901236))
                                                      ])
            if self.attack_method == "FGSM":
                self.attack_method = "UAP"
            data_dir_adv = 'Adv_Fool_Img\\Adversarial Scene Dataset\\UC_adv_fool\\{}'.format(self.attack_method)
            adv_data_gen = datasets.ImageFolder(data_dir_adv, data_transforms)
            adv_data = torch.utils.data.DataLoader(adv_data_gen,
                                                   batch_size=len(adv_data_gen),
                                                   shuffle=False,
                                                   num_workers=0)
            for data_test in adv_data:
                X_test_adv, Y_test_adv = data_test
            X_test_adv = X_test_adv.numpy()
            Y_test_adv = Y_test_adv.numpy()
            X_test_adv = X_test_adv.transpose(0, 2, 3, 1)

            print("X_test_adv: ", X_test_adv.shape)
            X_test_adv = reduce_precision_py(X_test_adv, 224)

            X_all = np.concatenate([x_test, X_test_adv])
            Y_all = np.concatenate([np.zeros(len(x_test), dtype=bool), np.ones(len(X_test_adv), dtype=bool)])
            print(X_all.shape)
            X1_seqeezed_bit = bit_depth_py(X_all[0, :, :, :], 5)
            print("X1_seqeezed_bit:",X1_seqeezed_bit.shape)
            print("Detecting now")

            Y_all_pred, Y_all_pred_score = self.detect_test(model, dataset, X_all, threshold)

            acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(Y_all, Y_all_pred)

            fprs_all, tprs_all, thresholds_all = roc_curve(Y_all, Y_all_pred_score)
            roc_auc_all = auc(fprs_all, tprs_all)

            time_elapsed = time.time() - since
            print("{:>15} attack, "
                  "AUC: {:.4f}%, Overall Accuracy: {:.4f}%,\n "
                  "Detection Rates:{:7.2f}%, FPR : {:.4f}%".format(self.attack_method, 100 * roc_auc_all, 100 * acc_all,
                                                                   100 * tpr_all, 100 * fpr_all))
            self.out.extend(['\n', 'Time：', time_elapsed, 's', '\n', 'AUC(Area Under Curve)： ', str(100*roc_auc_all), '%',
                             '\n', 'Detection Rate： ', str(100*acc_all),'%', '\n', 'True Positive Rate(TPR): ',str(100*tpr_all),'%', '\n',
                             "False Positive Rate(FPR):", str(100*fpr_all),'%'])
            out = [str(i) for i in self.out]
            out = "".join(out)
            return ({0:out, 1: fpr_all, 2: tpr_all, 3:self.attack_method, 4:self.dataset_name, 5:self.classifier,6:X1_seqeezed_bit,7:X_all[0,:,:,:].squeeze()})



        elif self.classifier == "InputMFS" or self.classifier == "LayerMFS":
            since = time.time()
            print('Loading characteristics...')
            characteristics = np.load('characteristics\\' + self.dataset_name + '_' + self.attack_method + '_' + self.classifier + '.npy',
                allow_pickle=True)
            characteristics_adv = np.load('characteristics\\' + self.dataset_name + '_' + self.attack_method + '_' + self.classifier  + '_adv.npy',
                allow_pickle=True)
            print('characteristics\\' + self.dataset_name + '_' + self.attack_method + '_' + self.classifier  + '.npy')
            print('characteristics\\' + self.dataset_name + '_' + self.attack_method + '_' + self.classifier  + '_adv.npy')
            shape = np.shape(characteristics)
            k = shape[0]
            adv_X_train, adv_X_test, adv_y_train, adv_y_test = train_test_split(characteristics_adv, np.ones(k),
                                                                                test_size=0.2, random_state=42)
            b_X_train, b_X_test, b_y_train, b_y_test = train_test_split(characteristics, np.zeros(k), test_size=0.2,
                                                                        random_state=42)

            X_train = np.concatenate((b_X_train, adv_X_train))
            y_train = np.concatenate((b_y_train, adv_y_train))

            X_test = np.concatenate((b_X_test, adv_X_test))
            y_test = np.concatenate((b_y_test, adv_y_test))

            scaler = MinMaxScaler().fit(X_test)
            X_test = scaler.transform(X_test)

            filename = './Data/'+ 'Detectors_MFS/'+ self.classifier + '_' + self.attack_method + '_' + self.dataset_name + '.sav'
            print(filename)
            with open(filename, 'rb') as f:
                detect = pickle.load(f, encoding='bytes')

            print(" Evaluating Detector...")

            prediction = detect.predict(X_test)
            prediction_pr = detect.predict_proba(X_test)[:, 1]

            benign_rate = 0
            benign_guesses = 0
            ad_guesses = 0
            ad_rate = 0
            ad_wrong = 0
            for i in range(len(prediction)):
                if prediction[i] == 0:
                    benign_guesses += 1
                    if y_test[i] == 0:
                        benign_rate += 1
                else:
                    ad_guesses += 1
                    if y_test[i] == 1:
                        ad_rate += 1
                    else:
                        ad_wrong += 1

            acc_all, tpr_all, fpr_all, tp_all, ap_all, fb_all, an_all = evalulate_detection_test(y_test, prediction)
            AUC = roc_auc_score(y_test, prediction_pr)
            fprs_all, tprs_all, thresholds_all = roc_curve(y_test, prediction_pr)
            acc = (benign_rate + ad_rate) / len(prediction)
            TP = 2 * ad_rate / len(prediction)
            FP = 2 * ad_wrong / len(prediction)
            precision = ad_rate / ad_guesses
            time_elapsed = time.time() - since

            # print('True positive rate/adversarial detetcion rate/recall/sensitivity is ', round(100 * TP, 2))
            # print('True negative rate/normal detetcion rate/selectivity is ', round(100 * TN, 2))
            # print('Precision', round(100 * precision, 1))
            # print('The accuracy is', round(100 * acc, 2))
            # print('The AUC score is', round(100 * roc_auc_score(y_test, prediction_pr), 2))

            self.out.extend(
                ['\n', 'Time：', time_elapsed, 's', '\n', 'AUC(Area Under Curve)： ', str(round(100 * AUC, 4)), '%',
                 '\n', 'Detection Rate： ', str(100 * acc), '%', '\n', 'True Positive Rate(TPR): ',
                 str(100 * TP), '%', '\n', "False Positive Rate(FPR):", str(100 * FP), '%'])
            out = [str(i) for i in self.out]
            out = "".join(out)

            return ({0:out, 1:TP, 2:FP, 3:self.attack_method, 4:self.dataset_name, 5:self.classifier})



if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    interface = QtWidgets.QDialog()
    ui = Ui_CNN_Detector()
    ui.setupUi(interface)
    ui.caolianjie()
    interface.show()
    sys.exit(app.exec_())

