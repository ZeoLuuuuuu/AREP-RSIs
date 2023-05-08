#! python3

from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget
from pATRMainWindow import Ui_HQ16pATR
from Ui_test import Ui_test
from ui_CNN_train import Ui_CNN_train
from ui_CNN_train_fugaishi import Ui_Form_train_fugaishi
from ui_multisource_adv_classification import Ui_multisource_adv_classification
from ui_CNN_jiagu_singlesource import Ui_Form_jiagu_singlesource
from ui_CNN_jiagu_multisource import Ui_Form_jiagu_multisource
from ui_adv_samples_classification import Ui_CNN_data_test
from Ui_adv_jiagu_val import Ui_CNN_dataset_val
from ui_adv_samples_create import Ui_CNN_dataset_adv_creat
from ui_adv_detector import Ui_CNN_Detector
from ui_adv_detector_train import Ui_CNN_Detector_train
from ui_asc_attack import  Ui_asc_attack
from ui_sparse_attack import Ui_Sparse_Attack
from Ui_CNN_Batch_Dataset_test import Ui_CNN_dataset_test
from ui_CNN_jiagu_singlesource import Ensemble

class test_wiget(Ui_test, QWidget):
    def __init__(self):
        super(test_wiget, self).__init__()
        self.setupUi(self)

class CNN_train_widget(Ui_CNN_train,QWidget):
    def __init__(self):
        super(CNN_train_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie(self)

class CNN_train_fugaishi_widget(Ui_Form_train_fugaishi,QWidget):
    def __init__(self):
        super(CNN_train_fugaishi_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_Detector_widget(Ui_CNN_Detector,QWidget):
    def __init__(self):
        super(CNN_Detector_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class Multisource_adv_classification_widget(Ui_multisource_adv_classification,QWidget):
    def __init__(self):
        super(Multisource_adv_classification_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_samples_create_widget(Ui_CNN_dataset_adv_creat,QWidget):
    def __init__(self):
        super(CNN_samples_create_widget,self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_data_test_widget(Ui_CNN_data_test,QWidget):
    def __init__(self):
        super(CNN_data_test_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_adv_detector_train_widget(Ui_CNN_Detector_train,QWidget):
    def __init__(self):
        super(CNN_adv_detector_train_widget,self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_asc_attack_widget(Ui_asc_attack,QWidget):
    def __init__(self):
        super(CNN_asc_attack_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_jiagu_singlesource_widget(Ui_Form_jiagu_singlesource,QWidget):
    def __init__(self):
        super(CNN_jiagu_singlesource_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_jiagu_multisource_widget(Ui_Form_jiagu_multisource,QWidget):
    def __init__(self):
        super(CNN_jiagu_multisource_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_adv_dataset_test_widget(Ui_CNN_dataset_test,QWidget):
    def __init__(self):
        super(CNN_adv_dataset_test_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_adv_dataset_val_widget(Ui_CNN_dataset_val,QWidget):
    def __init__(self):
        super(CNN_adv_dataset_val_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class CNN_sparse_adv_widget(Ui_Sparse_Attack,QWidget):
    def __init__(self):
        super(CNN_sparse_adv_widget, self).__init__()
        self.setupUi(self)
        self.caolianjie()

class MyMainWindow(QMainWindow, Ui_HQ16pATR):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        self.caolianjie()

        # instantiation window

        self.win2 = test_wiget()

        # signal and slot
        self.init_signal_slot()

        self.CNN_train_win = CNN_train_widget()
        self.CNN_train_fugaishi_win = CNN_train_fugaishi_widget()
        self.CNN_Detector_win = CNN_Detector_widget()
        self.Multisource_adv_classification_widget_win = Multisource_adv_classification_widget()
        # self.CNN_image_test_win = CNN_image_test_widget()
        self.CNN_data_test_win = CNN_data_test_widget()
        self.CNN_data_val_win = CNN_adv_dataset_val_widget()
        self.CNN_adv_Detector_Train_win = CNN_adv_detector_train_widget()
        self.CNN_jiagu_singlesource_win = CNN_jiagu_singlesource_widget()
        self.CNN_jiagu_multisource_win = CNN_jiagu_multisource_widget()
        self.CNN_adv_dataset_test_win = CNN_adv_dataset_test_widget()
        self.CNN_sample_create_win = CNN_samples_create_widget()
        self.CNN_sparse_attack_win = CNN_sparse_adv_widget()
        self.CNN_asc_attack_win = CNN_asc_attack_widget()

    def init_signal_slot(self):

        # self.actionManualAnnotation.triggered.connect(lambda: self.change_win(self.win2))
        self.pushButtonSEQ.clicked.connect(lambda: self.CNN_sample_create_win.show())
        self.pushButtonVID.clicked.connect(lambda: self.CNN_Detector_win.show())
        self.pushButtonVGA.clicked.connect(lambda: self.CNN_jiagu_singlesource_win.show())

        # self.actionSVMtest.triggered.connect(lambda: self.conventional_image_test_win.show())
        # self.actionSVMtest.triggered.connect(lambda: self.textBrowserIMGProperty.setText('传统方法单张图像测试'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        # self.actionSVMAcc.triggered.connect(lambda: self.conventional_dataset_test_win.show())
        # self.actionSVMAcc.triggered.connect(lambda: self.textBrowserIMGProperty.setText('传统方法数据集测试'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actionCNNtrain.triggered.connect(lambda: self.CNN_train_win.show())
        self.actionCNNtrain.triggered.connect(lambda: self.textBrowserIMGProperty.setText('定制式智能模型训练'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actionmodel.triggered.connect(lambda: self.CNN_Detector_win.show())
        self.actionmodel.triggered.connect(lambda: self.textBrowserIMGProperty.setText('深度特征检测'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actiondetector.triggered.connect(lambda: self.CNN_adv_Detector_Train_win.show())
        self.actiondetector.triggered.connect(lambda: self.textBrowserIMGProperty.setText('深度特征检测器训练'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actionCNNtrain_fugaishi.triggered.connect(lambda: self.CNN_train_fugaishi_win.show())
        self.actionCNNtrain_fugaishi.triggered.connect(lambda: self.textBrowserIMGProperty.setText('覆盖式智能模型训练'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actionMultisource_adv_classification.triggered.connect(lambda: self.Multisource_adv_classification_widget_win.show())
        self.actionMultisource_adv_classification.triggered.connect(lambda: self.textBrowserIMGProperty.setText('多源对抗样本分类'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actiondefense_singlesource.triggered.connect(lambda: self.CNN_jiagu_singlesource_win.show())
        self.actiondefense_singlesource.triggered.connect(lambda: self.textBrowserIMGProperty.setText('单源特征强化加固'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actiondefense_valsource.triggered.connect(lambda: self.CNN_data_val_win.show())
        self.actiondefense_valsource.triggered.connect(lambda: self.textBrowserIMGProperty.setText('主动加固欺骗验证' + '\n' + self.textBrowserIMGProperty.toPlainText()))
        self.actiondefense_multisource.triggered.connect(lambda: self.CNN_jiagu_multisource_win.show())
        self.actiondefense_multisource.triggered.connect(lambda: self.textBrowserIMGProperty.setText('多源特征融合加固'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actionadvtest.triggered.connect(lambda: self.CNN_adv_dataset_test_win.show())
        self.actionadvtest.triggered.connect(lambda: self.textBrowserIMGProperty.setText('Batch批测试'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actionadversarialattack.triggered.connect(lambda: self.CNN_sample_create_win.show())
        self.actionadversarialattack.triggered.connect(lambda: self.textBrowserIMGProperty.setText('对抗样本影像制作'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actionsparseattack.triggered.connect(lambda: self.CNN_sparse_attack_win.show())
        self.actionsparseattack.triggered.connect(lambda: self.textBrowserIMGProperty.setText('稀疏攻击'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actionstandardtest.triggered.connect(lambda: self.CNN_data_test_win.show())
        self.actionstandardtest.triggered.connect(lambda: self.textBrowserIMGProperty.setText('单张遥感影像测试'+'\n'+self.textBrowserIMGProperty.toPlainText()))
        self.actionblackattack.triggered.connect(lambda: self.CNN_asc_attack_win.show())
        self.actionblackattack.triggered.connect(lambda: self.textBrowserIMGProperty.setText('ASC物理特性攻击'+'\n'+self.textBrowserIMGProperty.toPlainText()))

        self.actionIMGFolder.triggered.connect(self.choose_data_dir_pichuli_class)
    def change_win(self, widget_obj):
        self.win2.show()

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    ui = MyMainWindow()
    ui.show()
    sys.exit(app.exec_())

# input("please input any key to exit!")
