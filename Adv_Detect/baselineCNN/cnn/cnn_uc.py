from __future__ import division, absolute_import, print_function
import torch
from torchvision import datasets, models, transforms
import pathlib
import tensorflow as tf
from Adv_Detect.common.util import *
from Adv_Detect.setup_paths import *

class UC10CNN:
    def __init__(self, mode='train', filename="resnet18_uc.h5", norm_mean=False, epochs=100, batch_size=16):
        self.mode = mode #train or load
        self.filename = filename
        self.norm_mean = norm_mean
        self.epochs = epochs
        self.batch_size = batch_size

        #====================== load data ========================

        self.num_classes = 21
        self.filters = 64
        self.width = 224
        self.channals = 3

        self.data_dir_all = 'Data/Scene/UCMerced_LandUse/All'
        self.data_dir = pathlib.Path(self.data_dir_all)
        self.CLASS_NAMES = np.array([item.name for item in self.data_dir.glob('*') if item.name != "LICENSE.txt"])
        print(self.CLASS_NAMES)

        self.image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        self.data_transforms = transforms.Compose([transforms.Resize(224),
                                              transforms.CenterCrop(224),
                                              transforms.ToTensor(),
                                              ])

        self.all_data_gen = datasets.ImageFolder(self.data_dir_all,self.data_transforms)
        self.all_data = torch.utils.data.DataLoader(self.all_data_gen,
                                                     batch_size=len(self.all_data_gen),
                                                     shuffle=False,
                                                     num_workers=0)
        for data_test in self.all_data:
            self.X_test_all, self.Y_test_all = data_test
        self.X_test_all = self.X_test_all.numpy()
        self.Y_test_all = self.Y_test_all.numpy()

        self.input_shape = (self.width, self.width, self.channals)
        self.input_1 = Input(shape=self.input_shape)
        #====================== Model =============================
        self.model = self.build_model()

        if mode=='train':
            self.model = self.train(self.model)
        elif mode=='load':
            self.model.load_weights("{}".format(self.filename))
        else:
            raise Exception("Sorry, select the right mode option (train/load)")

    def conv1(self):
        zpad = ZeroPadding2D(padding=(3, 3), data_format='channels_last')(self.input_1)
        '''
        加ZeroPadding2D是因为通过keras的API导出的ResNet_50模型中有ZeroPadding2D层，而ResNet_18和ResNet_50的开头都是一样的。
        '''
        con = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')(zpad)
        bn = BatchNormalization()(con)
        ac = Activation('relu')(bn)
        zpad = ZeroPadding2D()(ac)
        mp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(zpad)
        return mp

    ##########################################################################
    def conv2_x(self,cat):
        '''
        参数cat是指前一个模块，接下来搭建的层接到这个cat模块上。
        在第一张图中可以看出conv2_x是两个有两层卷积的模块。所以这里搭建两个卷积层。
        '''
        con = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(cat)
        bn = BatchNormalization()(con)
        ac = Activation('relu')(bn)

        con = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(ac)
        bn = BatchNormalization()(con)

        conv2_x_add = add([bn, cat])

        ac = Activation('relu')(conv2_x_add)

        return ac

    def conv3_x(self,cat, strides=1):
        '''
        参数cat是指前一个模块，接下来搭建的层接到这个cat模块上。
        在第一张图中可以看出conv3_x是两个有两层卷积的模块。所以这里搭建两个卷积层。
        与此同时，还要判断当前的两个卷积层的上一个残差连接模块的维度是64还是128？
        通过strides参数来判断。
        strides = 1说明上一个残差连接模块的维度是128。
        strides = 2说明上一个残差连接模块的维度是64。需要把上个残差模块的特征图进行卷积降维到28。也就是特征图尺寸减半，滤波器数量加倍。
        '''
        if strides == 2:
            con = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(
                cat)  # 特征图尺寸减半，滤波器数量加倍。
        else:
            con = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(
                cat)  # 相同的输出特征图，层具有相同数量的滤波器

        bn = BatchNormalization()(con)
        ac = Activation('relu')(bn)
        con = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(ac)
        bn1 = BatchNormalization()(con)

        if strides == 2:
            con = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='same')(
                cat)  # 特征图尺寸减半，滤波器数量加倍。
            bn2 = BatchNormalization()(con)
            conv2_x_add = add([bn1, bn2])
        else:
            conv2_x_add = add([bn1, cat])  # 相同的输出特征图，层具有相同数量的滤波器

        ac = Activation('relu')(conv2_x_add)
        return ac

    def conv4_x(self,cat, strides=1):
        '''
        参数说明与conv3_x一样，这里仅改变filters的个数
        '''
        if strides == 2:
            con = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(
                cat)  # 特征图尺寸减半，滤波器数量加倍。
        else:
            con = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(
                cat)  # 相同的输出特征图，层具有相同数量的滤波器

        bn = BatchNormalization()(con)
        ac = Activation('relu')(bn)
        con = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(ac)
        bn1 = BatchNormalization()(con)

        if strides == 2:
            con = Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same')(
                cat)  # 特征图尺寸减半，滤波器数量加倍。
            bn2 = BatchNormalization()(con)
            conv2_x_add = add([bn1, bn2])
        else:
            conv2_x_add = add([bn1, cat])  # 相同的输出特征图，层具有相同数量的滤波器

        ac = Activation('relu')(conv2_x_add)
        return ac

    def conv5_x(self, cat, strides=1):
        '''
        参数说明与conv3_x一样，这里仅改变filters的个数
        '''
        if strides == 2:
            con = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(
                cat)  # 特征图尺寸减半，滤波器数量加倍。
        else:
            con = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(
                cat)  # 相同的输出特征图，层具有相同数量的滤波器

        bn = BatchNormalization()(con)
        ac = Activation('relu')(bn)
        con = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same')(ac)
        bn1 = BatchNormalization()(con)

        if strides == 2:
            con = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(
                cat)  # 特征图尺寸减半，滤波器数量加倍。
            bn2 = BatchNormalization()(con)
            conv2_x_add = add([bn1, bn2])
        else:
            conv2_x_add = add([bn1, cat])  # 相同的输出特征图，层具有相同数量的滤波器

        ac = Activation('relu')(conv2_x_add)
        return ac

    def build_model(self):
    ##########################################################################
        con1 = self.conv1()
        ##########################################################################
        con2 = self.conv2_x(con1)
        con2 = self.conv2_x(con2)

        ##########################################################################
        con3 = self.conv3_x(con2, strides=2)
        con3 = self.conv3_x(con3, strides=1)
        ##########################################################################
        con4 = self.conv4_x(con3, strides=2)
        con4 = self.conv4_x(con4, strides=1)

        ##########################################################################
        con5 = self.conv5_x(con4, strides=2)
        con5 = self.conv5_x(con5, strides=1)
        ##########################################################################
        avg = AveragePooling2D(pool_size=(7, 7), padding='same')(con5)
        flatten = Flatten()(avg)
        if self.num_classes == 1:
            dense = Dense(units=self.num_classes, activation='sigmoid')(flatten)
        else:
            dense = Dense(units=self.num_classes, activation='softmax')(flatten)

        model = Model(inputs=[self.input_1], outputs=[dense])

        return model

    # if __name__ == '__main__':
    #     model = ResNet18((INPUT_SIZE, INPUT_SIZE, 3), CLASS_NUM)
    #     print('Done.')
    #
    #
    #
    # def build_model(self):
    #     #================= Settings =========================
    #     weight_decay = 0.0005
    #     basic_dropout_rate = 0.1
    #
    #     #================= Input ============================
    #     input = Input(shape=self.input_shape, name='l_0')
    #
    #     #================= CONV ============================
    #     task0 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_1')(input)
    #     task0 = BatchNormalization(name='l_2')(task0)
    #     task0 = Activation('relu', name='l_3')(task0)
    #
    #     task0 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_4')(task0)
    #     task0 = BatchNormalization(name='l_5')(task0)
    #     task0 = Activation('relu', name='l_6')(task0)
    #     task0 = MaxPooling2D(pool_size=(2, 2), name='l_7')(task0)
    #     task0 = Dropout(basic_dropout_rate, name='l_8')(task0)
    #
    #     task0 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_9')(task0)
    #     task0 = BatchNormalization(name='l_10')(task0)
    #     task0 = Activation('relu', name='l_11')(task0)
    #
    #     task0 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_12')(task0)
    #     task0 = BatchNormalization(name='l_13')(task0)
    #     task0 = Activation('relu', name='l_14')(task0)
    #     task0 = MaxPooling2D(pool_size=(2, 2), name='l_15')(task0)
    #     task0 = Dropout(basic_dropout_rate+0.1, name='l_16')(task0)
    #
    #     task0 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_17')(task0)
    #     task0 = BatchNormalization(name='l_18')(task0)
    #     task0 = Activation('relu', name='l_19')(task0)
    #
    #     task0 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_20')(task0)
    #     task0 = BatchNormalization(name='l_21')(task0)
    #     task0 = Activation('relu', name='l_22')(task0)
    #     task0 = MaxPooling2D(pool_size=(2, 2), name='l_23')(task0)
    #     task0 = Dropout(basic_dropout_rate + 0.2, name='l_24')(task0)
    #
    #     task0 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=l2(weight_decay), name='l_25')(task0)
    #     task0 = BatchNormalization(name='l_26')(task0)
    #     task0 = Activation('relu', name='l_27')(task0)
    #     task0 = MaxPooling2D(pool_size=(2, 2), name='l_28')(task0)
    #     task0 = Dropout(basic_dropout_rate + 0.3, name='l_29')(task0)
    #
    #     #================= Dense ============================
    #     task0 = Flatten(name='l_30')(task0)
    #     task0 = Dense(512, kernel_regularizer=l2(weight_decay), name='l_31')(task0)
    #     #task0 = Dropout(basic_dropout_rate + 0.3, name='l_32')(task0)
    #
    #     #================= Output - classification head ============================
    #     classification_output = Dense(self.num_classes, name="classification_head_before_softmax")(task0)
    #     classification_output = Activation('softmax', name="classification_head")(classification_output)
    #
    #     #================= The final model ============================
    #     model = Model(inputs=input, outputs=classification_output)
    #     return model
    
    def train(self, model):
        #================= Settings =========================
        learning_rate = 0.001
        lr_decay = 1e-6
        lr_drop = 30
        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = LearningRateScheduler(lr_scheduler)
        weights_file = "{}{}".format(checkpoints_dir, self.filename)
        model_checkpoint = ModelCheckpoint(weights_file, monitor='val_accuracy', save_best_only=True, verbose=1)
        callbacks=[reduce_lr, model_checkpoint]

        #================= Data augmentation =========================
        # datagen = ImageDataGenerator(
        #     featurewise_center=False,  # set input mean to 0 over the dataset
        #     samplewise_center=False,  # set each sample mean to 0
        #     featurewise_std_normalization=False,  # divide inputs by std of the dataset
        #     samplewise_std_normalization=False,  # divide each input by its std
        #     zca_whitening=False,  # apply ZCA whitening
        #     rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        #     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        #     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #     horizontal_flip=True,  # randomly flip images
        #     vertical_flip=False)  # randomly flip images
        # datagen.fit(self.x_train)

        #================= Train =========================
        sgd = optimizers.SGD(lr=learning_rate,decay=lr_decay,momentum=0.99, nesterov=True)
        model.compile(loss=categorical_crossentropy, optimizer=sgd, metrics=['accuracy'])

        historytemp = model.fit_generator(self.train_data_gen,epochs=self.epochs,
                                          validation_data=self.val_data_gen, validation_freq=2,
                                          callbacks=callbacks)
        
        #================= Save model and history =========================
        with open("{}{}_history.pkl".format(checkpoints_dir, self.filename[:-3]), 'wb') as handle:
            pickle.dump(historytemp.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # # model.save_weights(weights_file)

        return model