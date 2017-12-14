# -*- coding: utf-8 -*-
# 增量式图像识别程序
from __future__ import print_function
import inception_v4
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import cv2
import tkFileDialog
import shutil
import matplotlib.pyplot as plt
import math

def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 1.0)
    x = np.multiply(x, 2.0)
    return x
def get_processed_image(img):
	# Load image and convert from BGR to RGB
	im = np.asarray(img)[:,:,::-1]
	im = cv2.resize(im, (299, 299))
	im = preprocess_input(im)
	im = im.reshape(-1,299,299,3)
	return im

def get_feature_mean(train_data,height):
    temp = np.zeros((1,1536),dtype='float32')
    for m in xrange(height):
        temp[0] = temp[0]+train_data[m]
    temp[0] = temp[0]/height
    return temp

#用夹角余弦进行相似性度量并判别
def classification_with_cosine(W,labels,origion_out):
    distance = np.zeros(len(labels),dtype='float32')
    num = 0
    for i in labels:
        temp = (sum(origion_out[0] * W[i])) / ((math.sqrt(sum(pow(origion_out[0], 2)))) * (math.sqrt(sum(pow(W[i], 2)))))
        distance[num] = temp
        num += 1
    a = np.argmax(distance)
    label = labels[a]
    return label
#数据扩充
def data_augmentation(img):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir='augmentation', save_prefix='1', save_format='jpeg'):
        i += 1
        if i >= 30:
            break
#提取特征
def get_features(origion_out,img):
    data_augmentation(img)
    imgs = os.listdir('augmentation')
    num = len(imgs) + 1
    features = np.zeros((num, 1536), dtype='float32')
    features[0] = origion_out
    n = 1
    for f in imgs:
        image = cv2.imread('augmentation/'+f)
        im = get_processed_image(image)
        out = model.predict(im)
        features[n] = out
        n += 1
    shutil.rmtree('augmentation')
    os.mkdir('augmentation')
    feature = np.array(features)
    return feature


if __name__ == '__main__':
    first_learn = True #设定为第一次学习
    model = inception_v4.create_model(weights='imagenet', include_top=True)
    classes = 1  #类别计数
    class_name = [] #存储类别名字
    class_count = np.zeros(1000,dtype='int')
    #判断临时文件夹时候是否存在，augmentation文件夹用于保存数据增强时生成的文件
    dir = os.path.exists('augmentation')
    if not dir:
        os.mkdir('augmentation')
    #用于存储每一类特征向量每一位的最大最小
    features_max = np.zeros((1000,1536),dtype='float32')
    features_min = np.zeros((1000,1536), dtype='float32')
    W = np.zeros((1000,1536),dtype='float32') #存储每一类特征向量的均值
    cv2.namedWindow('image')
    while (True):
        filename = tkFileDialog.askopenfilename(initialdir='/home/deep/dataset/ourdata')#文件选择框
        img = load_img(filename)
        # preprocessing
        image = cv2.imread(filename)
        im = get_processed_image(image)
        out = model.predict(im) #得到原始输入图片的特征向量
        cv2.imshow('image',image)
        cv2.waitKey(100)
        #第一次学习
        if first_learn:
            str = raw_input('Can you tell me what is this?:'); #询问所输入的图片是什么物品
            class_name.append(str) #存储该类物品的名称
            class_count[0] += 1
            print ("I'm learning...")
            #数据增强并获得特征，再求得特征最大最小
            feature = get_features(out,img)
            for i in xrange(1536):
                features_max[0][i] = feature[:, i].max()
                features_min[0][i] = feature[:, i].min()
            height, width = feature.shape
            #获取特征均值
            W[0] = get_feature_mean(feature, height)
            first_learn = False
            print ('I finished my studies..')
        #第二次及之后的输入
        else:
            predict_classes = []
            #计算输入的图片的特征满足特征最大最小的位数
            for i in xrange(classes):
                m = 0
                for j in xrange(1536):
                    if out[0][j]<=(features_max[i][j])*1.05 and out[0][j]>=(features_min[i][j])*0.95:
                        m = m+1
                if m>=1480:
                    predict_classes.append(i)
            #先采取特征向量每一位最大最小比较的方法，predict_classes分为三种情况，
            #当predict_classes为空时，即输入的图片不满足当前系统所学习任何模型
            #当predict_classes的长度为一时，输入的图片满足当前系统中的某一类的模型
            #当predict_classes的长度大于一时，输入的图片满足当前系统所学的多种，此时要用夹角余弦再分类

            # m=0 系统进行学习
            if len(predict_classes) == 0:
                print ('I can not recognize it..')
                str = raw_input('Can you tell me what is this?:'); #询问所输入的图片是什么物品
                # 查询是否为之前学习过的物品，是的话更新该类的模型，否的话则为新的一类物品，系统学习该新物体
                if not str in class_name:
                    class_name.append(str)
                    classes += 1
                    class_count[classes] +=1
                    print ("I'm learning...")
                    feature = get_features(out,img)
                    for i in xrange(1536):
                        features_max[classes][i] = feature[:, i].max()
                        features_min[classes][i] = feature[:, i].min()
                    height, width = feature.shape
                    W[classes]= get_feature_mean(feature, height)
                    print ('I finished my studies..')
                else:
                    print ("I'm learning...")
                    index = class_name.index(str)
                    class_count[index] += 1
                    feature = get_features(out,img)
                    a = np.row_stack((feature, features_max[index]))
                    b = np.row_stack((feature, features_min[index]))
                    for i in xrange(1536):
                        features_max[index][i] = a[:, i].max()
                        features_min[index][i] = b[:, i].min()
                    height, width = feature.shape
                    temp_W = get_feature_mean(feature, height)
                    W[index] = (W[index]+temp_W[0])/class_count[index]
                    print ('I finished my studies..')
            #m=1，输入系统认为的类别，判断的正确与否用户反馈给系统，系统根据用户的反馈进行相应的处理
            elif len(predict_classes) == 1:
                print ('This is a(n) ',class_name[predict_classes[0]])
                str = raw_input(r'Did i judge it correctly?(y/n):');
                if str == 'n':
                    #判断错误，用户告诉系统这究竟是什么，系统根据用户的反馈来进行学习
                    str = raw_input('Oh~ So what is this?:');
                    # 查询是否为之前学习过的物品，是的话更新该类的模型，否的话则为新的一类物品，系统学习该新物体
                    if not str in class_name:
                        class_name.append(str)
                        classes += 1
                        class_count[classes] +=1
                        print ("I'm learning...")
                        feature = get_features(out,img)
                        for i in xrange(1536):
                            features_max[classes][i] = feature[:, i].max()
                            features_min[classes][i] = feature[:, i].min()
                        height, width = feature.shape
                        W[classes] = get_feature_mean(feature, height)
                        print ('I finished my studies..')
                    else:
                        print ("I'm learning...")
                        index = class_name.index(str)
                        class_count[index] += 1
                        feature = get_features(out,img)
                        a = np.row_stack((feature, features_max[index]))
                        b = np.row_stack((feature, features_min[index]))
                        for i in xrange(1536):
                            features_max[index][i] = a[:, i].max()
                            features_min[index][i] = b[:, i].min()
                        height, width = feature.shape
                        temp_W = get_feature_mean(feature, height)
                        W[index] = (W[index] + temp_W[0]) / class_count[index]
                        print ('I finished my studies..')
            # m>=2，此时用特征向量每一位最大最小的方法已不能判断该输入的图片
            # 用夹角余弦分类方法进行细分
            else:
                labels = []
                for f in predict_classes:
                    labels.append(f)
                label = classification_with_cosine(W,labels,out) #用夹角余弦进行判别
                #输出判别结果，用户给予系统反馈，系统根据用户的反馈做出相应的处理
                print ('This is a(n) :',class_name[label]) #输出判别结果
                str = raw_input(r'Did i judge it correctly?(y/n):');
                if str == 'n':
                    str = raw_input('Oh~ So what is this?:');
                    if not str in class_name:
                        class_name.append(str)
                        classes += 1
                        class_count[classes] += 1
                        print ("I'm learning...")
                        feature = get_features(out,img)
                        for i in xrange(1536):
                            features_max[classes][i] = feature[:, i].max()
                            features_min[classes][i] = feature[:, i].min()
                        height, width = feature.shape
                        W[classes] = get_feature_mean(feature, height)
                        print ('I finished my studies..')
                    else:
                        print ("I'm learning...")
                        index = class_name.index(str)
                        class_count[index] += 1
                        feature = get_features(out,img)
                        height, width = feature.shape
                        temp_W = get_feature_mean(feature, height)
                        W[index] = (W[index] + temp_W[0]) / class_count[index]
                        print ('I finished my studies..')

        str = raw_input(r'Continue or not or open camera to test?(y/n/cam):');
        print ('\n')
        if str == 'n':
            break
        if str == 'cam':
           cap = cv2.VideoCapture(0)
           fps_count = 0
           while (True):
               predict_classes = []
               ret,frame = cap.read()
               if fps_count == 1:
                   fps_count = 0
                   im = get_processed_image(frame)
                   out = model.predict(im)
                   for i in xrange(classes):
                       m = 0
                       for j in xrange(1536):
                           if out[0][j] <= (features_max[i][j]) * 1.05 and out[0][j] >= (features_min[i][j]) * 0.95:
                               m = m + 1
                       if m >= 1480:
                           predict_classes.append(i)
                   if len(predict_classes) == 0:
                       str = 'I can not recognize anything! please let me learn more..'
                       cv2.putText(frame,str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                   elif len(predict_classes) == 1:
                       name = class_name[predict_classes[0]]
                       str = 'This is a(n) '+ name
                       cv2.putText(frame, str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                   else:
                       labels = []
                       for f in predict_classes:
                           labels.append(f)
                       label = classification_with_cosine(W, labels, out)
                       name = class_name[label]
                       str = 'This is a(n) '+name
                       cv2.putText(frame, str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
               fps_count += 1
               str = "press 'q' to quit.."
               cv2.putText(frame,str, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
               cv2.imshow('image',frame)
               if cv2.waitKey(10) & 0xff == ord('q'):
                   break
           cap.release()

