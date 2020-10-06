//
// Created by c1535 on 2020/10/5.
//

#ifndef OPENCV_TEST_POLARIZATION_PREPROCESS_H
#define OPENCV_TEST_POLARIZATION_PREPROCESS_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include<string>
#include<vector>
#include <cstdio>
#include <cstdlib>
#include<cmath>
#include <io.h>
using namespace std;
using namespace cv;
double WienerFilterImpl(const Mat& src, Mat& dst, double noiseVariance, const Size& block);
//WienerFilter 用来维纳滤波噪声估计
double WienerFilter(const Mat& src, Mat& dst, const Size& block = Size(5, 5));
//WienerFilter 维纳滤波函数
void WienerFilter(const Mat& src, Mat& dst, double noiseVariance, const Size& block = Size(5, 5));
//on_mouse 鼠标事件判定函数，主要与回调函数setMouseCallback一起使用
void on_mouse(int event, int x, int y, int flags, void*);//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
//filestring 图像所在文件夹，末尾必须要有"\"；imagepath为图像ROI选取鼠标操作图像，是图像绝对路径
//shape 为图像格式，如：".jpg";Imax,Imin为运算结果，DEBUG：常量-》2为截取+截取后，1为截取后+分割，0为不显示
//Threshold为阈值，sigma为滤波偏差
void ImageProcessGetmaxmin(const string& filestring, string imagepath, const string& shape, Mat& Imax, Mat& Imin,
                           int &DEBUG,int &Threshold,double &sigma);
#endif //OPENCV_TEST_POLARIZATION_PREPROCESS_H