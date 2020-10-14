#include <iostream>
using namespace std;
#include "polarization_preprocess.h"
//测试文件
int main()
{
    string mdir = R"(C:\Users\CAJ\Desktop\program\)";
    string imgpath = R"(C:\Users\CAJ\Desktop\program\0.jpg)";
    string geshi = ".jpg";
    Mat imagemin, imagemax;
    int DEBUG = 2;
    int Threshold = 15;
    double sigma = 1.6;
    ImageProcessGetmaxmin(mdir, imgpath, geshi, imagemax, imagemin,DEBUG,Threshold,sigma);
    Mat add,sub,dop;
    dop = Mat::zeros(imagemax.size(),imagemax.type());
    add = imagemax + imagemin;
    sub = imagemax - imagemin;
    for(int i = 0;i < imagemax.rows;i++){
        for(int j = 0;j < imagemax.cols;j++){
            double c = add.at<double>(i,j);
            double d = sub.at<double>(i,j);
            dop.at<double>(i,j) =  d / c;
        }
    }
    dop.convertTo(dop, CV_8U);
    namedWindow("dop",0);
    imshow("dop",dop);
//    imagemax.convertTo(imagemax, CV_8U);
//    imagemin.convertTo(imagemin, CV_8U);
//    imshow("Imax", imagemax);
//    imshow("Imin", imagemin);
    waitKey(0);
}
