#include <iostream>
using namespace std;
#include "polarization_preprocess.h"
int main()
{
    string mdir = R"(C:\Users\c1535\Desktop\Program\)";
    string imgpath = R"(C:\Users\c1535\Desktop\Program\0.jpg)";
    string geshi = ".jpg";
    Mat imagemin, imagemax;
    int DEBUG = 2;
    int Threshold = 15;
    double sigma = 1.6;
    ImageProcessGetmaxmin(mdir, imgpath, geshi, imagemax, imagemin,DEBUG,Threshold,sigma);
    imshow("Imax", imagemax);
    imshow("Imin", imagemin);
    waitKey(0);
}
