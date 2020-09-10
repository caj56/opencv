#include <iostream>
#include <opencv2/opencv.hpp>
#include<string>
#include<vector>
#include <cstdio>
#include <cstdlib>
#include<cmath>
using namespace std;
using namespace cv;
//org:为输入图像数据;img:为ROI区域显示的副本（及对此图操作不影响原图）
Mat org, img, tmp;
int array1[4];
int DEBUG= 2;
int Threshold = 15;
float sigma = 1.6;
int _access(const char *pathname, int mode);
double WienerFilter(const cv::Mat& src, cv::Mat& dst, const cv::Size& block = cv::Size(5, 5));
void WienerFilter(const cv::Mat& src, cv::Mat& dst, double noiseVariance, const cv::Size& block = cv::Size(5, 5));
void on_mouse(int event, int x, int y, int flags, void *ustc)//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    static Point pre_pt = Point(-1, -1);//初始坐标
    static Point cur_pt = Point(-1, -1);//实时坐标
    char temp[16];
    if (event == CV_EVENT_LBUTTONDOWN)//左键按下，读取初始坐标，并在图像上该点处划圆
    {
        org.copyTo(img);//将原始图片复制到img中
        sprintf(temp, "(%d,%d)", x, y);
        pre_pt = Point(x, y);
        putText(img, temp, pre_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255), 1, 8);//在窗口上显示坐标
        circle(img, pre_pt, 2, Scalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);//划圆
        imshow("img", img);
    }
    else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))//左键没有按下的情况下鼠标移动的处理函数
    {
        img.copyTo(tmp);//将img复制到临时图像tmp上，用于显示实时坐标
        sprintf(temp, "(%d,%d)", x, y);
        cur_pt = Point(x, y);
        putText(tmp, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));//只是实时显示鼠标移动的坐标
        imshow("img", tmp);
    }
    else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))//左键按下时，鼠标移动，则在图像上划矩形
    {
        img.copyTo(tmp);
        sprintf(temp, "(%d,%d)", x, y);
        cur_pt = Point(x, y);
        putText(tmp, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
        rectangle(tmp, pre_pt, cur_pt, Scalar(0, 255, 0, 0), 1, 8, 0);//在临时图像上实时显示鼠标拖动时形成的矩形
        imshow("img", tmp);
    }
    else if (event == CV_EVENT_LBUTTONUP)//左键松开，将在图像上划矩形
    {
        org.copyTo(img);
        sprintf(temp, "(%d,%d)", x, y);
        cur_pt = Point(x, y);
        putText(img, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
        circle(img, pre_pt, 2, Scalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
        rectangle(img, pre_pt, cur_pt, Scalar(0, 255, 0, 0), 1, 8, 0);//根据初始点和结束点，将矩形画到img上
        imshow("img", img);
        img.copyTo(tmp);
        //获取所选区域的矩形数据，即左顶点坐标及长宽，并保存到数组array1中
        int w = abs(pre_pt.x - cur_pt.x);
        int h = abs(pre_pt.y - cur_pt.y);
        if (w == 0 || h == 0)
        {
            printf("width == 0 || height == 0");
            return;
        }
        array1[0] = min(cur_pt.x, pre_pt.x);
        array1[1] = min(cur_pt.y, pre_pt.y);
        array1[2] = w;
        array1[3] = h;
        for(int a;a < 4;a++){
            cout << array1[a] << "\t";
        }
        cout << endl;
    }
}
double WienerFilterImpl(const Mat& src, Mat& dst, double noiseVariance, const Size& block){
    assert(("Invalid block dimensions", block.width % 2 == 1 && block.height % 2 == 1 && block.width > 1 && block.height > 1));
    assert(("src and dst must be one channel grayscale images", src.channels() == 1, dst.channels() == 1));
    int h = src.rows;
    int w = src.cols;
    dst = Mat1b(h, w);
    Mat1d means, sqrMeans, variances;
    Mat1d avgVarianceMat;

    boxFilter(src, means, CV_64F, block, Point(-1, -1), true, BORDER_REPLICATE);
    sqrBoxFilter(src, sqrMeans, CV_64F, block, Point(-1, -1), true, BORDER_REPLICATE);

    Mat1d means2 = means.mul(means);
    variances = sqrMeans - (means.mul(means));

    if (noiseVariance < 0){
        // 估计噪声大小
        reduce(variances, avgVarianceMat, 1, CV_REDUCE_SUM, -1);
        reduce(avgVarianceMat, avgVarianceMat, 0, CV_REDUCE_SUM, -1);
        noiseVariance = avgVarianceMat(0, 0) / (h*w);
    }

    for (int r = 0; r < h; ++r){
        // 得到每行的坐标
        auto const * const srcRow = src.ptr<uchar>(r);
        auto * const dstRow = dst.ptr<uchar>(r);
        auto * const varRow = variances.ptr<double>(r);
        auto * const meanRow = means.ptr<double>(r);
        for (int c = 0; c < w; ++c) {
            dstRow[c] = saturate_cast<uchar>(
                    meanRow[c] + max(0., varRow[c] - noiseVariance) / max(varRow[c], noiseVariance) * (srcRow[c] - meanRow[c])
            );
        }
    }
    return noiseVariance;
}
void WienerFilter(const Mat& src, Mat& dst, double noiseVariance, const Size& block) {
    WienerFilterImpl(src, dst, noiseVariance, block);
}
double WienerFilter(const Mat& src, Mat& dst, const Size& block){
    return WienerFilterImpl(src, dst, -1, block);
}
int main() {
    string mdir = "C:\\Users\\c1535\\Desktop\\Program\\";
    string imgpath ="C:\\Users\\c1535\\Desktop\\Program\\0.jpg";
    string geshi = ".jpg";
    org = imread(imgpath);
    if(org.empty()){
        cout << "读取图像文件错误，检查文件路径是否正确！" << endl;
        return -1;
    }
    int length = imgpath.length();
    imgpath.erase(length-4,length);
    string txt_path = imgpath.append(".txt");
    if (_access(txt_path.c_str(),0) != -1){ //若文件存在,则读出每一个数据存到array1
        cout << "文本文件已存在" << endl;
        ifstream data(txt_path); //待读取文件的目录
        vector<int> res;
        string line;
        while (getline(data, line)) {
            stringstream ss; //输入流
            ss << line; //向流中传值
            if (!ss.eof()) {
                int temp;
                while (ss >> temp) //提取int数据
                    res.push_back(temp); //保存到vector
            }
        }
        for(int i =0;i < res.size();i++){
            array1[i] = res[i];
        }
    }else{
        cout << "文本文件不存在" << endl;
        org.copyTo(img);
        namedWindow("img");//定义一个img窗口
        setMouseCallback("img", on_mouse, nullptr);//调用回调函数
        imshow("img", img);
        waitKey(0);
//        FILE *fp;
//        fp = fopen(txt_path.c_str(),"w");
//        for(int a = 0;a < sizeof(array1)/sizeof(int);a++){
//            fprintf(fp,"%d\t",array1[a]);
//        }
//        fclose(fp);
    }
    int x1 = round(array1[0]);
    int y1 = round(array1[1]);
    int w1 = round(array1[2]);
    int h1 = round(array1[3]);
    //图像裁剪
    Mat I;
    Mat I0_all,I45_all,I90_all,I135_all;
    for(int i = 0;i <= 135; i += 45){
        char str[16] = {0};
        itoa(i,str,10);
        string fliename = mdir + str + geshi;
        Mat img2 = imread(fliename);
        if(img2.empty()){
            cout << "分割图像读取错误！请检查路径是否正确！" << endl;
            break;
        }
        //这儿不做精度转换，因为不影响分割操作
        Mat gray;
        cvtColor(img2, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray,CV_64F,1/255.0);
        if (DEBUG == 2) {
            imshow("原图", gray);
        }
        if (i == 45) {
            Rect rect(x1,y1 + 5,w1,h1);
            I = gray(rect);
        } else if (i == 90) {
            Rect rect(x1 - 2,y1 + 2, w1, h1);
            I = gray(rect);
        } else if (i == 135) {
            Rect rect(x1 - 4,y1 + 4, w1, h1);
            I = gray(rect);
        } else {
            Rect rect(x1,y1,w1,h1);
            I = gray(rect);
        }
        string saveimage = mdir + str + "1.jpg";
        //这次转换数据类型用于保存图像
        I.convertTo(I,CV_8U,255);
        //imwrite(saveimage,I);

        if (DEBUG > 0) {
            imshow("截取后原图",I);
            waitKey(0);
        }
        //阈值分割
        I.convertTo(I,CV_64F,1/255.0);//转换为精度double类型，并进行归一化
        Mat I_all_threshold;
        threshold(I, I_all_threshold, Threshold/(pow(2, 8) - 1), 1, THRESH_BINARY);
        Mat I_all;
        int h = I_all_threshold.rows;
        int w = I_all_threshold.cols;
        I_all = Mat::zeros(I.size(), I.type());
        for (int a = 0; a < h; a++) {
            for (int b = 0; b < w; b++) {
                double u = I.at<double>(a,b);
                double v = I_all_threshold.at<double>(a,b);
                I_all.at<double>(a,b) = u * v;
            }
        }
        //去噪:灰度double图像映射到8uint图像计算噪声，进行维纳滤波
        //然后转换到double数据类型进行高斯滤波
        I_all.convertTo(I_all,CV_8U,255);
        Mat gausFilter,I_all1;
        double noise = WienerFilter(I_all, gausFilter,Size(5,5));
        WienerFilter(I_all, I_all1, noise, Size(5, 5));
        I_all1.convertTo(I_all1,CV_64F,1/255.0);
        GaussianBlur(I_all1, I_all, Size(5, 5), sigma);
        //
        if (DEBUG == 1) {
            imshow("截取后分割图", I_all);
            waitKey(0);
        }
        if(i == 0){
            I_all.convertTo(I0_all,CV_64F);
        } else if(i == 45){
            I_all.convertTo(I45_all,CV_64F);
        }else if(i == 90){
            I_all.convertTo(I90_all,CV_64F);
        }else if(i == 135){
            I_all.convertTo(I135_all,CV_64F);
        }
    }
    //
    I = I0_all + I90_all;
    Mat Q = I0_all-I90_all;
    Mat U = (I45_all+I135_all)-(I0_all+I90_all);
    Mat Imax,Imin;
    Imax = Mat::zeros(I.size(), I.type());
    Imin = Mat::zeros(I.size(), I.type());
    for (int i = 0; i < I.rows; i++) {
        for (int j = 0; j < I.cols; j++) {
            double a = Q.at<double>(i,j);
            double b = U.at<double>(i,j);
            double c = I.at<double>(i,j) + sqrt(pow(a,2)+pow(b,2));
            Imax.at<double>(i,j) = c/2.0;
            Imin.at<double>(i,j) = c/2.0;
        }
    }
    imshow("Imax",Imax);
    imshow("Imin",Imin);
    waitKey(0);
}
