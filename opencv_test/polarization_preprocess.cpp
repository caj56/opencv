//
// Created by c1535 on 2020/10/5.
//
#include "polarization_preprocess.h"
//org:为输入图像数据;img:为ROI区域显示的副本（及对此图操作不影响原图）
static Mat org, img, tmp;
//该容器用来存放ROI区域的坐标信息
static vector<int> res;
double WienerFilterImpl(const Mat& src, Mat& dst, double noiseVariance, const Size& block) {
    //判断图像是否是单通道灰度图像以便进行运算
    if(!(block.width % 2 == 1 && block.height % 2 == 1 && block.width > 1 && block.height > 1)){
        cout << "Invalid block dimensions" << endl;
        return -1;
    }
    if(!(src.channels() == 1 && dst.channels() == 1)){
        cout << "src and dst must be one channel grayscale images" << endl;
        return -1;
    }
    int h = src.rows;
    int w = src.cols;
    dst = Mat1b(h, w);
    Mat1d means, sqrMeans, variances;
    Mat1d avgVarianceMat;
    //方框滤波
    boxFilter(src, means, CV_64F, block, Point(-1, -1), true, BORDER_REPLICATE);
    sqrBoxFilter(src, sqrMeans, CV_64F, block, Point(-1, -1), true, BORDER_REPLICATE);

    Mat1d means2 = means.mul(means);
    variances = sqrMeans - (means.mul(means));

    if (noiseVariance < 0) {
        // 估计噪声的数值
        reduce(variances, avgVarianceMat, 1, CV_REDUCE_SUM, -1);
        reduce(avgVarianceMat, avgVarianceMat, 0, CV_REDUCE_SUM, -1);
        double hw;
        hw = (double)h * (double)w;
        noiseVariance = avgVarianceMat(0, 0) / hw;
    }

    for (int r = 0; r < h; ++r) {
        // 求取图像每行的坐标位置
        auto const* const srcRow = src.ptr<uchar>(r);
        auto* const dstRow = dst.ptr<uchar>(r);
        auto* const varRow = variances.ptr<double>(r);
        auto* const meanRow = means.ptr<double>(r);
        for (int c = 0; c < w; ++c) {
            dstRow[c] = saturate_cast<uchar>(
                    meanRow[c] + max(0., varRow[c] - noiseVariance) / max(varRow[c], noiseVariance) * (srcRow[c] - meanRow[c])
            );
        }
    }
    return noiseVariance;
}
double WienerFilter(const Mat& src, Mat& dst, const Size& block) {
    return WienerFilterImpl(src, dst, -1, block);
}
void WienerFilter(const Mat& src, Mat& dst, double noiseVariance, const Size& block) {
    WienerFilterImpl(src, dst, noiseVariance, block);
}
void on_mouse(int event, int x, int y, int flags, void*)//event鼠标事件代号，x,y鼠标坐标，flags拖拽和键盘操作的代号
{
    static Point pre_pt = Point(-1, -1);//初始坐标
    static Point cur_pt = Point(-1, -1);//实时坐标
    char temp[16];
    if (event == CV_EVENT_LBUTTONDOWN)//左键按下，读取初始坐标，并在图像上该点处划圆
    {
        org.copyTo(img);//将原始图片复制到img中
        sprintf_s(temp, "(%d,%d)", x, y);
        pre_pt = Point(x, y);
        putText(img, temp, pre_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255), 1, 8);//在窗口上显示坐标
        circle(img, pre_pt, 2, Scalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);//划圆
        imshow("img", img);
    }
    else if (event == CV_EVENT_MOUSEMOVE && !(flags & CV_EVENT_FLAG_LBUTTON))//左键没有按下的情况下鼠标移动的处理函数
    {
        img.copyTo(tmp);//将img复制到临时图像tmp上，用于显示实时坐标
        sprintf_s(temp, "(%d,%d)", x, y);
        cur_pt = Point(x, y);
        putText(tmp, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));//只是实时显示鼠标移动的坐标
        imshow("img", tmp);
    }
    else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))//左键按下时，鼠标移动，则在图像上划矩形
    {
        img.copyTo(tmp);
        sprintf_s(temp, "(%d,%d)", x, y);
        cur_pt = Point(x, y);
        putText(tmp, temp, cur_pt, FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0, 255));
        rectangle(tmp, pre_pt, cur_pt, Scalar(0, 255, 0, 0), 1, 8, 0);//在临时图像上实时显示鼠标拖动时形成的矩形
        imshow("img", tmp);
    }
    else if (event == CV_EVENT_LBUTTONUP)//左键松开，将在图像上划矩形
    {
        org.copyTo(img);
        sprintf_s(temp, "(%d,%d)", x, y);
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
        res.push_back(min(cur_pt.x, pre_pt.x));
        res.push_back(min(cur_pt.y, pre_pt.y));
        res.push_back(w);
        res.push_back(h);
        for (int re : res) {
            cout << re << endl;
        }
        cout << endl;
    }
}
void ImageProcessGetmaxmin(const string& filestring, string imagepath, const string& shape, Mat& Imax, Mat& Imin,
                           int &DEBUG,int &Threshold,double &sigma) {
    org = imread(imagepath);
    if (org.empty()) {
        cout << "读取图像文件错误，检查文件路径是否正确！" << endl;
        return;
    }
    int length = (int)imagepath.length();
    int n = (int)imagepath.find(shape);
    imagepath.replace(n,length,".txt");
    if (_access(imagepath.c_str(), 0) != -1) { //若文件存在,则读出每一个数据存到array1
        cout << "文本文件已存在" << endl;
        ifstream data(imagepath); //待读取文件的目录
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
    }
    else {
        cout << "文本文件不存在" << endl;
        org.copyTo(img);
        namedWindow("img");//定义一个img窗口
        setMouseCallback("img", on_mouse, nullptr);//调用回调函数
        imshow("img", img);
        waitKey(0);
        FILE* fp;
        errno_t err = fopen_s(&fp, imagepath.c_str(),"a+");
        if (err == 0){
            for (int re : res) {
                fprintf(fp, "%d\t", re);
            }
        }
        fclose(fp);
    }
    int x1 = (int)round(res[0]);
    int y1 = (int)round(res[1]);
    int w1 = (int)round(res[2]);
    int h1 = (int)round(res[3]);
    //图像裁剪
    Mat I;
    Mat I0_all, I45_all, I90_all, I135_all;
    for (int i = 0; i <= 135; i += 45) {
        char str[16] = { 0 };
        _itoa_s(i, str, 10);
        string filename = filestring + str + shape;
        Mat img2 = imread(filename);
        if (img2.empty()) {
            cout << "分割图像读取错误！请检查路径是否正确！" << endl;
            break;
        }
        //做精度转换，但不影响分割操作
        Mat gray;
        cvtColor(img2, gray, COLOR_BGR2GRAY);
        gray.convertTo(gray, CV_64F, 1 / 255.0);
        if (DEBUG == 2) {
            imshow("initial image", gray);
            waitKey(1000);
        }
        if (i == 45) {
            Rect rect(x1, y1 + 5, w1, h1);
            I = gray(rect);
        }
        else if (i == 90) {
            Rect rect(x1 - 2, y1 + 2, w1, h1);
            I = gray(rect);
        }
        else if (i == 135) {
            Rect rect(x1 - 4, y1 + 4, w1, h1);
            I = gray(rect);
        }
        else {
            Rect rect(x1, y1, w1, h1);
            I = gray(rect);
        }
        string saveimage = filestring + str + "1.jpg";
        //这次转换数据类型用于保存图像
        I.convertTo(I, CV_8U, 255);
        //imwrite(saveimage,I);

        if (DEBUG > 0) {
            imshow("intercept image", I);
            waitKey(1000);
        }
        //阈值分割
        I.convertTo(I, CV_64F, 1 / 255.0);//转换为精度double类型，并进行归一化
        Mat I_all_threshold;
        threshold(I, I_all_threshold, Threshold / (pow(2, 8) - 1), 1, THRESH_BINARY);
        Mat I_all;
        int h = I_all_threshold.rows;
        int w = I_all_threshold.cols;
        I_all = Mat::zeros(I.size(), I.type());
        for (int a = 0; a < h; a++) {
            for (int b = 0; b < w; b++) {
                double u = I.at<double>(a, b);
                double v = I_all_threshold.at<double>(a, b);
                I_all.at<double>(a, b) = u * v;
            }
        }
        //去噪:灰度double图像映射到uint8图像计算噪声，进行维纳滤波
        //然后转换到double数据类型进行高斯滤波
        I_all.convertTo(I_all, CV_8U, 255);
        Mat gausFilter, I_all1;
        double noise = WienerFilter(I_all, gausFilter, Size(5, 5));
        WienerFilter(I_all, I_all1, noise, Size(5, 5));
        I_all1.convertTo(I_all1, CV_64F, 1 / 255.0);
        GaussianBlur(I_all1, I_all, Size(5, 5), sigma);
        //
        if (DEBUG == 1) {
            imshow("split image", I_all);
            waitKey(1000);
        }
        if (i == 0) {
            I_all.convertTo(I0_all, CV_64F);
        }
        else if (i == 45) {
            I_all.convertTo(I45_all, CV_64F);
        }
        else if (i == 90) {
            I_all.convertTo(I90_all, CV_64F);
        }
        else if (i == 135) {
            I_all.convertTo(I135_all, CV_64F);
        }
    }
    //
    I = I0_all + I90_all;
    Mat Q = I0_all - I90_all;
    Mat U = (I45_all + I135_all) - (I0_all + I90_all);
    Imax = Mat::zeros(I.size(), I.type());
    Imin = Mat::zeros(I.size(), I.type());
    for (int i = 0; i < I.rows; i++) {
        for (int j = 0; j < I.cols; j++) {
            double a = Q.at<double>(i, j);
            double b = U.at<double>(i, j);
            double c = I.at<double>(i, j) + sqrt(pow(a, 2) + pow(b, 2));
            Imax.at<double>(i, j) = c / 2.0;
            Imin.at<double>(i, j) = c / 2.0;
        }
    }
}

