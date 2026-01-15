#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void sobel_x(const string &path){
    Mat img = imread(path);
    Mat grey;
    cvtColor(img,grey,COLOR_BGR2GRAY);
    Mat sobelx_kernel = (Mat_<float>(3,3) << 
        -1,0,1,
        -2,0,-2,
        -1,0,-1);
    Mat x_grad ;
    filter2D(grey,x_grad,CV_32F ,sobelx_kernel);
    Mat abs_xgrad;
    convertScaleAbs(x_grad,abs_xgrad);
    imwrite("data/results/sobel_x.jpg",abs_xgrad);
}
void sobel_y(const string &path){
    Mat img = imread(path);
    Mat grey;
    cvtColor(img,grey,COLOR_BGR2GRAY);
    Mat sobely_kernel = (Mat_<float>(3,3) << 
        -1,-2,-1,
        0,0,0,
        -1,-2,-1);
    Mat y_grad;
    filter2D(grey, y_grad , CV_32F ,sobely_kernel);
    Mat abs_ygrad;
    convertScaleAbs(y_grad , abs_ygrad);
    imwrite("data/results/sobel_y.jpg",abs_ygrad);
}
void gaussian_smoothing(const string &path){
    //from defining kernel
    Mat img = imread(path);
    Mat gaussian_kernel = (Mat_<float>(5,5) << 1,4,6,4,1,
                                               4,16,24,16,4,
                                               6,24,36,24,6,
                                               4,16,24,16,4,
                                               1,4,6,4,1) / 250.0;
    Mat smooth_k;
    filter2D(img,smooth_k,-1,gaussian_kernel);
    //from opencv inbuilt function
    Mat smooth_f;
    GaussianBlur(img,smooth_f,Size(5,5),1.5,1.5);

    imwrite("data/results/gaussian_k.jpg",smooth_k);
    imwrite("data/results/gaussian_f.jpg",smooth_f);


}
int main(){
    sobel_x("data/raw/sae.jpg");
    sobel_y("data/raw/sae.jpg");
    gaussian_smoothing("data/raw/sae.jpg");

}