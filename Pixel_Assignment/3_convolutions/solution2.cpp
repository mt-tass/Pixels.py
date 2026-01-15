#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

Mat apply_gaussian(const Mat &src , int ksize , double sigma){
    Mat kernel = getGaussianKernel(ksize,sigma , CV_32F);
    Mat kernel2D = kernel * kernel.t();
    Mat blurred ;
    filter2D(src,blurred , CV_32F ,kernel2D);
    return blurred;

}
void calc_grad(const Mat &src , Mat &magnitude , Mat &direction){
    Mat sobel_x = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat sobel_y = sobel_x.t();
    Mat grad_x , grad_y;
    filter2D(src,grad_x,CV_32F,sobel_x);
    filter2D(src,grad_y,CV_32F , sobel_y);
    magnitude = Mat::zeros(src.size(),CV_32F);
    direction = Mat::zeros(src.size(),CV_32F);
    for (int i = 0; i < src.rows ; i++){
        for (int j = 0 ; j <src.cols ; j++){
            float gx = grad_x.at<float>(i,j);
            float gy = grad_y.at<float>(i,j);
            magnitude.at<float>(i,j) = sqrt(gx*gx+gy*gy);
            direction.at<float>(i,j) = atan(gy/gx)*180.0/CV_PI;
        }
    }
}
Mat nms(const Mat& magnitude, const Mat& direction) {
    Mat suppressed = Mat::zeros(magnitude.size(), CV_32F);
    for (int i = 1; i < magnitude.rows - 1; i++) {
        for (int j = 1; j < magnitude.cols - 1; j++){
            float angle = direction.at<float>(i, j);
            float mag = magnitude.at<float>(i, j);
            if (angle < 0){
                angle += 180;
            }
            float neighbor1 = 0, neighbor2 = 0;
            if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle < 180)) { // this is 0 degree
                neighbor1 = magnitude.at<float>(i, j - 1);
                neighbor2 = magnitude.at<float>(i, j + 1);
            }
            else if (angle >= 22.5 && angle < 67.5){ // this is 45 degrees
                neighbor1 = magnitude.at<float>(i - 1, j + 1);
                neighbor2 = magnitude.at<float>(i + 1, j - 1);
            }
            else if (angle >= 67.5 && angle < 112.5){ // this is 90 degrees
                neighbor1 = magnitude.at<float>(i - 1, j);
                neighbor2 = magnitude.at<float>(i + 1, j);
            }
            else if (angle >= 112.5 && angle < 157.5){ // this is 135 degrees
                neighbor1 = magnitude.at<float>(i - 1, j - 1);
                neighbor2 = magnitude.at<float>(i + 1, j + 1);
            }
            if (mag >= neighbor1 && mag >= neighbor2){
                suppressed.at<float>(i, j) = mag;
            }
        }
    }
    return suppressed;
}

Mat double_thresh_hysteresis(const Mat& src, float lowThreshold, float highThreshold) {
    Mat result = Mat::zeros(src.size(), CV_8U);
    for (int i = 0; i < src.rows; i++){
        for (int j = 0; j < src.cols; j++){
            float val = src.at<float>(i, j);
            if (val >=highThreshold){
                result.at<uchar>(i, j) = 255; //strong
            }
            else if (val >=lowThreshold){
                result.at<uchar>(i, j) = 75; //weak
            }
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = 1; i < result.rows - 1; i++){
            for (int j = 1; j < result.cols - 1; j++){
                if (result.at<uchar>(i, j) == 75){
                    bool hasStrongNeighbor = false;
                    for (int di = -1; di <= 1; di++){
                        for (int dj = -1; dj <= 1; dj++) {
                            if (di == 0 && dj == 0) continue;
                            if (result.at<uchar>(i + di, j + dj) >= 225){
                                hasStrongNeighbor = true;
                                break;
                            }
                        }
                        if (hasStrongNeighbor){
                            break;
                        }
                    }
                    if (hasStrongNeighbor){
                        result.at<uchar>(i, j) = 255;
                        changed = true;
                    }
                    else{
                        result.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
    }
    for (int i = 0; i < result.rows; i++){
        for (int j = 0; j < result.cols; j++){
            if (result.at<uchar>(i, j) == 75){
                result.at<uchar>(i, j) = 0;
            }
        }
    }
    return result;
}
void cannyedge(const string &path){
    Mat img = imread(path);
    Mat grey , greyfloat;
    cvtColor(img,grey,COLOR_BGR2GRAY);
    grey.convertTo(greyfloat,CV_32F);
    Mat blurred = apply_gaussian(greyfloat,5,1.4);
    Mat magnitude,direction;
    calc_grad(blurred , magnitude , direction);
    Mat suppresed = nms(magnitude,direction);
    Mat edges = double_thresh_hysteresis(suppresed , 20 , 50);
    //comparing with builtin
    Mat canny_builtin;
    Canny(grey,canny_builtin,20,50);

    imwrite("data/results/canny_manual.jpg",edges);
    imwrite("data/results/canny_builtin.jpg",canny_builtin);

}
int main(){
    cannyedge("data/raw/sae.jpg");
}
