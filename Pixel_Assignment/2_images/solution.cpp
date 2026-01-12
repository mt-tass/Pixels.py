#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    Mat img = imread("data/raw/img.jpg", IMREAD_COLOR);

    Mat img1 = img.clone();
    Point topLeft(100, 100);
    Point bottomRight(100 + 200, 100 + 150);
    Scalar red(0, 0, 255);
    rectangle(img1, topLeft, bottomRight, red, FILLED);
    imwrite("data/results/filled_rectangle.jpg", img1);

    Mat img2 = img.clone();
    Point center(250, 200);
    int radius = 80;
    Scalar green(0, 255, 0);
    circle(img2, center, radius, green, FILLED);
    imwrite("data/results/output_filled_circle.jpg", img2);
    
    Mat img3 = img.clone();
    Point hollowTopLeft(100, 100);
    Point hollowBottomRight(300, 250);
    Scalar blue(255, 0, 0);
    int thickness = 10;
    rectangle(img3, hollowTopLeft, hollowBottomRight, blue, thickness);
    imwrite("data/results/output_hollow_rectangle.jpg", img3);
    
    Mat img4 = img.clone();
    Point circleCenter(250, 200);
    int circleRadius = 80;
    Scalar yellow(0, 255, 255);
    int circleThickness = 10;
    circle(img4, circleCenter, circleRadius, yellow, circleThickness);
    imwrite("data/results/output_hollow_circle.jpg", img4);
    Mat rotated180;
    rotate(img, rotated180, ROTATE_180);
    imwrite("data/results/output_rotated180.jpg", rotated180);
    
    Point2f centerPoint(img.cols / 2.0, img.rows / 2.0);
    double rotationAngle = 45.0;
    double scale = 1.0;
    Mat rotationMatrix = getRotationMatrix2D(centerPoint, rotationAngle, scale);
    Mat rotated45NonBound;
    warpAffine(img, rotated45NonBound, rotationMatrix, Size(img.cols, img.rows));
    imwrite("data/results/output_rotated45nb.jpg", rotated45NonBound);
    
    Mat rotationMatrixBound = getRotationMatrix2D(centerPoint, rotationAngle, scale);
    double radians = rotationAngle/180.0 *CV_PI;
    double sine = abs(sin(radians));
    double cosine = abs(cos(radians));
    int newWidth = int(img.cols * cosine + img.rows * sine);
    int newHeight = int(img.cols * sine + img.rows * cosine);
    rotationMatrixBound.at<double>(0, 2) += (newWidth / 2.0) - centerPoint.x;
    rotationMatrixBound.at<double>(1, 2) += (newHeight / 2.0) - centerPoint.y;
    Mat rotated45Bound;
    warpAffine(img, rotated45Bound, rotationMatrixBound, Size(newWidth, newHeight));
    imwrite("data/results/output_rotated45b.jpg", rotated45Bound);
    return 0;
}