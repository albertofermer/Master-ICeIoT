#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
using namespace std;

//Structure that contains the Stereo Pair Calbration information.
//This will be calculated using stereo_calibrate
struct StereoParams{
   cv::Mat mtxL,distL,R_L,T_L;
   cv::Mat mtxR,distR,R_R,T_R;
   cv::Mat Rot, Trns, Emat, Fmat;
};

void rectifyStereoImages(const StereoParams &sti,cv::Mat &left,cv::Mat &rigth){
   cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;
   cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
   cv::Mat Right_Stereo_Map1, Right_Stereo_Map2;
   cv::stereoRectify(sti.mtxL, sti.distL,sti.mtxR,sti.distR,left.size(),sti.Rot,sti.Trns,
                     rect_l,rect_r,proj_mat_l,proj_mat_r,
                     Q,cv::CALIB_ZERO_DISPARITY, 0);
   cv::initUndistortRectifyMap(sti.mtxL,sti.distL,rect_l,proj_mat_l,
                               left.size(),CV_16SC2,
                               Left_Stereo_Map1,Left_Stereo_Map2);
   cv::initUndistortRectifyMap(sti.mtxR,sti.distR,
                               rect_r,proj_mat_r,
                               left.size(),CV_16SC2,
                               Right_Stereo_Map1,Right_Stereo_Map2);
   cv::Mat AuxImage, Right_nice;
   cv::remap(left, AuxImage, Left_Stereo_Map1, Left_Stereo_Map2,
             cv::INTER_LANCZOS4,cv::BORDER_CONSTANT,0);
   AuxImage.copyTo(left);
   cv::remap(rigth,  AuxImage,  Right_Stereo_Map1,  Right_Stereo_Map2,
             cv::INTER_LANCZOS4,cv::BORDER_CONSTANT,0);
   AuxImage.copyTo(rigth);
}

// Global variables for mouse callback
cv::Mat originalDisplay, rectifiedDisplay;
int mouseY = -1;

// Mouse callback for drawing dynamic horizontal line
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_MOUSEMOVE) {
        mouseY = y;

        // Redibuja la imagen original con una linea en y
        cv::Mat originalCopy = originalDisplay.clone();
        cv::line(originalCopy, cv::Point(0, mouseY), cv::Point(originalCopy.cols, mouseY), cv::Scalar(0, 0, 255), 1);
        cv::imshow("Original Images", originalCopy);

        // Vuelve a dibujar la imagen rectificada con una linea en y
        cv::Mat rectifiedCopy = rectifiedDisplay.clone();
        cv::line(rectifiedCopy, cv::Point(0, mouseY), cv::Point(rectifiedCopy.cols, mouseY), cv::Scalar(0, 0, 255), 1);
        cv::imshow("Rectified Images", rectifiedCopy);
    }
}

int main(int argc, char *argv[]){
    if (argc != 3) {
        std::cerr << "Usage: ./stereo_checkundistorted stereo_image.jpg stereocalibrationfile.yml\n";
        return -1;
    }

    std::string stereoImagePath = argv[1];
    std::string calibrationFile = argv[2];

    cv::Mat stereoImg = cv::imread(stereoImagePath);

    // Split stereo image into left and right
    cv::Mat leftImg = stereoImg(cv::Rect(0, 0, stereoImg.cols / 2, stereoImg.rows)).clone();
    cv::Mat rightImg = stereoImg(cv::Rect(stereoImg.cols / 2, 0, stereoImg.cols / 2, stereoImg.rows)).clone();

    // Parametros de calibracion
    StereoParams params;
    cv::FileStorage fs(calibrationFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error abriendo el fichero de calibracion" << endl;
        return -1;
    }

    fs["LEFT_K"] >> params.mtxL;
    fs["LEFT_D"] >> params.distL;
    fs["RIGHT_K"] >> params.mtxR;
    fs["RIGHT_D"] >> params.distR;
    fs["R"] >> params.Rot;
    fs["T"] >> params.Trns;
    fs["E"] >> params.Emat;
    fs["F"] >> params.Fmat;

    fs.release();

    // Display original images
    cv::hconcat(leftImg, rightImg, originalDisplay);
    cv::imshow("Original Images", originalDisplay);

    // Rectify images
    rectifyStereoImages(params, leftImg, rightImg);

    // Display rectified images
    cv::hconcat(leftImg, rightImg, rectifiedDisplay);
    cv::imshow("Rectified Images", rectifiedDisplay);

    // Set mouse callback
    cv::setMouseCallback("Original Images", onMouse, nullptr);
    cv::setMouseCallback("Rectified Images", onMouse, nullptr);

    cv::waitKey(0);

    return 0;
}