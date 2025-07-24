#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;
// Stereo parameters structure
struct StereoParams {
    cv::Mat mtxL, distL, R_L, T_L;
    cv::Mat mtxR, distR, R_R, T_R;
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

void writeToOBJ(std::string path,std::vector<cv::Point3f> points){

   std::ofstream file(path,std::ios::binary);
   for(auto p:points)
       file<<"v "<<p.x<<" "<<p.y<<" "<<p.z<<endl;
}


int main(int argc, char *argv[]){
    
    if (argc != 4) {
        cerr << "Usage: stereo_disparity stereo_image.jpg calibration.yml out.obj" << std::endl;
        return -1;
    }
    // 1. Cargar las imagenes stereo
    cv::Mat stereoImage = cv::imread(argv[1]);

    if (stereoImage.empty()) {
        cerr << "Error: Could not read the stereo image." << std::endl;
        return -1;
    }

    // 2. Rectificar imagenes

    // Carga de parametros
    StereoParams stParams;
    cv::FileStorage fs(argv[2], cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Error: No se pudo abrir el fichero de calibracion." << std::endl;
        return -1;
    }

    fs["LEFT_K"] >> stParams.mtxL;
    fs["LEFT_D"] >> stParams.distL;
    fs["RIGHT_K"] >> stParams.mtxR;
    fs["RIGHT_D"] >> stParams.distR;
    fs["R"] >> stParams.Rot;
    fs["T"] >> stParams.Trns;
    fs["E"] >> stParams.Emat;
    fs["F"] >> stParams.Fmat;
    
    cv::Mat leftRectified = stereoImage.colRange(0, stereoImage.cols / 2).clone();
    cv::Mat rightRectified = stereoImage.colRange(stereoImage.cols / 2, stereoImage.cols).clone();

    rectifyStereoImages(stParams, leftRectified, rightRectified);

    // Convertimos las imagenes a escala de grises para usar la funcion StereoBM
    cv::cvtColor(leftRectified, leftRectified, cv::COLOR_BGR2GRAY);
    cv::cvtColor(rightRectified, rightRectified, cv::COLOR_BGR2GRAY);

    // 3. Calculamos la disparidad con StereoSB
    cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create(16, 15); // numDisparities, blockSize
    cv::Mat disp;
    stereo->compute(leftRectified, rightRectified, disp);
    // 4. Converting disparity values to CV_32F from CV_16S
    disp.convertTo(disp,CV_32F, 1.0);
    disp=disp/16.f;

    // 5. Triangulacion  Z= |T|*f /d; X= (x-cx)*Z/f ; Y= (y-cy)*Z/f;

    float fx = stParams.mtxL.at<double>(0, 0);
    float fy = stParams.mtxL.at<double>(1, 1);
    float cx = stParams.mtxL.at<double>(0, 2);
    float cy = stParams.mtxL.at<double>(1, 2);
    float baseline = cv::norm(stParams.Trns);

    std::vector<cv::Point3f> points;
    for (int row = 0; row < disp.rows; row++) {
        for (int col = 0; col < disp.cols; col++) {
            float d = disp.at<float>(row, col);
            // Para aquellos puntos con disparidad > 10
            if (d > 10.0f) {
                float Z = (fx * baseline) / d;
                float X = (col - cx) * Z / fx;
                float Y = (row - cy) * Z / fy;
                points.emplace_back(X, Y, Z);
            }
        }
    }

    // 6. Guarde los puntos en obj
    writeToOBJ(argv[3], points);
    return 0;
}