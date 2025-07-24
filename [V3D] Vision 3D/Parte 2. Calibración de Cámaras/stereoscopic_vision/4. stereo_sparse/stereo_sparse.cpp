#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
using namespace std;

struct StereoParams {
    cv::Mat mtxL, distL, R_L, T_L;
    cv::Mat mtxR, distR, R_R, T_R;
    cv::Mat Rot, Trns, Emat, Fmat;
};

void rectifyStereoImages(const StereoParams &sti,cv::Mat &left,cv::Mat &rigth){
   cv::Mat rect_l, rect_r, proj_mat_l, proj_mat_r, Q;
   cv::Mat Left_Stereo_Map1, Left_Stereo_Map2;
   cv::Mat right_Stereo_Map1, right_Stereo_Map2;
   cv::stereoRectify(sti.mtxL, sti.distL,sti.mtxR,sti.distR,left.size(),sti.Rot,sti.Trns,
                     rect_l,rect_r,proj_mat_l,proj_mat_r,
                     Q,cv::CALIB_ZERO_DISPARITY, 0);
   cv::initUndistortRectifyMap(sti.mtxL,sti.distL,rect_l,proj_mat_l,
                               left.size(),CV_16SC2,
                               Left_Stereo_Map1,Left_Stereo_Map2);
   cv::initUndistortRectifyMap(sti.mtxR,sti.distR,
                               rect_r,proj_mat_r,
                               left.size(),CV_16SC2,
                               right_Stereo_Map1,right_Stereo_Map2);
   cv::Mat AuxImage, right_nice;
   cv::remap(left, AuxImage, Left_Stereo_Map1, Left_Stereo_Map2,
             cv::INTER_LANCZOS4,cv::BORDER_CONSTANT,0);
   AuxImage.copyTo(left);
   cv::remap(rigth,  AuxImage,  right_Stereo_Map1,  right_Stereo_Map2,
             cv::INTER_LANCZOS4,cv::BORDER_CONSTANT,0);
   AuxImage.copyTo(rigth);
}

void writeToOBJ(std::string path,std::vector<cv::Point3f> points){
    std::ofstream file(path,std::ios::binary);
    for(auto p:points)
        file<<"v "<<p.x<<" "<<p.y<<" "<<p.z<<std::endl;
}

int main(int argc, char *argv[]){

    if (argc != 4) {
        std::cerr << "Usage: stereo_sparse stereo_image.jpg calibration.yml out.obj" << std::endl;
        return -1;
    }

    // leer fichero de calibracion 
    StereoParams stParams;
    cv::FileStorage fs(argv[2], cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: No se pudo abrir el fichero de calibracion." << std::endl;
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
    // Leer imagenes estereo
    cv::Mat stereoImage = cv::imread(argv[1]);

    if (stereoImage.empty()) {
        std::cerr << "Error: No se pudo leer la imagen." << std::endl;
        return -1;
    }

    // Rectificar imagenes
    cv::Mat left = stereoImage.colRange(0, stereoImage.cols / 2).clone();
    cv::Mat right = stereoImage.colRange(stereoImage.cols / 2, stereoImage.cols).clone();

    rectifyStereoImages(stParams, left, right);

    // Convertir a escala de grises
    cv::cvtColor(left, left, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right, cv::COLOR_BGR2GRAY);

    //Busca keypoints en ambas imágenes usando AKAZE y el descriptor matcher BruteForce-Hamming
    std::vector<cv::KeyPoint> keypoints_query, keypoints_train;
    cv::Mat descriptors_query, descriptors_train;
    auto Detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4f, 8);
    Detector->detectAndCompute(left, cv::Mat(), keypoints_query, descriptors_query);
    Detector->detectAndCompute(right, cv::Mat(), keypoints_train, descriptors_train);

    // Match descriptors using BruteForce-Hamming
    std::vector<cv::DMatch> matches;
    auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    matcher->match(descriptors_query, descriptors_train, matches, cv::Mat());

    // Filtrar los matches horizontales
    std::vector<cv::DMatch> filtered_matches;
    for (const auto &match : matches) {
        // Filtrado basado en lineas horizontales
        // Comprobar la coordenada y de los matches
        // mantener los matches que estan cercanos verticalmente
        if (std::abs(keypoints_query[match.queryIdx].pt.y - keypoints_train[match.trainIdx].pt.y) < 10.0) {
            filtered_matches.push_back(match);
        }
    }

    // Dibujar matches
    cv::Mat img_matches_antes, img_matches_despues;
    cv::drawMatches(left, keypoints_query, right, keypoints_train, matches, img_matches_antes);
    cv::drawMatches(left, keypoints_query, right, keypoints_train, filtered_matches, img_matches_despues);


    // Mostrar matches
    cv::imshow("Matches Antes del Filtrado", img_matches_antes);
    cv::imshow("Matches Después del Filtrado", img_matches_despues);
    cv::waitKey(0);

    // Triangula los matches y guardar en obj (Z= |T|*f /d; X= (x-cx)*Z/f ; Y= (y-cy)*Z/f;)
    std::vector<cv::Point3f> points;
    for (const auto &match : filtered_matches) {
        float d = match.distance; 
        float Z = stParams.Trns.at<double>(0, 0) * stParams.mtxL.at<double>(0, 0) / d;
        float X = (keypoints_query[match.queryIdx].pt.x - stParams.mtxL.at<double>(0, 2)) * Z / stParams.mtxL.at<double>(0, 0);
        float Y = (keypoints_query[match.queryIdx].pt.y - stParams.mtxL.at<double>(1, 2)) * Z / stParams.mtxL.at<double>(1, 1);
        points.push_back(cv::Point3f(X, Y, Z));
    }

    // Save in OBJ object
    writeToOBJ(argv[3], points);

    return 0;
}