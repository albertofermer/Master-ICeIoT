#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
using namespace std;

int CHECKERBOARD[2]{7, 5};
double SquareSize = 0.02875; // size of each square

int main(int argc, char *argv[])
{

    // vector de puntos 3d
    std::vector<std::vector<cv::Point3f>> objpoints;

    // vectores de puntos 2D
    std::vector<std::vector<cv::Point2f>> imgpointsLeft;
    std::vector<std::vector<cv::Point2f>> imgpointsRight;

    // Coordenadas del mundo en 3D
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++)
    {
        for (int j = 0; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j * SquareSize, i * SquareSize, 0));
    }

    // Extraemos el path de las imagenes
    std::vector<cv::String> images;
    std::string path = std::string(argv[1]) + "/*.jpg";

    cv::glob(path, images);

    cv::Mat frame, gray;
    // coordenadas pixel detectadas por el tablero
    std::vector<cv::Point2f> corner_ptsLeft, corner_ptsRight;
    bool successRight, successLeft;
    // criteria
    cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 60, 1e-6);
    // Itera sobre las imagenes
    for (size_t i = 0; i < images.size(); i++)
    {
        std::cout << "Reading: " << images[i] << std::endl;

        // Lee la iamgen
        frame = cv::imread(images[i]);
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // Dividimos la imagen en dos mitades
        cv::Mat left = frame.colRange(0, frame.cols / 2);
        cv::Mat right = frame.colRange(frame.cols / 2, frame.cols);

        successRight = false;
        successLeft = false;

        // Procesamos la imagen de la izquierda
        cv::Mat grayLeft;
        cv::cvtColor(left, grayLeft, cv::COLOR_BGR2GRAY);
        successLeft = cv::findChessboardCorners(grayLeft, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsLeft, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        // Si ambos son correctas entonces tenemos las esquinas del tablero
        if (successLeft){

            cv::cornerSubPix(grayLeft, corner_ptsLeft, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cv::drawChessboardCorners(left, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsLeft, true);

            imgpointsLeft.push_back(corner_ptsLeft);
        }

        // Procesamos la imagen de la derecha
        cv::Mat grayRight;
        cv::cvtColor(right, grayRight, cv::COLOR_BGR2GRAY);
        successRight = cv::findChessboardCorners(grayRight, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsRight, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (successRight){

            cv::cornerSubPix(grayRight, corner_ptsRight, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            cv::drawChessboardCorners(right, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_ptsRight, true);

            imgpointsRight.push_back(corner_ptsRight);
        }

        objpoints.push_back(objp);

        // Resize image to fit into screen
        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(1920, 1080));
        cv::imshow("Image", resized);
        int key = 0;
        while (key != 13 && key != 27)
            key = cv::waitKey(0) & 0xff;
        cv::waitKey(27);
    }

    cv::destroyAllWindows();

    // Parametros intrinsecos de la camara
    cv::Mat cameraMatrixLeft, distCoeffsLeft, RLeft, TLeft;
    cv::Mat cameraMatrixRight, distCoeffsRight, RRight, TRight;

    // Calibracion de la camara izquierda
    cv::calibrateCamera(objpoints, imgpointsLeft, cv::Size(gray.cols / 2, gray.rows), cameraMatrixLeft, distCoeffsLeft, RLeft, TLeft);

    // PCalibracion de la camara derecha
    cv::calibrateCamera(objpoints, imgpointsRight, cv::Size(gray.cols / 2, gray.rows), cameraMatrixRight, distCoeffsRight, RRight, TRight);

    // Parametros extrinsecos de la camara
    cv::Mat R, T, E, F, perViewErrors;

    int	flags = 0;
    cv::stereoCalibrate(objpoints, imgpointsLeft, imgpointsRight,
                        cameraMatrixLeft, distCoeffsLeft,
                        cameraMatrixRight, distCoeffsRight,
                        cv::Size(gray.rows, gray.cols / 2), R, T, E, F,perViewErrors, flags ,criteria);

    std::cout << "LEFT_K: " << cameraMatrixLeft << std::endl;
    std::cout << "LEFT_D: " << distCoeffsLeft << std::endl;
    std::cout << "RIGHT_K : " << cameraMatrixRight << std::endl;
    std::cout << "RIGHT_D : " << distCoeffsRight << std::endl;
    std::cout << "R : " << R << std::endl;
    std::cout << "T : " << T << std::endl;
    std::cout << "E : " << E << std::endl;
    std::cout << "F : " << F << std::endl;

    // guardamos los parametros de calibracion en yml
    cv::FileStorage fs("stereoparms.yml", cv::FileStorage::WRITE);
    fs << "LEFT_K" << cameraMatrixLeft << "LEFT_D" << distCoeffsLeft 
       << "RIGHT_K" << cameraMatrixRight << "RIGHT_D" << distCoeffsRight
       << "R" << R << "T" << T << "E" << E << "F" << F;

    return 0;
}