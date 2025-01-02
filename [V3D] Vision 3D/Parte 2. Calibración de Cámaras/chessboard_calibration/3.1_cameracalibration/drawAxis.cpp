#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{5,8};

int main(int argc, char *argv[])
{
    // Calibrate the camera 
    // that recorded the video file provided using the images in folder calibration.

    // Usamos el programa calibration.cpp y obtenemos calibration.yml
    /*
    cameraMatrix : [1072.37152145942, 0, 643.7486938581734;
    0, 1071.776435622937, 353.0166412537565;
    0, 0, 1]
    distCoeffs : [-0.0103710418999946, -0.09644489389769062, -0.003044509756054933, 0.0009581619215796805, 0.08919948044559256]
    Rotation vector : [-0.2581138351042582, -0.7761668886073747, 1.324128595184926;
    -0.02060760577291612, -0.5750696255781069, 1.403739028896906;
    0.3791716707104024, -0.1561729870303938, 1.506069485422278;
    0.6634402576725924, 0.04153942230335741, 1.521590594443938;
    0.6999901832504787, -0.4356081597677096, 1.517084093679889;
    0.3037701128909273, -0.4286393050145727, 1.515596359283339;
    0.1816155917566033, -0.3289416015734458, 1.535600944952157]
    Translation vector : [-0.1038763118172926, -4.684784269063331, 23.85617750128505;
    3.19156723276491, -2.647390851579323, 17.7147770063127;
    2.799498336477862, -1.583943371126268, 14.97516328495779;
    2.36474937715967, -1.402479023744577, 13.7200148294385;
    4.095897959884276, -0.5431225625970644, 13.95446172330613;
    5.213096791712922, -1.034783143048469, 12.01546507044648;
    4.482163152407296, -1.972006388726797, 9.701824606527779]
    */

    int size = std::stoi(argv[1]); // tamaño de los ejes que se van a dibujar
    std::string calibration_yml = argv[2]; // ruta del fichero de calibracion
    std::string input_video = argv[3]; //input video sobre el que se va a dibujar

    // Cargamos el fichero de calibracion "calibration.yml"
    cv::FileStorage fs(calibration_yml, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "Failed to open file: " << calibration_yml << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix, distCoeffs;
    fs["cameraMatrix"] >> cameraMatrix;
    fs["distCoeffs"] >> distCoeffs;
    fs.release();

    // Creamos el vector de puntos 3D que se van a dibujar sobre el tablero
    std::vector<cv::Point3f> objp;
    for (int i = 0; i < CHECKERBOARD[1]; i++)
    {
        
        for (int j = 0; j < CHECKERBOARD[0]; j++){
            
            objp.push_back(cv::Point3f(j, i, 0));
        }
    }

    // Creamos el objeto de video
    cv::VideoCapture cap(input_video);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video file or camera" << std::endl;
        return -1;
    }

    cv::Mat frame, gray;

    // Obtiene el tamaño del cuadro y la tasa de fotogramas
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    // Define el codec y crea el objeto VideoWriter
cv::VideoWriter video_writer(
    "video_salida.mp4",
    cv::VideoWriter::fourcc('a', 'v', 'c', '1'), // H.264 para MP4
    fps,
    cv::Size(frame_width, frame_height)
);

    if (!video_writer.isOpened()) {
        std::cerr << "Error: No se pudo crear el video de salida.\n";
        return -1;
    }


    // Para cada frame del video:
        // 1. Detectar el tablero con cv::findChessBoardCorners y refinar las
        //    esquinas con cv::cornerSubPix

        // 2. Estimar la posicion de la camara con respecto al tablero usando
        //    cv::solvePnP

        // 3, Dibujar los ejes utilizando cv::projectPoints y cv::line

    while(cap.read(frame)){
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        // Detect the board
        std::vector<cv::Point2f> corner_pts;
        bool success = cv::findChessboardCorners(gray, cv::Size(CHECKERBOARD[0], CHECKERBOARD[1]), corner_pts, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (success)
        {
            // Refinamos las esquinas
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);
            cv::cornerSubPix(gray, corner_pts, cv::Size(11, 11), cv::Size(-1, -1), criteria);

            // Estimamos la posicion de la camara respecto al tablero usando solvePnP
            cv::Mat rvec, tvec;
            cv::solvePnP(objp, corner_pts, cameraMatrix, distCoeffs, rvec, tvec);

            // Proyectamos los ejes
            std::vector<cv::Point3f> axisPoints{
                cv::Point3f(0, 0, 0),
                cv::Point3f(size, 0, 0),    // X
                cv::Point3f(0, size, 0),    // Y
                cv::Point3f(0, 0, -size)};   // Z

            std::vector<cv::Point2f> imagePoints;
            cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

            cv::line(frame, corner_pts[0], imagePoints[1], cv::Scalar(0, 0, 255), 5); // X-axis (red)
            cv::line(frame, corner_pts[0], imagePoints[2], cv::Scalar(0, 255, 0), 5); // Y-axis (green)
            cv::line(frame, corner_pts[0], imagePoints[3], cv::Scalar(255, 0, 0), 5); // Z-axis (blue)
        }


        video_writer.write(frame);
        cv::imshow("Camera Calibration (Axis)", frame);
        

        if (cv::waitKey(10) == 27) // Press ESC to exit
            break;
    }

    cap.release();
    video_writer.release();
    cv::destroyAllWindows();

    return 0;

}