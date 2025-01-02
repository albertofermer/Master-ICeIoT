#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{5, 8};

void drawCube(cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec, const cv::Mat &cameraMatrix, const cv::Mat &distCoeffs, int size, int row, int col)
{
    float cubeSize = 1.0f;
    // Definimos los vertices del cubo
    std::vector<cv::Point3f> cubePoints{
        cv::Point3f(cubeSize * row, cubeSize * col, 0),                               // base
        cv::Point3f(cubeSize + cubeSize * row, cubeSize * col, 0),                    // base
        cv::Point3f(cubeSize + cubeSize * row, cubeSize + cubeSize * col, 0),         // base
        cv::Point3f(cubeSize * row, cubeSize + cubeSize * col, 0),                    // base
        cv::Point3f(cubeSize * row, cubeSize * col, -cubeSize),                       // top
        cv::Point3f(cubeSize + cubeSize * row, cubeSize * col, -cubeSize),            // top
        cv::Point3f(cubeSize + cubeSize * row, cubeSize + cubeSize * col, -cubeSize), // top
        cv::Point3f(cubeSize * row, cubeSize + cubeSize * col, -cubeSize)};           // top

    // Project cube points onto the image plane
    std::vector<cv::Point2f> imagePoints;
    cv::projectPoints(cubePoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

    // Draw the cube
    for (int i = 0; i < 4; ++i)
    {
        cv::line(frame, imagePoints[i], imagePoints[(i + 1) % 4], cv::Scalar(255, 0, 0), 1);         // Base del cubo
        cv::line(frame, imagePoints[i + 4], imagePoints[(i + 1) % 4 + 4], cv::Scalar(255, 0, 0), 1); // Top del cubo
        cv::line(frame, imagePoints[i], imagePoints[i + 4], cv::Scalar(255, 0, 0), 1);               // conexiones entre la base y la tapa del cubo
    }
}

int main(int argc, char *argv[])
{
    // Usamos el programa calibration y obtenemos calibration.yml

    int size = 1;                          // tamaño de la arista de los cubos
    std::string calibration_yml = argv[2]; // ruta del fichero de calibracion
    std::string input_video = argv[3];     // input video sobre el que se va a dibujar

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
        for (int j = 0; j < CHECKERBOARD[0]; j++)
            objp.push_back(cv::Point3f(j, i, 0));
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
        "../../video/video_salida_cubos.mp4",
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'), // H.264 para MP4
        fps,
        cv::Size(frame_width, frame_height));

    if (!video_writer.isOpened())
    {
        std::cerr << "Error: No se pudo crear el video de salida.\n";
        return -1;
    }

    // Para cada frame del video:
    // 1. Detectar el tablero con cv::findChessBoardCorners y refinar las
    //    esquinas con cv::cornerSubPix

    // 2. Estimar la posicion de la camara con respecto al tablero usando
    //    cv::solvePnP

    // 3, Dibujar los ejes utilizando cv::projectPoints y cv::line

    while (cap.read(frame))
    {
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
                cv::Point3f(size, 0, 0),   // X
                cv::Point3f(0, size, 0),   // Y
                cv::Point3f(0, 0, -size)}; // Z

            std::vector<cv::Point2f> imagePoints;
            cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

            cv::line(frame, corner_pts[0], imagePoints[1], cv::Scalar(0, 0, 255), 5); // X-axis (red)
            cv::line(frame, corner_pts[0], imagePoints[2], cv::Scalar(0, 255, 0), 5); // Y-axis (green)
            cv::line(frame, corner_pts[0], imagePoints[3], cv::Scalar(255, 0, 0), 5); // Z-axis (blue)

            for (int row = 0; row < CHECKERBOARD[0]; row++)
            {
                for (int col = 0; col < CHECKERBOARD[1]; col++)
                {
                    if ((row + col) % 2 == 0) // los cuadrados negros son si la suma es par
                        drawCube(frame, rvec, tvec, cameraMatrix, distCoeffs, size, row, col);
                }
            }
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