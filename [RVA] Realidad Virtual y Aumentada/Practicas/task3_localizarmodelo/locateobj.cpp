// This program reads a model image and a scene image, and finds the bounding-box of the model in the scene

// (c) Realidad Virtual y Aumentada - Universidad de Cordoba - Manuel J. Marin-Jimenez

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include "rva.h"

using namespace std;

// Main function
int main(int argc, char ** argv) {

    // Get the arguments: model and scene images path using OpenCV
    cv::CommandLineParser parser(argc, argv, "{@model | model.jpg | input model image}{@scene | scene.jpg | input scene image}{@descriptor | ORB | descriptor type (ORB, SIFT, AKAZE)}");
    cv::String model_path = parser.get<cv::String>(0);
    cv::String scene_path = parser.get<cv::String>(1);
    cv::String descriptor_type = parser.get<cv::String>(2);
    std::string descriptor_type_str = descriptor_type;
    std::transform(descriptor_type_str.begin(), descriptor_type_str.end(), descriptor_type_str.begin(), ::toupper);


    // Load the images
    cv::Mat img_model = cv::imread(model_path, cv::IMREAD_GRAYSCALE);
    cv::Mat img_scene = cv::imread(scene_path, cv::IMREAD_GRAYSCALE);

    // Check if the images are loaded
    if (img_model.empty() || img_scene.empty()) {
        cout << "Error: images not loaded" << endl;
        return -1;
    }

    // Compute keypoints and descriptors for the model image
    std::vector<cv::KeyPoint> keypoints_model;
    cv::Mat descriptors_model;
    rva_calculaKPsDesc(img_model, keypoints_model, descriptors_model, descriptor_type_str);

    // Compute keypoints and descriptors for the scene image
    std::vector<cv::KeyPoint> keypoints_scene;
    cv::Mat descriptors_scene;
    rva_calculaKPsDesc(img_scene, keypoints_scene, descriptors_scene, descriptor_type_str);

    // Match the descriptors
    std::vector<cv::DMatch> matches;
    rva_matchDesc(descriptors_model, descriptors_scene, matches, descriptor_type_str);

    // Compute the bounding-box of the model in the scene
    cv::Mat H;
    std::vector<cv::Point2f> pts_obj_in_scene;
    rva_localizaObj(img_model, img_scene, keypoints_model, keypoints_scene, matches, H, pts_obj_in_scene);

    // Draw the bounding-box on the color image    
    cv::Mat img_scene_col = cv::imread(scene_path, cv::IMREAD_COLOR);
    rva_draw_contour(img_scene_col, pts_obj_in_scene, cv::Scalar(0, 255, 0), 4);

    // Show the scene
    // Obtener el tamaño de la pantalla (ajusta estos valores según tu resolución)
    int screen_width = 1280;  // Ajusta según tu pantalla
    int screen_height = 720;

    // Obtener el tamaño actual de la imagen
    int img_width = img_scene_col.cols;
    int img_height = img_scene_col.rows;

    // Calcular el factor de escalado manteniendo la proporción
    double scale_x = (double)screen_width / img_width;
    double scale_y = (double)screen_height / img_height;
    double scale = std::min(scale_x, scale_y); // Escalar para que encaje en la pantalla

    // Redimensionar la imagen
    cv::Mat img_resized;
    cv::resize(img_scene_col, img_resized, cv::Size(), scale, scale);

    // Mostrar la imagen redimensionada
    std::string filename = cv::format("../Data/%s_locate_result.jpg", descriptor_type.c_str());
    cv::imwrite(filename, img_resized);
    cv::imshow("Detected", img_resized);
    cv::waitKey(0);

    return 0;
}
