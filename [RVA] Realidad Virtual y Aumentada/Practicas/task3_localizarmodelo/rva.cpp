// (c) Realidad Virtual y Aumentada - Universidad de Cordoba - Manuel J. Marin-Jimenez

#include <iostream>
#include <opencv2/opencv.hpp>

#include "rva.h"


// TASK 1

cv::Mat rva_compute_homography(std::vector<cv::Point2f> points_image1, std::vector<cv::Point2f> points_image2) {
    // Verificar que haya exactamente 4 puntos en cada conjunto
    if (points_image1.size() != 4 || points_image2.size() != 4) {
        std::cerr << "Error: Se requieren exactamente 4 puntos en cada imagen para calcular la homografía." << std::endl;
        return cv::Mat();
    }

    // Calcular la matriz de homografía usando el método Direct Linear Transform (DLT)
    cv::Mat homography = cv::findHomography(points_image1, points_image2, cv::RANSAC);

    // Verificar si la homografía se calculó correctamente
    if (homography.empty()) {
        std::cerr << "Error: No se pudo calcular la matriz de homografía." << std::endl;
    }

    return homography;
}


void rva_draw_contour(cv::Mat image, std::vector<cv::Point2f> points, cv::Scalar color, int thickness) {
    // Verificar que haya al menos 4 puntos para dibujar el contorno
    if (points.size() < 4) {
        std::cerr << "Error: Se requieren al menos 4 puntos para dibujar un contorno." << std::endl;
        return;
    }

    // Dibujar líneas entre los puntos en orden
    for (size_t i = 0; i < points.size(); i++) {
        cv::line(image, points[i], points[(i + 1) % points.size()], color, thickness);
    }
}


void rva_deform_image(const cv::Mat& im_input, cv::Mat& im_output, cv::Mat homography) {
    // Verificar que la homografía no esté vacía
    if (homography.empty()) {
        std::cerr << "Error: La matriz de homografía está vacía. No se puede deformar la imagen." << std::endl;
        return;
    }

    // Aplicar la transformación de perspectiva
    cv::warpPerspective(im_input, im_output, homography, im_output.size());
}


// TASK 2

void rva_calculaKPsDesc(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, const std::string &type)
{
    cv::Ptr<cv::Feature2D> detector;

    if (type == "ORB") {
        detector = cv::ORB::create();
    } else if (type == "SIFT") {
        detector = cv::SIFT::create();
    } else if (type == "AKAZE") {
        detector = cv::AKAZE::create();
    } else {
        std::cerr << "Tipo de detector no reconocido: " << type << std::endl;
        return;
    }

    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
}

void rva_matchDesc(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches, const std::string &type)
{
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    if (type == "SIFT") {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    } else {
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    }

    matcher->match(descriptors1, descriptors2, matches);
}

void rva_dibujaMatches(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches, cv::Mat &img_matches)
{
    // Dibujar los matches
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
}


// TASK 3

void rva_localizaObj(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches, cv::Mat &homography, std::vector<cv::Point2f> &pts_im2)
{
    // Paso 1: Detectar los keypoints y calcular los descriptores
    cv::Mat descriptors1, descriptors2;
    rva_calculaKPsDesc(img1, keypoints1, descriptors1, "AKAZE");
    rva_calculaKPsDesc(img2, keypoints2, descriptors2, "AKAZE");

    // Paso 2: Coincidir los descriptores
    rva_matchDesc(descriptors1, descriptors2, matches, "AKAZE");

    // Paso 3: Filtrar los matches utilizando la prueba de Lowe (relación de distancia)
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < matches.size() - 1; i++) {  // Cambio aquí: evitar acceso fuera de rango
        if (matches[i].distance < ratio_thresh * matches[i + 1].distance) {
            good_matches.push_back(matches[i]);
        }
    }

    // Paso 5: Localizar el objeto en la imagen 2 usando la homografía
    std::vector<cv::Point2f> obj_points;
    std::vector<cv::Point2f> scene_points;

    // Obtener los puntos correspondientes de los matches
    for (size_t i = 0; i < good_matches.size(); i++) {
        obj_points.push_back(keypoints1[good_matches[i].queryIdx].pt);
        scene_points.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }

    // Calcular la homografía entre las imágenes
    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
 
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }
    homography = findHomography( obj, scene, cv::RANSAC );

    // Paso 6: Definir las esquinas del objeto (modelo) en img1
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point2f(0, 0);
    obj_corners[1] = cv::Point2f( (float)img1.cols, 0 );
    obj_corners[2] = cv::Point2f( (float)img1.cols, (float)img1.rows );
    obj_corners[3] = cv::Point2f( 0, (float)img1.rows );

    std::vector<cv::Point2f> scene_corners(4);
    cv::perspectiveTransform(obj_corners, scene_corners, homography);

    // Paso 9: Almacenar las coordenadas de las esquinas transformadas
    pts_im2 = scene_corners;
}


