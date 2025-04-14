// (c) Realidad Virtual y Aumentada - Universidad de Cordoba - Manuel J. Marin-Jimenez

#include <iostream>
#include <opencv2/opencv.hpp>

#include "rva.h"

// CREA TUS FUNCIONES AQUI


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