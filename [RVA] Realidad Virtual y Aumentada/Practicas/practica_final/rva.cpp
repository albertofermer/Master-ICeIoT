// (c) Realidad Virtual y Aumentada - Universidad de Cordoba - Manuel J. Marin-Jimenez

#include <iostream>
#include <opencv2/opencv.hpp>

#include "rva.h"

// TASK 1

cv::Mat rva_compute_homography(std::vector<cv::Point2f> points_image1, std::vector<cv::Point2f> points_image2)
{
    // Verificar que haya exactamente 4 puntos en cada conjunto
    if (points_image1.size() != 4 || points_image2.size() != 4)
    {
        std::cerr << "Error: Se requieren exactamente 4 puntos en cada imagen para calcular la homografía." << std::endl;
        return cv::Mat();
    }

    // Calcular la matriz de homografía usando el método Direct Linear Transform (DLT)
    cv::Mat homography = cv::findHomography(points_image1, points_image2, cv::RANSAC);

    // Verificar si la homografía se calculó correctamente
    if (homography.empty())
    {
        std::cerr << "Error: No se pudo calcular la matriz de homografía." << std::endl;
    }

    return homography;
}

void rva_draw_contour(cv::Mat image, std::vector<cv::Point2f> points, cv::Scalar color, int thickness)
{
    // Verificar que haya al menos 4 puntos para dibujar el contorno
    if (points.size() < 4)
    {
        std::cerr << "Error: Se requieren al menos 4 puntos para dibujar un contorno." << std::endl;
        return;
    }

    // Dibujar líneas entre los puntos en orden
    for (size_t i = 0; i < points.size(); i++)
    {
        cv::line(image, points[i], points[(i + 1) % points.size()], color, thickness);
    }
}

void rva_deform_image(const cv::Mat &im_input, cv::Mat &im_output, cv::Mat homography)
{
    // Verificar que la homografía no esté vacía
    if (homography.empty())
    {
        std::cerr << "Error: La matriz de homografía está vacía. No se puede deformar la imagen." << std::endl;
        return;
    }

    // Aplicar la transformación de perspectiva
    cv::warpPerspective(im_input, im_output, homography, im_output.size());
}

// TASK 2

void rva_calculaKPsDesc(const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    cv::Ptr<cv::Feature2D> detector;

    detector = cv::SIFT::create();
    // if (type == "ORB") {
    //     detector = cv::ORB::create();
    // } else if (type == "SIFT") {
    //     detector = cv::SIFT::create();
    // } else if (type == "AKAZE") {
    //     detector = cv::AKAZE::create();
    // } else {
    //     std::cerr << "Tipo de detector no reconocido: " << type << std::endl;
    //     return;
    // }

    detector->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
}

void rva_matchDesc(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches)
{
    cv::Ptr<cv::DescriptorMatcher> matcher;

    // if (type == "SIFT") {
    //     matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    // } else {
    //     matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    // }

    //matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
    matcher->match(descriptors1, descriptors2, matches);
}

void rva_dibujaMatches(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches, cv::Mat &img_matches)
{
    // Dibujar los matches
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
}

// TASK 3 - los matches ya estan filtrados en el main por la funcion rva_filterMatches
void rva_localizaObj(const cv::Mat &img1, const cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches, cv::Mat &homography, std::vector<cv::Point2f> &pts_im2)
{
    // Usamos los matches ya filtrados, no es necesario recalcular descriptores aquí
    std::vector<cv::Point2f> obj_points;
    std::vector<cv::Point2f> scene_points;

    // Obtener los puntos correspondientes de los matches
    for (size_t i = 0; i < matches.size(); i++)
    {
        obj_points.push_back(keypoints1[matches[i].queryIdx].pt);
        scene_points.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Calcular la homografía entre las imágenes utilizando los matches (ya filtrados previamente)
    homography = cv::findHomography(obj_points, scene_points, cv::RANSAC);

    // Definir las esquinas del objeto (modelo) en img1
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point2f(0, 0);
    obj_corners[1] = cv::Point2f((float)img1.cols, 0);
    obj_corners[2] = cv::Point2f((float)img1.cols, (float)img1.rows);
    obj_corners[3] = cv::Point2f(0, (float)img1.rows);

    // Transformar las esquinas usando la homografía calculada
    std::vector<cv::Point2f> scene_corners(4);
    cv::perspectiveTransform(obj_corners, scene_corners, homography);

    // Almacenar las coordenadas de las esquinas transformadas
    pts_im2 = scene_corners;
}

// TASK 4

void rva_filterMatches(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints_model, const std::vector<cv::KeyPoint> &keypoints_scene, std::vector<cv::DMatch> &good_matches)
{
    /*

    Obtienes los puntos correspondientes a partir de los matches iniciales.
    Calculas la homografía usando findHomography() con RANSAC.
    OpenCV devuelve una máscara (inliers_mask) que indica qué matches son inliers (coherentes con la homografía).
    Los matches que no cumplan esa coherencia geométrica se descartan.

    */
    if (matches.empty())
    {
        std::cerr << "Error: No hay matches para filtrar." << std::endl;
        return;
    }

    // Extraer los puntos de cada match
    std::vector<cv::Point2f> points_model, points_scene;
    for (const auto &match : matches)
    {
        points_model.push_back(keypoints_model[match.queryIdx].pt);
        points_scene.push_back(keypoints_scene[match.trainIdx].pt);
    }

    // Calcular homografía con RANSAC para obtener inliers
    cv::Mat inlier_mask; // nos quedamos con los matches buenos
    double default_threshold = 3.0;
    cv::findHomography(points_model, points_scene, cv::RANSAC, default_threshold, inlier_mask);

    // Filtrar los matches usando la máscara
    for (size_t i = 0; i < matches.size(); ++i)
        if (inlier_mask.at<uchar>(i))
            good_matches.push_back(matches[i]);
}

void rva_dibujaPatch(const cv::Mat &scene, const cv::Mat &patch, const cv::Mat &homography, cv::Mat &output)
{
    // Verificar que la homografía no esté vacía
    if (homography.empty())
    {
        std::cerr << "Error: La matriz de homografía está vacía. No se puede dibujar el patch." << std::endl;
        return;
    }

    // Obtener las dimensiones del patch
    int patch_width = patch.cols;
    int patch_height = patch.rows;

    // Calcular las esquinas del patch (objeto) en la imagen original (patch)
    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point2f(0, 0);
    obj_corners[1] = cv::Point2f((float)patch_width, 0);
    obj_corners[2] = cv::Point2f((float)patch_width, (float)patch_height);
    obj_corners[3] = cv::Point2f(0, (float)patch_height);

    // Transformar las esquinas del patch a la imagen de la escena usando la homografía
    std::vector<cv::Point2f> scene_corners(4);
    cv::perspectiveTransform(obj_corners, scene_corners, homography);

    // Definir la región de la imagen de la escena donde el patch será colocado
    cv::Mat scene_roi = scene.clone();
    
    // Aquí aplicamos la transformación inversa a las coordenadas del patch
    cv::Mat warped_patch;
    cv::warpPerspective(patch, warped_patch, homography, scene.size());

    // Crear la imagen de salida (scene con el patch dibujado)
    output = scene.clone();

    // Para evitar que el patch sobrescriba la escena de manera incorrecta, usamos la máscara
    cv::Mat mask = cv::Mat::zeros(scene.size(), CV_8UC1);
    std::vector<std::vector<cv::Point>> contours(1);
    for (size_t i = 0; i < scene_corners.size(); i++)
    {
        contours[0].push_back(cv::Point(scene_corners[i].x, scene_corners[i].y));
    }
    cv::fillPoly(mask, contours, cv::Scalar(255));

    // Usamos la máscara para colocar el patch en la región correspondiente
    warped_patch.copyTo(output, mask); 
}

void rva_mostrarInfo(cv::Mat &image, const std::vector<cv::String> &info, const std::vector<cv::Point2f> &vertices)
{
    if (vertices.size() < 4 || info.empty())
        return;

    // Calcular el bounding box de los vértices
    float min_x = vertices[0].x, max_x = vertices[0].x;
    float min_y = vertices[0].y, max_y = vertices[0].y;
    for (const auto &v : vertices)
    {
        min_x = std::min(min_x, v.x);
        max_x = std::max(max_x, v.x);
        min_y = std::min(min_y, v.y);
        max_y = std::max(max_y, v.y);
    }

    float box_width = max_x - min_x;
    float box_height = max_y - min_y;

    // Parámetros
    float line_spacing_factor = 1.3f; // un poco más de espacio entre líneas
    float margin_x = box_width * 0.05f;
    float margin_y = box_height * 0.05f;

    // Calcular altura máxima disponible por línea
    float available_height = box_height - 2 * margin_y;
    float max_line_height = available_height / (info.size() * line_spacing_factor);

    // Calcular font_scale basado en altura
    // La altura media de un carácter para FONT_HERSHEY_SIMPLEX ~ 20 px a escala 1.0, mejor tomar 20.0 como referencia más realista
    float font_scale_height = max_line_height / 10.0f;

    // Además, comprobar que no se salga en ancho: limitar el tamaño en función del texto más largo
    float font_scale_width = 10.0f; // algo grande para empezar
    for (const auto &line : info)
    {
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseline);
        float candidate_scale = (box_width - 2 * margin_x) / (float)(text_size.width + 1e-6);
        if (candidate_scale < font_scale_width)
            font_scale_width = candidate_scale;
    }

    // Usar el mínimo de ambos
    float font_scale = std::min(font_scale_height, font_scale_width);

    // Empezar a escribir desde la parte superior con margen
    float start_x = min_x + margin_x;
    float start_y = min_y + margin_y;

    for (size_t i = 0; i < info.size(); i++)
    {
        cv::Point2f text_pos(start_x, start_y + i * (max_line_height * line_spacing_factor));

        // Obtener tamaño del texto para el fondo
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(info[i], cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);

        // Definir rectángulo de fondo (con margen extra)
        cv::Rect background_rect(
            text_pos.x - 2, 
            text_pos.y - text_size.height - 2, 
            text_size.width + 4, 
            text_size.height + baseline + 4
        );

        // Dibujar el fondo (puedes elegir color y opacidad si usas cv::addWeighted)
        cv::rectangle(image, background_rect, cv::Scalar(255, 255, 255), cv::FILLED);

        // Dibujar el texto encima
        cv::putText(image, info[i], text_pos, cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}
