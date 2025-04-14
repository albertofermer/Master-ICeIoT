// (c) Realidad Virtual y Aumentada - Universidad de Cordoba - Manuel J. Marin-Jimenez

#include <iostream>
#include <opencv2/opencv.hpp>

#include "rva.h"

void rva_calculaKPsDesc(cv::Mat &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors, const std::string &type)
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

void rva_dibujaMatches(cv::Mat &img1, cv::Mat &img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches, cv::Mat &img_matches)
{
    // Dibujar los matches
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
}
