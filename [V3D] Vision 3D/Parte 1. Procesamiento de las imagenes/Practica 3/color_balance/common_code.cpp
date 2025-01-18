#include "common_code.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

const int histSize = 256;
cv::Mat fsiv_color_rescaling(const cv::Mat &in, const cv::Scalar &from, const cv::Scalar &to)
{
    CV_Assert(in.type() == CV_8UC3);
    cv::Mat out;
    // TODO
    // HINT: use cv:divide to compute the scaling factor.
    // HINT: use method cv::Mat::mul() to scale the input matrix.

    // Convert 'from' and 'to' scalars to 3x1 matrices
    cv::Mat fromMat = cv::Mat(from, false);
    cv::Mat toMat = cv::Mat(to, false);
    cv::Mat scaling_factor;

    cv::divide(toMat, fromMat, scaling_factor);
    out = in.mul(scaling_factor);

    // O = a * I
    // a = 128 / R ; 128 / G ; 128 / B;

    //
    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}

cv::Mat fsiv_gray_world_color_balance(cv::Mat const &in)
{
    CV_Assert(in.type() == CV_8UC3);
    cv::Mat out;
    // TODO
    //  HINT: use cv::mean to compute the mean pixel value.
    cv::Scalar mean = cv::mean(in);
    out = fsiv_color_rescaling(in, mean, cv::Scalar(128, 128, 128)); // 128 gris promedio

    //
    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}

cv::Mat fsiv_convert_bgr_to_gray(const cv::Mat &img, cv::Mat &out)
{
    CV_Assert(img.channels() == 3);
    // TODO
    // HINT: use cv::cvtColor()
    cv::cvtColor(img, out,cv::COLOR_BGR2GRAY);
    //
    CV_Assert(out.channels() == 1);
    return out;
}

cv::Mat fsiv_compute_image_histogram(cv::Mat const &img)
{
    CV_Assert(img.type() == CV_8UC1);
    cv::Mat hist;
    // TODO
    // Hint: use cv::calcHist().

    float range[] = {0, 256};
    const float *histRange = {range};
    // hue varies from 0 to 179, see cvtColor
    const float rgb_ranges[] = {0, 256.0f};
    std::vector<cv::Mat> bgr(3);
    cv::split(img,bgr);
    calcHist(   &img, 
                1,
                0,
                cv::Mat(), // do not use mask
                hist,
                1,
                &histSize,
                &histRange,
                true,
                false);    //
    CV_Assert(!hist.empty());
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.rows == 256 && hist.cols == 1);
    return hist;
}

float fsiv_compute_histogram_percentile(cv::Mat const &hist, float p_value)
{
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.cols == 1);
    CV_Assert(0.0 <= p_value && p_value <= 1.0);

    int p = 0;
    cv::Scalar area = cv::sum(hist);
    float threshold = p_value*area[0]; // umbral
    float suma_acumulada = 0.0;

    while(suma_acumulada < threshold && p < histSize){
        suma_acumulada += hist.at<float>(p);
        p++;
    }

    // Corregimos el desfase de la variable en el bucle
    p = p-1;

    // TODO
    // Remember: find the smaller index 'p' such that
    //           sum(h[0], h[1], ... , h[p]) >= p_value*area(hist)
    // Hint: use cv::sum() to compute the histogram area.

    //

    CV_Assert(0 <= p && p < hist.rows);
    return p;
}

cv::Mat fsiv_white_patch_color_balance(cv::Mat const &in, float p)
{
    CV_Assert(in.type() == CV_8UC3);
    CV_Assert(0.0f <= p && p <= 100.0f);
    cv::Mat out;
    // HINT: convert to GRAY color space to get the illuminance.
    cv::Mat gray_img;
    fsiv_convert_bgr_to_gray(in, gray_img);
    cv::Point brightest_pixel = cv::Point();
    if (p == 0.0)
    {
        // TODO
        
        // HINT: use cv::minMaxLoc to locate the brightest pixel.
        cv::minMaxLoc(gray_img, nullptr, nullptr, nullptr, &brightest_pixel);
        // Obtenemos el color del píxel mas brillante 
        cv::Scalar from = in.at<cv::Vec3b>(brightest_pixel);
        // HINT: use fsiv_color_rescaling when the "from" scalar was computed.
        out = fsiv_color_rescaling(in,from,cv::Scalar(255,255,255));

        //
    }
    else
    {
        // HINT: Compute a gray level histogram to find the 100-p percentile.
        cv::Mat hist = fsiv_compute_image_histogram(gray_img);
        float max_value = fsiv_compute_histogram_percentile(hist, float(100-p)/100);
        
        // HINT: use operator >= to get the mask with p% brighter pixels and use it
        //        to compute the mean value.
        cv::Mat mask = gray_img >= max_value;
        cv::Scalar from = cv::mean(in,mask);
        // HINT: use fsiv_color_rescaling when the "from" scalar was computed.
        out = fsiv_color_rescaling(in, from, cv::Scalar(255,255,255));

        //
    }

    CV_Assert(out.type() == in.type());
    CV_Assert(out.rows == in.rows && out.cols == in.cols);
    return out;
}
