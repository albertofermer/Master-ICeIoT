#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "common_code.hpp"

void fsiv_compute_derivate(cv::Mat const &img, cv::Mat &dx, cv::Mat &dy, int g_r,
                           int s_ap)
{
    CV_Assert(img.type() == CV_8UC1);
    // TODO
    // Remember: if g_r > 0 apply a previous Gaussian Blur operation with kernel size 2*g_r+1.
    // Hint: use Sobel operator to compute derivate.
    if (g_r > 0) {
        cv::GaussianBlur(img, img, cv::Size(2*g_r+1,2*g_r+1), 0);
    }

    // use Sobel operator to compute derivate.
    cv::Sobel(img, dx, CV_32FC1, 1, 0, s_ap);
    cv::Sobel(img, dy, CV_32FC1, 0, 1, s_ap);
    //
    CV_Assert(dx.size() == img.size());
    CV_Assert(dy.size() == dx.size());
    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(dy.type() == CV_32FC1);
}

void fsiv_compute_gradient_magnitude(cv::Mat const &dx, cv::Mat const &dy,
                                     cv::Mat &gradient)
{
    CV_Assert(dx.size() == dy.size());
    CV_Assert(dx.type() == CV_32FC1);
    CV_Assert(dy.type() == CV_32FC1);

    // TODO
    // Hint: use cv::magnitude.
    cv::magnitude(dx,dy,gradient);
    //

    CV_Assert(gradient.size() == dx.size());
    CV_Assert(gradient.type() == CV_32FC1);
}

void fsiv_compute_gradient_histogram(cv::Mat const &gradient, int n_bins, cv::Mat &hist, float &max_gradient)
{
    // TODO
    // Hint: use cv::minMaxLoc to get the gradient range {0, max_gradient}
    double max_value;
    cv::minMaxLoc(gradient, nullptr, &max_value);
    max_gradient = (float)(max_value);
    
    // Calcular histograma:
    int histSize[] = {n_bins};
    float range[] = {0,float(max_gradient)};
    const float* histRange[] = {range};
    cv::calcHist(&gradient, 1, 0, cv::Mat(), hist, 1, histSize, histRange, true, false);
    //cv::normalize(hist, hist, 1, 0, cv::NORM_L1);
    //
    CV_Assert(max_gradient > 0.0);
    CV_Assert(hist.rows == n_bins);
}

int fsiv_compute_histogram_percentile(cv::Mat const &hist, float percentile)
{
    CV_Assert(percentile >= 0.0 && percentile <= 1.0);
    CV_Assert(hist.type() == CV_32FC1);
    CV_Assert(hist.cols == 1);
    int idx = 0;
    // TODO
    // Hint: use cv::sum to compute the histogram area.
    // Remember: The percentile p is the first i that sum{h[0], h[1], ..., h[i]} >= p

    float suma_acumulada = 0.0;

    while(suma_acumulada < percentile && idx < hist.rows){
        suma_acumulada += hist.at<float>(idx);
        idx++;
    }
    //Corregimos desfase si ha entrado en el bucle
    if (idx > 0 && idx != hist.rows - 1){
        idx--;
    }

    //
    CV_Assert(idx >= 0 && idx < hist.rows);
    CV_Assert(idx == 0 || cv::sum(hist(cv::Range(0, idx), cv::Range::all()))[0] / cv::sum(hist)[0] < percentile);
    CV_Assert(cv::sum(hist(cv::Range(0, idx + 1), cv::Range::all()))[0] / cv::sum(hist)[0] >= percentile);
    return idx;
}

float fsiv_histogram_idx_to_value(int idx, int n_bins, float max_value,
                                  float min_value)
{
    CV_Assert(idx >= 0);
    CV_Assert(idx < n_bins);
    float value = 0.0;
    // TODO
    // Remember: Map integer range [0, n_bins) into float
    // range [min_value, max_value)
    value = float(idx) / float(n_bins) * (max_value - min_value) + min_value;
    //
    CV_Assert(value >= min_value);
    CV_Assert(value < max_value);
    return value;
}

void fsiv_percentile_edge_detector(cv::Mat const &gradient, cv::Mat &edges,
                                   float th, int n_bins)
{
    CV_Assert(gradient.type() == CV_32FC1);

    // TODO
    // Remember: user other fsiv_xxx to compute histogram and percentiles.
    // Remember: map histogram range {0, ..., n_bins} to the gradient range
    // {0.0, ..., max_grad}
    // Hint: use "operator >=" to threshold the gradient magnitude image.
    cv::Mat hist;
    float max_gradient;
    // Calcula el histograma de la imagen
    fsiv_compute_gradient_histogram(gradient, n_bins, hist, max_gradient);
    //normalizamos el histograma
    cv::normalize(hist, hist, 1, 0, cv::NORM_L1);
    // Obtiene el id del histograma con el threshold 
    int th_idx = fsiv_compute_histogram_percentile(hist, th);
    // Usamos la funcion de mapeo para calcular el nivel de gris:
    float threshold_value = fsiv_histogram_idx_to_value(th_idx, n_bins, max_gradient);
    // Obtenemos la imagen binaria de los bordes
    edges = (gradient >= threshold_value) * 255;
    //
    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == gradient.size());
}

void fsiv_otsu_edge_detector(cv::Mat const &gradient, cv::Mat &edges)
{
    CV_Assert(gradient.type() == CV_32FC1);

    // TODO
    // Hint: normalize input gradient into range [0, 255] to use
    // cv::threshold properly.
    //
    //Normalizamos el input en el rango [0,255]
    cv::Mat normalized_gradient;
    cv::normalize(gradient, normalized_gradient, 0, 255, cv::NORM_MINMAX);
    // Convertimos los datos a CV_8UC1
    cv::Mat gradient_8u;
    normalized_gradient.convertTo(gradient_8u, CV_8UC1);

    cv::threshold(gradient_8u, edges, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    //
    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == gradient.size());
}

void fsiv_canny_edge_detector(cv::Mat const &dx, cv::Mat const &dy, cv::Mat &edges,
                              float th1, float th2, int n_bins)
{
    CV_Assert(dx.size() == dy.size());
    CV_Assert(th1 < th2);

    // TODO
    // Hint: convert the intput derivatives to CV_16C1 to be used with canny.
    cv::Mat dx_16, dy_16;
    dx.convertTo(dx_16, CV_16SC1);
    dy.convertTo(dy_16, CV_16SC1);

    // Remember: th1 and th2 are given as percentiles so you must transform to
    //           gradient range to be used in canny method.

    // Remember: we compute gradients with L2_NORM so we must indicate this in
    //           the canny method too.
    cv::Mat gradient;
    fsiv_compute_gradient_magnitude(dx, dy, gradient);
    gradient.convertTo(gradient, CV_8U);

    cv::Mat hist;
    float max_gradient;
    fsiv_compute_gradient_histogram(gradient, n_bins, hist, max_gradient);
    cv::normalize(hist, hist,1,0,cv::NORM_L1);
    // Calculamos los indices que cumplen el percentil.
    int th1_idx = fsiv_compute_histogram_percentile(hist, th1);
    int th2_idx = fsiv_compute_histogram_percentile(hist, th2);
    //Convertimos los indices a valores de gris
    float threshold1 = fsiv_histogram_idx_to_value(th1_idx, n_bins, max_gradient);
    float threshold2 = fsiv_histogram_idx_to_value(th2_idx, n_bins, max_gradient);
    // Aplicamos el detector de bordes de Canny
    cv::Canny(dx_16, dy_16, edges, threshold1, threshold2, true);
    edges.convertTo(edges, CV_8UC1);
    //
    CV_Assert(edges.type() == CV_8UC1);
    CV_Assert(edges.size() == dx.size());
}

void fsiv_compute_ground_truth_image(cv::Mat const &consensus_img,
                                     float min_consensus, cv::Mat &gt)
{
    //! TODO
    // Hint: use cv::normalize to normalize consensus_img into range (0, 100)
    cv::Mat normalized_consensus_img;
    cv::normalize(consensus_img,normalized_consensus_img, 0,100, cv::NORM_MINMAX);
    // Hint: use "operator >=" to threshold the consensus image.
    gt = (normalized_consensus_img >= min_consensus) * 255;
    //
    CV_Assert(consensus_img.size() == gt.size());
    CV_Assert(gt.type() == CV_8UC1);
}

void fsiv_compute_confusion_matrix(cv::Mat const &gt, cv::Mat const &pred, cv::Mat &cm)
{
    CV_Assert(gt.type() == CV_8UC1);
    CV_Assert(pred.type() == CV_8UC1);
    CV_Assert(gt.size() == pred.size());

    // TODO
    // Remember: a edge detector confusion matrix is a 2x2 matrix where the
    // rows are ground truth {Positive: "is edge", Negative: "is not edge"} and
    // the columns are the predictions labels {"is edge", "is not edge"}
    // A pixel value means edge if it is <> 0, else is a "not edge" pixel.
    
    // Inicializamos la matriz de confusión: 
    // [True Positive, False Negative]
    // [False Positive, True Negative]
    cm = cv::Mat::zeros(2, 2, CV_32FC1);

    // Recorremos los píxeles de las imágenes
    for (int i = 0; i < gt.rows; i++)
    {
        for (int j = 0; j < gt.cols; j++)
        {
            // Normalizamos los pixeles para trabajar con valores 0 o 1
            bool gt_is_edge = (gt.at<uchar>(i, j) > 0);
            bool pred_is_edge = (pred.at<uchar>(i, j) > 0);

            // Incrementamos la matriz de confusión según los valores
            cm.at<float>(gt_is_edge ? 0 : 1, pred_is_edge ? 0 : 1) += 1;

            //  [0,0] - TP
            //  [0,1] - FN
            //  [1,0] - FP
            //  [1,1] - TN
        }
    }
    //
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cv::abs(cv::sum(cm)[0] - (gt.rows * gt.cols)) < 1.0e-6);
}

float fsiv_compute_sensitivity(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float sensitivity = 0.0;
    // TODO
    // Hint: see https://en.wikipedia.org/wiki/Confusion_matrix
    // Sensitivity (True Positive Rate) = TP / (TP + FN)
    sensitivity = cm.at<float>(0, 0) / (cm.at<float>(0, 0) + cm.at<float>(0, 1));

    //
    return sensitivity;
}

float fsiv_compute_precision(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float precision = 0.0;
    // TODO
    // Hint: see https://en.wikipedia.org/wiki/Confusion_matrix
    // Precision = TP / (TP + FP)
    precision = cm.at<float>(0, 0) / (cm.at<float>(0, 0) + cm.at<float>(1, 0));


    //
    return precision;
}

float fsiv_compute_F1_score(cv::Mat const &cm)
{
    CV_Assert(cm.type() == CV_32FC1);
    CV_Assert(cm.size() == cv::Size(2, 2));
    float F1 = 0.0;
    // TODO
    // Hint: see https://en.wikipedia.org/wiki/Confusion_matrix
    //F1 Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)

    float sensitivity = fsiv_compute_sensitivity(cm);
    float precision = fsiv_compute_precision(cm);
    F1 = 2 * (precision * sensitivity) / (precision + sensitivity);


    //
    return F1;
}
