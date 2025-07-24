#include <iostream>
#include <filesystem> // C++17

#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

#include <chrono> // Para medir el tiempo

#include "rva.h"

using namespace std;

// Función para mostrar la barra de progreso
void showProgressBar(int progress, int total)
{
    int barWidth = 50; // Ancho de la barra de progreso
    float ratio = (float)progress / total;
    int pos = barWidth * ratio;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "#";
        else
            std::cout << "-";
    }
    std::cout << "] " << int(ratio * 100.0) << "%\r";
    std::cout.flush();
}

cv::Mat reescalar_imagen(int width, int height, cv::Mat img_scene)
{

    // Calcular el factor de escala para que la imagen encaje en la pantalla
    float scale_width = static_cast<float>(width) / img_scene.cols;
    float scale_height = static_cast<float>(height) / img_scene.rows;
    float scale_factor = std::min(scale_width, scale_height); // Mantener la proporción

    // Reescalar la imagen a la resolución de la pantalla
    cv::resize(img_scene, img_scene, cv::Size(), scale_factor, scale_factor);

    return img_scene;
}

void screenshot(string scene_path, string descriptor_type, cv::Mat img_scene, string parameter)
{
    std::filesystem::path path(scene_path);
    std::string nuevoNombre = path.stem().string() + "_" + parameter + "_screenshot_" + descriptor_type + path.extension().string();
    cv::imwrite("../../data/results/" + nuevoNombre, img_scene);
    std::cout << "Captura guardada como " << nuevoNombre << std::endl;
}

const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@model         |<none>| Path to image model.}"
    "{@scene         |<none>| Path to image scene.}"
    "{descriptor   |AKAZE| Name of the descriptor (AKAZE, ORB, SIFT).}"
    "{patch         |<none>| Path to image patch.}"
    "{video         |<none>| Path to playable video.}"
    "{save-video     |<none>| Path to save video }" // Parámetro opcional para guardar el video
    "{experimental  |1| number of times you want to execute the code for measure the mean value of time (ms) (>1).}";

int main(int argc, char **argv)
{

    // Get the arguments: model, video and patch using OpenCv parser
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    string model_path = parser.get<string>(0);
    string scene_path = parser.get<string>(1);

    string descriptor_type = parser.get<string>("descriptor");
    std::transform(descriptor_type.begin(), descriptor_type.end(), descriptor_type.begin(), ::toupper); // Lo convertimos a mayusculas para que no haya fallo

    string patch_path = parser.get<string>("patch");
    string video_path = parser.get<string>("video");

    string save_video_path = parser.get<std::string>("save-video");

    int experimental = parser.get<int>("experimental");
    if (experimental < 1)
        experimental = 1;

    bool use_video = !video_path.empty();
    bool use_patch = !patch_path.empty() && !use_video;
    bool save_video = !save_video_path.empty();
    bool use_info_text = !use_video && !use_patch;

    std::vector<cv::String> info_text = {
        "Titulo: El caminante sobre el mar de nubes",
        "Fecha: 1818",
        "Autor: Caspar David Friedrich"};

    // Mostramos información sobre la ejecución
    cout << "\
             ================\n \
             Cuadro: El caminante sobre el mar de nubes\n \
             Descriptor: "
         << descriptor_type << "\n \
             Patch: "
         << use_patch << "\n \
             Video: "
         << use_video << "\n \
             Ruta video: "
         << save_video << "\n \
             Numero de ejecuciones: "
         << experimental << "\n \
             ================"
         << endl;

    // Cargamos las imagenes
    cv::Mat img_model = cv::imread(model_path, cv::IMREAD_COLOR);
    cv::Mat img_patch;
    if (use_patch)
        img_patch = cv::imread(patch_path, cv::IMREAD_COLOR);

    if (img_model.empty() || (img_patch.empty() && use_patch))
    {
        cout << "Error: images not loaded" << endl;
        return -1;
    }

    // Change resolution of the image model to half
    cv::resize(img_model, img_model, cv::Size(), 0.5, 0.5);

    if (use_patch)
        cv::resize(img_patch, img_patch, img_model.size());

    // Load the scene image
    cv::Mat img_scene = cv::imread(scene_path, cv::IMREAD_COLOR);
    if (img_scene.empty())
    {
        cout << "Error: scene image not loaded" << endl;
        return -1;
    }

    // Pre-compute keypoints and descriptors for the model image
    std::vector<cv::KeyPoint> keypoints_model;
    cv::Mat descriptors_model;
    rva_calculaKPsDesc(img_model, keypoints_model, descriptors_model, descriptor_type);

    img_scene = reescalar_imagen(1920, 1080, img_scene);

    cv::Mat H;
    std::vector<cv::Point2f> pts_obj_in_scene_filtered, pts_obj_in_scene_no_filter;

    // ** Empezamos a medir el tiempo ** //
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < experimental; i++)
    {
        // Compute keypoints and descriptors for the scene image
        std::vector<cv::KeyPoint> keypoints_scene;
        cv::Mat descriptors_scene;
        rva_calculaKPsDesc(img_scene, keypoints_scene, descriptors_scene, descriptor_type);

        // Match the descriptors
        std::vector<cv::DMatch> matches;
        rva_matchDesc(descriptors_model, descriptors_scene, matches, descriptor_type);

        // Filter out outliers
        std::vector<cv::DMatch> good_matches;
        rva_filterMatches(matches, keypoints_model, keypoints_scene, good_matches);

        // calcula la bounding-box
        rva_localizaObj(img_model, img_scene, keypoints_model, keypoints_scene, good_matches, H, pts_obj_in_scene_filtered);

        cout << "Numero de matches: " << matches.size() << endl;
        cout << "Numero de matches filtrados: " << good_matches.size() << endl;

        // calcula la bounding-box sin filtrar
        rva_localizaObj(img_model, img_scene, keypoints_model, keypoints_scene, matches, H, pts_obj_in_scene_no_filter);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    std::cout << "Tiempo medio de ejecución: " << duration.count() / experimental << " ms" << std::endl;

    // Display image based on whether using video or patch
    if (use_info_text)
    {
        string par = "info";
        rva_draw_contour(img_scene, pts_obj_in_scene_filtered, cv::Scalar(0, 255, 0), 2);  // deteccion con los puntos filtrados
        rva_draw_contour(img_scene, pts_obj_in_scene_no_filter, cv::Scalar(0, 0, 255), 2); // detección con los puntos sin filtrar
        rva_mostrarInfo(img_scene, info_text, pts_obj_in_scene_filtered);

        while (true)
        {
            cv::imshow("AugmentedReality-Info", img_scene);
            int key = cv::waitKey(0);
            if (key == 's' || key == 'S')
                screenshot(scene_path, descriptor_type, img_scene, par); // guarda captura de pantalla
            else
                break; // si pulsa cualquier otra tecla se cierra
        }
    }
    else if (use_patch)
    {
        rva_dibujaPatch(img_scene, img_patch, H, img_scene);
        string par = "patch";

        while (true)
        {
            cv::imshow("AugmentedReality-Patch", img_scene);
            int key = cv::waitKey(0);
            if (key == 's' || key == 'S')
                screenshot(scene_path, descriptor_type, img_scene, par); // guarda captura de pantalla
            else
                break; // si pulsa cualquier otra tecla se cierra
        }
    }
    else
    {
        // Si se está usando video, procesar los fotogramas
        cv::VideoCapture cap2;
        string par = "video";
        if (use_video)
        {
            if (isdigit(video_path[0]))
                cap2.open(std::stoi(video_path));
            else
                cap2.open(video_path);

            if (!cap2.isOpened())
            {
                cout << "Error: video not loaded" << endl;
                return -1;
            }

            // Obtener información del video
            int totalFrames = static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_COUNT));
            int currentFrame = 0;
            cv::Mat frame, output;
            cv::VideoWriter video_writer;
            bool video_initialized = false;
            while (cap2.read(frame))
            {
                if (frame.empty())
                {
                    std::cerr << "Frame vacío. Finalizando..." << std::endl;
                    break;
                }

                img_scene.copyTo(output);
                cv::Mat resized_frame;
                cv::resize(frame, resized_frame, img_model.size());
                rva_dibujaPatch(output, resized_frame, H, output);

                // Mostrar el resultado en la ventana
                cv::imshow("AugmentedReality-Video", output);

                // Inicializar el VideoWriter solo la primera vez
                if (save_video && !video_initialized)
                {
                    video_writer.open(save_video_path, cv::VideoWriter::fourcc('H', '2', '6', '4'), 30, output.size(), true); // codec para guardar en .mp4 --> h264
                    if (!video_writer.isOpened())
                    {
                        std::cerr << "Error al abrir el archivo de video para guardar." << std::endl;
                        break;
                    }
                    video_initialized = true;
                }

                // Escribir el frame en el archivo de video si la opción de guardar está activada
                if (save_video)
                    video_writer.write(output);

                // Mostrar barra de progreso
                currentFrame++;
                showProgressBar(currentFrame, totalFrames);

                // Esperar 30ms para la tecla (y actualizar la barra de progreso sin bloquear el video)
                int key = cv::waitKey(30);
                if (key == 's' || key == 'S')
                    screenshot(scene_path, descriptor_type, output, par); // guarda captura de pantalla
                else if (key >= 0)
                    break; // Salir si se presiona alguna tecla
            }
        }
    }

    cv::destroyAllWindows();
    return 0;
}
