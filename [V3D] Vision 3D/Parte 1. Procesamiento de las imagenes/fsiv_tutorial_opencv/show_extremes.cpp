#include <iostream>
#include <exception>
using namespace std;
//OpenCV includes
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/calib3d.hpp> //Uncomment when it was appropiated.
//#include <opencv2/ml.hpp> //Uncomment when it was appropiated.

#include "common_code.hpp"

const char * keys =
    "{help h usage ? |      | print this message}"
    "{w              |20    | Wait time (miliseconds) between frames.}"
    "{v              |      | the input is a video file.}"
    "{c              |      | the input is a camera index.}"    
    "{@input         |<none>| input <fname|int>}"
    ;




int
main (int argc, char* const* argv)
{
  int retCode=EXIT_SUCCESS;
  
  try {    

      cv::CommandLineParser parser(argc, argv, keys);
      parser.about("Show the extremes values and their locations.");
      if (parser.has("help"))
      {
          parser.printMessage();
          return 0;
      }
      bool is_video = parser.has("v");
      bool is_camera = parser.has("c");
      int wait = parser.get<int>("w");
      cv::String input = parser.get<cv::String>("@input");
      if (!parser.check())
      {
          parser.printErrors();
          return 0;
      }

    // TODO
    /*
      Crea tu propio programa “mostrar_extremos” modificando el fichero
      “mostrar_extremos.cpp” e intenta hacer un programa que cargue una imagen, 
      la visualice e imprima los valores máximo/mínimo por canal usando 
      las funciones que has codificado en el módulo “common_code.cpp”. 
      
      Si se indica que la entrada es una fuente de vídeo (con -v) o una cámara (con -c), 
      los valores se mostrarán para cada imagen de dicha fuente de vídeo.
     */
    // Variables de posicion y valor
      std::vector<double> min_v;
      std::vector<double> max_v;
      std::vector<cv::Point> min_loc; 
      std::vector<cv::Point> max_loc;
      
    if (!is_video && !is_camera){
      cout << "LEYENDO IMAGEN" << endl;
      //Carga la imagen desde archivo.
      cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);

      //cv::Mat img = cv::imread(img_name, cv::IMREAD_GRAYSCALE);
      //cv::Mat img = cv::imread(img_name, cv::IMREAD_COLOR);

      if (img.empty())
      {
         std::cerr << "Error: no he podido abrir el fichero '" << input << "'." << std::endl;
         return EXIT_FAILURE;
      }

      //Creo la ventana grafica para visualizar la imagen.
      //El nombre de la ventana sirve como 'handle' para gestionarla despues.
      //Lee la documentacon de namedWindow para mas detalles.
      cv::namedWindow("SHOW_EXTREMES - IMG", cv::WINDOW_GUI_EXPANDED);

      //Visualizo la imagen cargada en la ventana.
      cv::imshow("SHOW_EXTREMES - IMG", img);

      //Para que se actualice la interfaz gráfica es necesario llamar a
      //waitKey. Además esta función nos permite interactuar con el teclado.
      //Lee la documentación de waitKey para mas detalles.
      do {
          fsiv_find_min_max_loc_2(img, min_v, max_v, min_loc, max_loc);
          cout << "Valor mínimo: " << "B:" << min_v.at(0) << ", G:" << min_v.at(1) <<", R:"<< min_v.at(2) << " (" << min_loc.at(0) << ")" << endl;
          cout << "Valor máximo: " << "B:" << max_v.at(0) << ", G:" << max_v.at(1) <<", R:"<< max_v.at(2) << " (" << max_loc.at(0) << ")" << endl;
          std::cout << "Pulsa ESC para salir." << std::endl;
      } while ((cv::waitKey(1) & 0xff) != 27); 

      //Debemos cerrar las ventanas abiertas.
      cv::destroyWindow("SHOW_EXTREMES - IMG");
      
    }
    else{
      // abrir video
      cv::VideoCapture vid;
      if (is_video)
        vid.open(input); // tanto el numero de camara, como el nombre del video se guardarian en esta variable.
      else{
        vid.open(parser.get<int>("c"));
      }
      
      if (!vid.isOpened())
      {
         std::cerr << "Error: no he podido abrir el la fuente de vídeo." << std::endl;
         return EXIT_FAILURE;
      }
        
      //Creo la ventana grafica para visualizar la imagen.
      //El nombre de la ventana sirve como 'handle' para gestionarla despues.
      //Lee la documentacon de namedWindow para mas detalles.
      cv::namedWindow("SHOWEXTREMES - VIDEO");
      
      cv::Mat frame;

      //Captura el primer frame.
      //Si el frame esta vacio, puede ser un error hardware o fin del video.
      vid >> frame;

      if (frame.empty())
      {
          std::cerr << "Error: could not capture any frame from source." << std::endl;
          return EXIT_FAILURE;
      }

      std::cout << "Input size (WxH): " << frame.cols << 'x' << frame.rows << std::endl;
      std::cout << "Frame rate (fps): " << vid.get(cv::CAP_PROP_FPS) << std::endl;
      std::cout << "Num of frames   : " << vid.get(cv::CAP_PROP_FRAME_COUNT) << std::endl;


      //Creamos la ventana para mostrar el video y
      //le conectamos una función "callback" para gestionar el raton.
      cv::namedWindow("SHOWEXTREMES - VIDEO");
      std::cerr << "Pulsa una tecla para continuar (ESC para salir)." << std::endl;
      int key = cv::waitKey(0) & 0xff;
      
      //Muestro frames hasta fin del video (frame vacio),
      //o que el usario pulse la tecla ESCAPE (codigo ascci 27)
      while (!frame.empty() && key!=27)
      {
         //muestro el frame.
        cv::imshow("SHOWEXTREMES - VIDEO", frame);
        fsiv_find_min_max_loc_2(frame, min_v, max_v, min_loc, max_loc);
        cout << "Valor mínimo: " << "B:" << min_v.at(0) << ", G:" << min_v.at(1) <<", R:"<< min_v.at(2) << " (" << min_loc.at(0) << ")" << endl;
        cout << "Valor máximo: " << "B:" << max_v.at(0) << ", G:" << max_v.at(1) <<", R:"<< max_v.at(2) << " (" << max_loc.at(0) << ")" << endl;
        std::cout << "Pulsa ESC para salir." << std::endl;
        key = cv::waitKey(wait) & 0xff;
        vid >> frame;
      }
    }
    cv::destroyWindow("SHOWEXTREMES - VIDEO");
  }
  catch (std::exception& e)
  {
    std::cerr << "Caught exception: " << e.what() << std::endl;
    retCode = EXIT_FAILURE;
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception!" << std::endl;
    retCode = EXIT_FAILURE;
  }
  return retCode;
}
