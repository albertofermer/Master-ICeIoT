using namespace std;
#include "common_code.hpp"
#include <opencv2/core.hpp>

#include <iostream>

void 
fsiv_find_min_max_loc_1(cv::Mat const& input,
    std::vector<cv::uint8_t>& min_v, std::vector<cv::uint8_t>& max_v,
    std::vector<cv::Point>& min_loc, std::vector<cv::Point>& max_loc)
{
    CV_Assert(input.depth()==CV_8U);

    //! TODO: do a rows/cols scanning to find the first min/max values. 
    // Hint: use cv::split to get the input image channels.
	
	// Sacamos el número de canales de la imagen
	int num_canales = input.channels(); // Obtenemos el número de canales de la imagen
	std::vector<cv::Mat> canales(num_canales); // Utilizamos un vector de matrices num_canales de longitud para manejarlo de forma dinámica
	split(input, canales); // Dividimos la imagen en num_canales matrices para poder extraer los valores mínimos y máximos de cada uno de ellos

	min_v.resize(num_canales, 255);  // Inicializamos min_v a 255 (máximo valor posible)
    	max_v.resize(num_canales, 0);    // Inicializamos max_v a 0 (mínimo valor posible)
    	min_loc.resize(num_canales, cv::Point(-1,-1));     // Almacenará el pixel de los minimos. Iniciamos en -1,-1 porque no es una posicion valida en una imagen.
    	max_loc.resize(num_canales, cv::Point(-1,-1));     // Almacenará el pixel de los maximos

	for(int chan = 0; chan < num_canales; chan++){ // Para cada canal de la imagen

		for(int row = 0; row < canales[chan].rows; row++){
			for (int col= 0; col < canales[chan].cols; col++){
				cv::uint8_t valor_pixel = canales[chan].at<cv::uint8_t>(row,col); // obtenemos el valor del pixel para no calcularlo dos veces
				if (min_v[chan] > valor_pixel){ // comparamos con el minimo por el momento
					min_v[chan] = valor_pixel;	// si cumple, entonces lo actualizamos
					min_loc[chan] = cv::Point(col,row); // y la ubicacion tambien

				}
				if (max_v[chan] < valor_pixel) { // comparamos con el máximo por el momento
					max_v[chan] = valor_pixel; // actualizamos el valor
					max_loc[chan] = cv::Point(col,row); // y la ubicacion
				}
			}	
		}
	}
	//DEBUG
	//cout << endl << "" << endl;
	
    CV_Assert(input.channels()==min_v.size());
    CV_Assert(input.channels()==max_v.size());
    CV_Assert(input.channels()==min_loc.size());
    CV_Assert(input.channels()==max_loc.size());
}

void 
fsiv_find_min_max_loc_2(cv::Mat const& input,
    std::vector<double>& min_v, std::vector<double>& max_v,
    std::vector<cv::Point>& min_loc, std::vector<cv::Point>& max_loc)
{

    //! TODO: Find the first min/max values using cv::minMaxLoc function.    
    // Hint: use cv::split to get the input image channels.
	int num_canales = input.channels(); // Obtenemos el número de canales de la imagen
	std::vector<cv::Mat> canales(num_canales); // Utilizamos un vector de matrices num_canales de longitud para manejarlo de forma dinámica
	split(input, canales); // Dividimos la imagen en num_canales matrices para poder extraer los valores mínimos y máximos de cada uno de ellos
	
	min_v.resize(num_canales);  // Inicializamos min_v
    	max_v.resize(num_canales);    // Inicializamos max_v
    	min_loc.resize(num_canales);     // Almacenará el pixel de los minimos.
    	max_loc.resize(num_canales);     // Almacenará el pixel de los maximos

	for (int chan = 0; chan < num_canales; chan++){
		cv::minMaxLoc(canales[chan], &min_v[chan], &max_v[chan], &min_loc[chan], &max_loc[chan]);
	}


    //

    CV_Assert(input.channels()==min_v.size());
    CV_Assert(input.channels()==max_v.size());
    CV_Assert(input.channels()==min_loc.size());
    CV_Assert(input.channels()==max_loc.size());

}
