#!/bin/bash

# Crear la carpeta build si no existe
mkdir -p build

# Navegar a la carpeta build
cd build

# Ejecutar cmake
cmake ..

# Ejecutar make
make
