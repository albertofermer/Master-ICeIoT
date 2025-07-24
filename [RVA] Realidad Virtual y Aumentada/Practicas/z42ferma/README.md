# Markerless Augmented Reality Application

> Aplicación de Realidad Aumentada sin marcadores

### ⚙️ Compilación

Para compilar la aplicación ejecutar el fichero src/build.sh. Este fichero creará una carpeta build y ejecutará cmake y make
para obtener los ficheros binarios que permiten ejecutar el programa.

---

### 🧑‍💻 Autor
**Alberto Fernández Merchán**  
Universidad de Córdoba  
Curso 2024/25  
Asignatura: Realidad Virtual y Aumentada

---

### 📌 Descripción del proyecto

Este proyecto tiene como objetivo el desarrollo de una aplicación de realidad aumentada sin marcadores, 
centrada en la detección del cuadro "El caminante sobre un mar de nubes" en distintas escenas reales.

La aplicación permite superponer información sobre la obra, así como visualizar una bounding box
que destaca el cuadro detectado en el entorno. Además, se incluye la posibilidad de superponer contenido multimedia sobre la obra,
como una imagen alternativa o un vídeo.

---


### ⚙️ Ejecución genérica

La aplicación puede ejecutarse desde la línea de comandos con el siguiente formato:

```bash
./armuseum <modelo> <escena> [--descriptor=DESCRIPTOR] [--patch=PATCH] [--video=VIDEO] [--save-video=OUTPUT]
```
📥 Parámetros
modelo (obligatorio): Ruta a la imagen del modelo que se desea detectar (por ejemplo, el cuadro).
escena (obligatorio): Ruta a la imagen de la escena donde se busca el modelo.
--descriptor (opcional): Nombre del descriptor a utilizar. Los valores válidos son AKAZE, ORB y SIFT. Por defecto se usa AKAZE.
--patch (opcional): Ruta a una imagen de refuerzo (patch) para mejorar la detección en caso de escenas difíciles o parcialmente visibles.
--video (opcional): Ruta a un archivo de vídeo que se superpondrá sobre el modelo detectado.
--save-video (opcional): Ruta donde se guardará un vídeo resultante con el contenido superpuesto.

---

### ▶️ Ejecuciones de ejemplo

A continuación, se muestran algunos ejemplos de uso de la aplicación desde la línea de comandos:

- **Detección por defecto**
	```bash
	./armuseum ../../data/modelo.jpeg ../../data/scene_1.jpg
	```
- 🔍 **Detección con descriptor ORB y patch adicional**  
  Detecta el cuadro `modelo.jpeg` en la imagen `scene_3.jpg`, utilizando el descriptor ORB y sustituyendo la detección por una imagen de patch:
	```bash
	./armuseum ../../data/modelo.jpeg ../../data/scene_3.jpg --descriptor=ORB --patch=../../data/scene_1.jpg
	```
- 🎥 Superposición de vídeo con descriptor SIFT
Detecta el cuadro en scene_3.jpg usando el descriptor SIFT y superpone un vídeo (video.mp4). El resultado se guarda como un nuevo archivo:
	```bash
	./armuseum ../../data/modelo.jpeg ../../data/scene_3.jpg --descriptor="SIFT" --video=../../data/video.mp4 --save-video="../../data/results/video-scene3.mp4"
	```
---
### 📹 Enlace al vídeo
	
https://youtu.be/93AbOWvtY7w