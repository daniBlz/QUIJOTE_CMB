# Proyecto de Análisis del Fondo Cósmico de Microondas (CMB)

Este proyecto contiene herramientas y scripts para el análisis de mapas del Fondo Cósmico de Microondas (CMB) utilizando datos de archivos FITS y la biblioteca `healpy`. Las funciones incluidas permiten la lectura, visualización, y extracción de submapas del CMB, así como la manipulación y análisis de catálogos astronómicos.

## Descripción de Archivos

1. **tools.py**:
    - Contiene funciones para:
        - Leer y visualizar mapas del CMB desde archivos FITS.
        - Extraer y visualizar submapas del CMB.
        - Obtener y filtrar coordenadas de catálogos astronómicos en formato FITS.
        - Convertir coordenadas galácticas a vectores unitarios y realizar consultas espaciales.
        - Guardar imágenes en formato JPG.

2. **quijote_predictions.ipynb**:
    - Notebook que realiza predicciones y análisis utilizando el conjunto de datos de simulaciones de Quijote.

3. **WMAP.ipynb**:
    - Notebook para el análisis de datos del satélite WMAP (Wilkinson Microwave Anisotropy Probe).

4. **planck_datagen.ipynb**:
    - Notebook para la generación de datos a partir de observaciones del satélite Planck.

5. **planck_predictions.ipynb**:
    - Notebook que realiza predicciones y análisis utilizando datos del satélite Planck.

6. **training.ipynb**:
    - Notebook que contiene el proceso de entrenamiento de modelos para el análisis del CMB.

7. **analisis_planck.ipynb**:
    - Notebook que realiza un análisis detallado de los datos del satélite Planck.

8. **analisis_qui.ipynb**:
    - Notebook que realiza un análisis detallado de los datos de simulaciones de Quijote.

9. **analisis_wmap.ipynb**:
    - Notebook que realiza un análisis detallado de los datos del satélite WMAP.

## Uso

Cada notebook contiene celdas de código que pueden ser ejecutadas secuencialmente para realizar los análisis descritos. El archivo `tools.py` debe ser importado en los notebooks donde se requieran sus funciones.


El resto de directorios del proyecto se pueden descargar a través de la siguiente url:

Para reducir el tamaño del archivo las imágenes del dataset de entrenamiento y de prueba son una muestra generada después de la realización del proyecto. 
