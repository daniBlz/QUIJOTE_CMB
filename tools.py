import pandas as pd
import healpy as hp
import astropy
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import astropy.units as u
from skimage.transform import resize

def leer_mapa(ruta_archivo_fits, campo=0):
    """
    Lee un mapa CMB de un archivo FITS usando healpy.

    Parámetros:
    - ruta_archivo_fits: str, la ruta hacia el archivo FITS.
    - campo: int, el índice del campo (mapa) en el archivo FITS a leer.

    Retorna:
    - mapa_cmb: ndarray, el mapa CMB leído del archivo FITS.
    """
    # Leer el mapa CMB del archivo FITS
    mapa_cmb, header = hp.read_map(ruta_archivo_fits, field=campo, h=True, verbose=False)

    # Opcional: Imprimir información del header si es necesario
    # print(header)

    return mapa_cmb


def ver_mapa(mapa, vista):
    if vista == 'mollweide':
        # Visualiza el mapa usando mollview para una proyección de Mollweide
        hp.mollview(mapa, cmap='jet', unit='mK', norm='hist', title="Mapa Celeste")
    elif vista == 'carto':
        # Visualiza el mapa usando cartview para una vista cartográfica
        hp.cartview(mapa, cmap='jet', unit='mK', norm='hist', title="Mapa Celeste")
    elif vista == 'orto':
        # Visualiza el mapa usando gnomview para una vista ortográfica (usamos gnomview como aproximación)
        hp.gnomview(mapa, cmap='jet', unit='mK', norm='hist', title="Mapa Celeste", rot=(0, 90), reso=5)
    elif vista == 'gnomon':
        # Visualiza el mapa usando gnomview para una vista gnómica
        hp.gnomview(mapa, cmap='jet', unit='mK', norm='hist', title="Mapa Celeste")
    else:
        print("Vista no soportada. Por favor, elige entre 'mollweide', 'carto', 'orto', o 'gnomon'.")

    # Añade la grilla de coordenadas celestes, solo si no es vista gnómica (gnomview incluye su propia grilla)
    if vista != 'gnomon':
        hp.graticule()

        


def extraer_submapa_cmb(mapa_cmb, N, vmin, res, lat = None,lon= None, nside=1024, retornar_imagen=False):
    """
Extrae y visualiza un submapa del Fondo Cósmico de Microondas (CMB) centrado en un píxel específico o uno seleccionado aleatoriamente.

Parámetros:
- mapa_cmb: ndarray, el mapa completo del CMB del cual se quiere extraer un submapa.
- N: int, tamaño del lado del submapa cuadrado a extraer, que también determina la resolución de la visualización.
- pix_id_central (opcional): int, el identificador del píxel central del submapa a extraer. Si no se proporciona, se selecciona uno al azar con declinación entre -30 y 30 grados.
- nside: int, parámetro de HEALPix que define la resolución del mapa completo del CMB. Un nside mayor indica una mayor resolución.
- retornar_imagen: bool, indica si la función debe retornar el submapa como un arreglo (True) o simplemente visualizar el submapa (False).

Retorna:
- Si retornar_imagen es True, retorna el submapa del CMB como un arreglo. De lo contrario, no retorna nada pero visualiza el submapa.

La función utiliza una proyección gnómica para visualizar el submapa, centrada en el píxel especificado, con un tamaño y resolución determinados por los parámetros N y tamaño_vista_grados. La visualización se ajusta para enfocarse en las características locales del CMB, utilizando una escala de colores para representar las variaciones de temperatura.
"""

    if lat is None:
        # Decide qué rango usar: 0 para el primer rango y 1 para el segundo
        range_choice = np.random.randint(0, 2)
        if range_choice == 0:
            # Genera un número aleatorio entre -90 y -30
            lat = np.random.uniform(-90, -30)
        else:
            # Genera un número aleatorio entre 30 y 90
            lat = np.random.uniform(30, 90)

    if lon is None:
        lon = np.random.uniform(0,360)
    
    # Retornar el mapa proyectado si se solicita
    if retornar_imagen:
        imagen = hp.visufunc.gnomview(mapa_cmb, rot=(lon,lat),
                                      xsize=N, ysize=N, reso=res, min = vmin,
                                      title=f"Extracción del CMB - Long: {lon} - Lat: {lat}, {N}x{N}",
                                      unit="mK", notext=True, cmap="jet", no_plot=True,
                                      return_projected_map=True)
       
        
        return imagen
    else:
        hp.visufunc.gnomview(mapa_cmb, rot=(lon,lat),
                             xsize=N, ysize=N, reso=res, min = vmin,
                             title=f"Extracción del CMB - Long: {lon} - Lat: {lat}, {N}x{N}", 
                             unit="mK", notext=True, cmap="jet")
        
        

def plot_submapa(submapa, title="Submapa del CMB"):
    """
    Visualiza un submapa del CMB utilizando matplotlib.

    Parámetros:
    - submapa: ndarray, el submapa del CMB a visualizar.
    - title: str, título de la figura.
    """
    plt.figure(figsize=(8, 6))  # Configura el tamaño de la figura
    plt.imshow(submapa, origin='lower', cmap='jet',vmin = -0.0001)
    plt.colorbar(label='mK')  # Añade una barra de colores con la etiqueta 'mK'
    plt.title(title)  # Establece el título de la figura
    plt.xlabel('X [grados]')
    plt.ylabel('Y [grados]')
    plt.show()




def obtener_coord(catalogo_fits):
    """
    Obtiene las coordenadas galácticas (glat, glon) de objetos en un catálogo FITS,
    filtrando aquellos que se encuentran fuera del intervalo de latitud galáctica (-30, 30 grados).

    La función lee un catálogo de objetos astronómicos proporcionado en formato FITS,
    extrayendo las coordenadas galácticas latitud (GLAT) y longitud (GLON). Luego, filtra
    los objetos cuya latitud galáctica se encuentra fuera del rango permitido y retorna un DataFrame.

    Parámetros:
    - catalogo_fits: str, la ruta al archivo FITS que contiene el catálogo de objetos astronómicos.

    Retorna:
    - df_filtrado: DataFrame, un DataFrame de pandas para los objetos filtrados, donde cada fila
      contiene las coordenadas 'glat', 'glon' y 'name'.

    Ejemplo de uso:
    catalogo_fits = './objetos.fits'  # Ruta al archivo .FITS de tu catálogo

    df_filtrado = obtener_coord(catalogo_fits)
    """
    
    # Leer el catálogo .FITS
    with fits.open(catalogo_fits) as hdul:
        data = hdul[1].data  # Asume que los datos están en la primera extensión
        glat = data['GLAT']
        glon = data['GLON']
        name = data['Name']
        flux = data['DETFLUX']
        flux_err = data['DETFLUX_ERR']
    # Crear el DataFrame con las coordenadas
    df = pd.DataFrame({'name': name, 'glat': glat, 'glon': glon,'flux':flux,'flux_err':flux_err})

    # Filtrar el DataFrame basado en la condición de latitud galáctica
    # Aquí asumimos que quieres filtrar; sin embargo, no especificaste la condición de filtrado.
    # Ajusta la condición de filtrado según tus necesidades.
    df['glat'] = df['glat'].apply(lambda x: float(x))
    df['glon'] = df['glon'].apply(lambda x: float(x))
    df['flux'] = df['flux'].apply(lambda x: float(x))
    df['flux_err'] = df['flux_err'].apply(lambda x: float(x))
    df_filtrado = df.loc[df.glat.abs()>30]

    return df_filtrado



def obtener_coordqui(catalogo_fits):
    """
    Obtiene las coordenadas galácticas (glat, glon) de objetos en un catálogo FITS,
    filtrando aquellos que se encuentran fuera del intervalo de latitud galáctica (-30, 30 grados).

    La función lee un catálogo de objetos astronómicos proporcionado en formato FITS,
    extrayendo las coordenadas galácticas latitud (GLAT) y longitud (GLON). Luego, filtra
    los objetos cuya latitud galáctica se encuentra fuera del rango permitido y retorna un DataFrame.

    Parámetros:
    - catalogo_fits: str, la ruta al archivo FITS que contiene el catálogo de objetos astronómicos.

    Retorna:
    - df_filtrado: DataFrame, un DataFrame de pandas para los objetos filtrados, donde cada fila
      contiene las coordenadas 'glat', 'glon' y 'name'.

    Ejemplo de uso:
    catalogo_fits = './objetos.fits'  # Ruta al archivo .FITS de tu catálogo

    df_filtrado = obtener_coord(catalogo_fits)
    """
    
    # Leer el catálogo .FITS
    with fits.open(catalogo_fits) as hdul:
        data = hdul[1].data  # Asume que los datos están en la primera extensión
        glat = data['GLAT']
        glon = data['GLON']
        name = data['ID']
        I11 = data['I 11 GHz']
        I11_err = data['I err 11 GHz']
        Q11 = data['Q 11 GHz']
        U11 = data['U 11 GHz']
        I13 = data['I 13 GHz']
        I13_err = data['I err 13 GHz']
        Q13 = data['Q 13 GHz']
        U13 = data['U 13 GHz']
        I17 = data['I 17 GHz']
        I17_err = data['I err 17 GHz']
        Q17 = data['Q 17 GHz']
        U17 = data['U 17 GHz']
        I19 = data['I 19 GHz']
        I19_err = data['I err 19 GHz']
        Q19 = data['Q 19 GHz']
        U19 = data['U 19 GHz']
    # Crear el DataFrame con las coordenadas
    df = pd.DataFrame({'name': name, 'glat': glat, 'glon': glon,'I 11 GHz':I11,'I err 11 GHz':I11_err,'Q 11 GHz':Q11,'U 11 GHz':U11,
                       'I 13 GHz':I13,'I err 13 GHz':I13_err,'Q 13 GHz':Q13,'U 13 GHz':U13,'I 17 GHz':I17,'I err 17 GHz':I17_err,
                       'Q 17 GHz':Q17,'U 17 GHz':U17,'I 19 GHz':I19,'I err 19 GHz':I19_err,'Q 19 GHz':Q19,'U 19 GHz':U19})

    # Filtrar el DataFrame basado en la condición de latitud galáctica
    # Aquí asumimos que quieres filtrar; sin embargo, no especificaste la condición de filtrado.
    # Ajusta la condición de filtrado según tus necesidades.
    df['glat'] = df['glat'].apply(lambda x: float(x))
    df['glon'] = df['glon'].apply(lambda x: float(x))

    df['I 11 GHz'] = df['I 11 GHz'].apply(lambda x: float(x))
    df['I err 11 GHz'] = df['I err 11 GHz'].apply(lambda x: float(x))
    df['Q 11 GHz'] = df['Q 11 GHz'].apply(lambda x: float(x))
    df['U 11 GHz']= df['U 11 GHz'].apply(lambda x: float(x))
    df['I 13 GHz'] = df['I 13 GHz'].apply(lambda x: float(x))
    df['I err 13 GHz'] = df['I err 13 GHz'].apply(lambda x: float(x))
    df['Q 13 GHz'] = df['Q 13 GHz'].apply(lambda x: float(x))
    df['U 13 GHz'] = df['U 13 GHz'].apply(lambda x: float(x))
    df['I 17 GHz'] = df['I 17 GHz'].apply(lambda x: float(x))
    df['I err 17 GHz'] = df['I err 17 GHz'].apply(lambda x: float(x))
    df['Q 17 GHz'] = df['Q 17 GHz'].apply(lambda x: float(x))
    df['U 17 GHz'] = df['U 17 GHz'].apply(lambda x: float(x))
    df['I 19 GHz'] = df['I 19 GHz'].apply(lambda x: float(x))
    df['I err 19 GHz'] = df['I err 19 GHz'].apply(lambda x: float(x))
    df['Q 19 GHz'] = df['Q 19 GHz'].apply(lambda x: float(x))
    df['U 19 GHz'] = df['U 19 GHz'].apply(lambda x: float(x))
    
    return df




def catalogo_a_vectores_galacticos(catalogo_fits):
    """
    Convierte las coordenadas de longitud galáctica (glon) y latitud galáctica (glat) de un catálogo de objetos astronómicos en formato FITS a vectores unitarios en el espacio tridimensional.

    Esta función lee un catálogo astronómico proporcionado en un archivo FITS y extrae las coordenadas de longitud y latitud galáctica de los objetos contenidos. Luego, convierte estas coordenadas angulares (en grados) a radianes y, finalmente, a vectores unitarios representando las direcciones de los objetos en la Vía Láctea. Esta conversión es útil para análisis espaciales y cálculos astronómicos cuando se trabaja con datos en coordenadas galácticas.

    Parámetros:
    - catalogo_fits: str, la ruta al archivo FITS que contiene el catálogo de objetos astronómicos.

    Retorna:
    - vectores: list, una lista de vectores unitarios tridimensionales que representan las posiciones de los objetos en el espacio. Cada vector es una lista de tres elementos [x, y, z].

    Ejemplo de uso:
    catalogo_fits = './objetos.fits'  # Ruta al archivo .FITS de tu catálogo
    vectores = catalogo_a_vectores_galacticos(catalogo_fits)
    """

    # Leer el catálogo .fits
    with fits.open(catalogo_fits) as hdul:
        data = hdul[1].data
        glon = data['GLON']  # Asegúrate de que 'GLON' y 'GLAT' sean los nombres correctos de las columnas
        glat = data['GLAT']
    
    # Convertir GLON, GLAT a radianes
    
    
    # Convertir coordenadas angulares a vectores unitarios
    vectores = [hp.ang2vec(g,l,lonlat = True) for g, l in zip(glon, glat)]
    
    return vectores

def query_disc_gen(mapa,vectores):
    #Previamente se deben pasar las coordenadas de las fuentes a vectores
    for vector in vectores:
        disc = hp.query_disc(1024,vector,radius = np.radians(1))
        mapa[disc] = np.nan
    return mapa






def guardar_imagen_jpg(imagen, nombre_archivo, nueva_resolucion=(150, 150), vmin = None):

    """
Guarda una imagen en formato JPG con una resolución especificada.

Esta función toma una imagen (generalmente un arreglo numpy), la reduce a una nueva resolución utilizando interpolación para preservar la calidad visual a pesar de la disminución del tamaño, y luego guarda la imagen resultante en formato JPG. La visualización de la imagen se realiza utilizando la escala de colores 'jet', y los ejes son eliminados para una presentación más limpia. La resolución final de la imagen guardada puede ajustarse modificando el parámetro 'dpi' en plt.savefig, lo que permite un control fino sobre el tamaño y la calidad de la imagen resultante.

Parámetros:
- imagen: ndarray, la imagen original a guardar.
- nombre_archivo: str, el nombre del archivo (incluyendo la ruta si es necesario) donde se guardará la imagen.
- nueva_resolucion: tuple, la nueva resolución de la imagen en píxeles como (ancho, alto). Por defecto es (150, 150).

Nota: Esta función requiere que 'resize' de skimage.transform, 'plt' de matplotlib.pyplot, y otras dependencias necesarias estén previamente importadas.
    """


    plt.figure()
    plt.imshow(imagen, cmap='jet',vmin =vmin)
    plt.axis('off')  # Elimina los ejes
    plt.savefig(nombre_archivo, bbox_inches='tight', pad_inches=0, dpi=80)  # Ajustar dpi para controlar la resolución final
    plt.close()