# coding: utf-8
import re
import string
import numpy as np
from itertools import product
from binascii import crc32
import tqdm

class vector_cadena:
    """
    Clase para vectorizar a partir de los n-gramas
    """
    #------------------------------------
    def __init__(self, lista, metodo='posicion', tam=2):
        """
        Constructor: el diccionario es una lista de letras permitidas 
        El feature space es un arreglo con todos los posibles n-gramas 
        """
        assert (metodo in ['frec','posicion','ambos'])
        
        self.conjunto      = lista.lower()
        self.__metodo__    = metodo
        self.__n__         = tam
        
        self.__ngramas__ = [''.join(gramema) for gramema in product(lista, repeat=tam)]
        self.__feat_space = dict(zip(self.__ngramas__, [0]*len(self.__ngramas__)))
        

    #------------------------------------    
    def genera_vector(self, cad, tipo='dict'):
        """
        Genera el vector de caracteristicas aplicado a
        la cadena cad usando el espacio de características
        definidas en el construtor
        
        - Parameters:
            cad: string para vectorizar
        
        - Returns:
            Vector con conteo de entradas
        """
        assert tipo in ('dict','array')
        fs = self.__feat_space.copy()
        k = self.__n__
        ngramas = self.__ngramas__
        vec = [cad[i:i+k] for i in range(0, len(cad)-k+1)]
        for v in vec:
            fs[v] += 1
        if(tipo=='dict'):
            return fs
        elif(tipo=='array'):
            return np.array(list(fs.values()))
    #------------------------------------    
    def genera_matriz(self, lista_cadenas, tipo='array'):
        """
        Genera los vectores de las 
        cadenas contenidas en lista_cadenas
        y los guarda en arreglo 

        - Parameters:
            lista_cadenas: Lista con cadenas

        - Returns:
            Array con vectores de características
        """
        assert tipo in ('array','dataframe')
        ngramas = self.__ngramas__
        k = len(self.__feat_space)
        matriz = np.zeros((len(lista_cadenas),k))
#         print(matriz.shape)
        for j, cad in enumerate(tqdm.tqdm(lista_cadenas)):
            v = self.genera_vector(cad, tipo='array')
            matriz[j] = v
        self.__matriz__ = matriz
        if(tipo=='array'):
            return matriz
        elif(tipo=='dataframe'):
            return pd.DataFrame(matriz, index=lista_cadenas)    
    #------------------------------------    
    def similitud_vectores(self, i,j, metrica='euclideana'):
        """
        Compara los vectores en las posiciones i y j
        - Parameters:
            i, j: entradas en la matriz calculada en genera_matriz

        - Returns:
            Float que indica la comparación entre los vectores matriz[i] y matriz[j]
            con el método en el parámetro
        """
        assert metrica in ['euclideana', 'coseno']
        matriz = self.__matriz__
        vec1, vec2 = matriz[i], matriz[j]
        if(metrica =='euclideana'):
            return np.linalg.norm(vec1 - vec2)
        elif(metrica == 'coseno'):
            dot = np.dot(vec1, vec2)
            n1  = np.linalg.norm(vec1)
            n2  = np.linalg.norm(vec2)
            cs  = dot / (n1*n2)
            return cs
    #------------------------------------    
    def calcula_matriz_similitud(self, metrica='euclideana'):
        """
        Calcula la matriz con la similitud de la matriz de vectores 
        de caracteristicas previamente calculadas 
        
        - Parameters:
            
        - Returns:
            Float que indica la comparación entre los vectores matriz[i] y matriz[j]
            con el método en el parámetro
        """
        matriz = self.__matriz__
        N = np.zeros((len(matriz), len(matriz)))
        I,J = N.shape
        for i in range(I):
            vec1 = matriz[i]
            for j in range(I):
                vec2 = matriz[j]
                N[i,j] = self.similitud_vectores(i,j, metrica)
        self.comparacion_calculada = True
        self.comparacion = N
    #------------------------------------    
    def matriz_similitud(self, metrica='euclideana'):
        """
        Recupera la matriz calculada con calcula_matriz_similitud
        
        """
        if(self.comparacion_calculada):
            return self.comparacion
    #------------------------------------    
    @property
    def feat_space(self):
        return self.__ngramas__

if __name__=='__main__':
    folios = ['160010004203_16_2', '1600124004203_16_2', '170010004203_16_1', '1700124004203_20_2',
              '080010001203_16_2', '0800124004203_16_2', '120010230423_16_1', '1200102304233_34_1']
C = VectorizedString(folios)