#! /usr/bin/python3

import tqdm
from datetime import datetime as dt

from util import *

def recorch(dirRec, lotesRec, modelo):
    """
    Determina la unidad cuyo modelo se ajusta mejor a cada señal a reconocer y escribe
    su nombre en el cuarto campo de una etiqueta LBO de un fichero de marcas ubicado
    en el directorio 'dirRec' y del mismo nombre que la señal, pero con extensión '.rec'.
    """

    for lote in tqdm.tqdm(lotesRec, ascii=' >='):
        for señal in lote:
            rec = modelo(señal)

            pathRec = pathName(dirRec, señal.sen, '.rec')
            chkPathName(pathRec)
            with open(pathRec, 'wt') as fpRec:
                fpRec.write(f'LBO: ,,,{rec}\n')


#################################################################################
# Invocación en línea de comandos
#################################################################################

if __name__ == '__main__':
    from docopt import docopt
    import sys

    Sinopsis = rf"""
    Reconoce las señales.

    Usage:
        {sys.argv[0]} [options] <dirRec>
        {sys.argv[0]} -h | --help
        {sys.argv[0]} --version

    Opciones:
        -x SCRIPT..., --execPre=SCRIPT...   Scripts Python a ejecutar antes del reconocimiento
        -R EXPR..., --lotesRec=EXPR...      Expresión que proporciona los lotes de reconocimiento
        -M EXPR..., --modelo=EXPR...        Expresión que crea o lee el modelo inicial
        
    Argumentos:
        <dirRec>  Directorio en el que se escribirán los ficheros de resultado del reconocimiento

    Notas:
        La opción --execPre premite indicar uno o más scripts a ejecutar antes del reconocimiento.
        Para indicar más de uno, los diferentes scripts deberán estar separados por coma.

        Las opciones --lotesRec y --modelo permiten indicar una o más expresiones Python a evaluar
        para obtener los lotes de reconocimiento y el modelo, respectivamente. Para indicar más de
        una expresión, éstas deberán estar separadas por punto y coma.
"""

    args = docopt(Sinopsis, version=f'{sys.argv[0]}: Ramses v4.1 (2022)')

    scripts = args['--execPre']
    if scripts:
        for script in scripts.split(','):
            exec(open(script).read())
    
    dirRec = args['<dirRec>']

    for expr in args['--lotesRec'].split(';'):
        lotesRec = eval(expr)

    for expr in args['--modelo'].split(';'):
        modelo = eval(expr)

    recorch(dirRec=dirRec, lotesRec=lotesRec, modelo=modelo)