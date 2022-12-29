#! /usr/bin/python3

import tqdm

from util import *
from mar import *

def evalua(dirRec, dirMar, *guiSen):
    """
    Calcula la tasa de exactitud en el reconocimiento de las señales indicadas por el
    fichero guía 'guiSen' o la lista de nombres de fichero 'ficRec' y la escribe en
    pantalla. La tasa de exactitud se define como el número de aciertos (corr)
    dividido por el número total de señales (total).

                                  Exac = corr / total

    También escribe la matriz de confusión correspondiente, con cada fila indicando la
    unidad que debería ser reconocida y cada columna la efectivamente reconocida.
    """

    matCnf = {}
    lisPal = set()

    for sen in tqdm.tqdm(leeLis(*guiSen), ascii=' >='):
        pathRec = pathName(dirRec, sen, '.rec')
        rec = cogeTrn(pathRec)

        pathMar = pathName(dirMar, sen, '.mar')
        mar = cogeTrn(pathMar)

        if not mar in matCnf:
            matCnf[mar] = {rec: 1}
        elif not rec in matCnf[mar]:
            matCnf[mar][rec] = 1
        else:
            matCnf[mar][rec] += 1

        lisPal |= {rec, mar}


    for rec in sorted(lisPal):
        print(f'\t{rec}', end='')
    print()
    for mar in sorted(lisPal):
        print(f'{mar}', end='')
        for rec in sorted(lisPal):
            conf = matCnf[mar][rec] if mar in matCnf and rec in matCnf[mar] else 0

            print(f'\t{conf}', end='')
        print()
    print()

    total, corr = 0, 0
    for mar in matCnf:
        for rec in matCnf[mar]:
            conf = matCnf[mar][rec]

            total += conf
            if rec == mar: corr += conf

    print(f'Exac = {corr / total:.2%}')


#################################################################################
# Invocación en línea de comandos
#################################################################################

if __name__ == '__main__':
    from docopt import docopt
    import sys

    Sinopsis = f"""
Evalua el resultado de un reconocimiento

Usage:
    {sys.argv[0]} [options] <guiSen>...
    {sys.argv[0]} -h | --help
    {sys.argv[0]} --version

Opciones:
    -r PATH, --dirRec=PATH  Directorio con los ficheros del resultado [default: .]
    -a PATH, --dirMar=PATH  Directorio con los ficheros de marcas [default: .]

Argumentos:
    <guiSen>  Nombre del fichero guía con los nombres de las señales reconocidas.
              Pueden especificarse tantos ficheros guía como sea necesario.

Evaluación:
    Siendo OK el número de unidades reconocidas correctamente y KO el de errores,
    el programa saca por pantalla la exactitud calculada como OK / (OK + KO)
"""

    args = docopt(Sinopsis, version=f'{sys.argv[0]}: Ramses v3.4 (2020)')

    dirRec = args['--dirRec']
    dirMar = args['--dirMar']

    guiSen = args['<guiSen>']

    evalua(dirRec, dirMar, *guiSen)
