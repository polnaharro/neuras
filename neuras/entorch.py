#! /usr/bin/python3
import tqdm
from datetime import datetime as dt

def entorch(modelo, nomMod, lotesEnt, lotesDev=[], numEpo=1):
    print(f'Inicio de {numEpo} épocas de entrenamiento ({dt.now():%d/%b/%y %H:%M:%S}):')
    for epo in range(numEpo):
        for lote in lotesEnt:
            modelo.inicEntr()
            for señal in tqdm.tqdm(lote, ascii=' >='):
                modelo += señal
            modelo.recaMod()

        modelo.inicEval()
        for lote in lotesDev:
            for señal in tqdm.tqdm(lote, ascii=' >='):
                modelo.addEval(señal)
        
        if lotesDev:
            modelo.recaEval()
            modelo.printEval(epo)

        if nomMod: modelo.escrMod(nomMod)

    print(f'Completadas {numEpo} épocas de entrenamiento ({dt.now():%d/%b/%y %H:%M:%S})')


#################################################################################
# Invocación en línea de comandos
#################################################################################

if __name__ == '__main__':
    from docopt import docopt
    import sys

    Sinopsis = rf"""
    Entrena un modelo acústico para el reconocimiento del habla.

    Usage:
        {sys.argv[0]} [options] [<nomMod>]
        {sys.argv[0]} -h | --help
        {sys.argv[0]} --version

    Opciones:
        -e INT, --numEpo=INT                Número de épocas de entrenamiento [default: 50]
        -x SCRIPT..., --execPre=SCRIPT...   Scripts Python a ejecutar antes del modelado
        -E EXPR..., --lotesEnt=EXPR...      Expresión que proporciona los lotes de entrenamiento
        -D EXPR..., --lotesDev=EXPR...      Expresión que proporciona los lotes de evaluación
        -M EXPR..., --modelo=EXPR...        Expresión que crea o lee el modelo inicial

    Argumentos:
        <nomMod>  Nombre del fichero en el que se almacenará el modelo

    Notas:
        La opción --execPre premite indicar uno o más scripts a ejecutar antes del entrenamiento.
        Para indicar más de uno, los diferentes scripts deberán estar separados por coma.

        Las opciones --lotesEnt, --lotesDev y --modelo permiten indicar una o más expresiones 
        Python a evaluar para obtener los lotes de entrenamiento y evaluación y el modelo, 
        respectivamente. Para indicar más de una expresión, éstas deberán estar separadas por 
        punto y coma.
    """

    args = docopt(Sinopsis, version=f'{sys.argv[0]}: Ramses v4.1 (2022)')

    numEpo = int(args['--numEpo'])

    nomMod = args['<nomMod>']

    scripts = args['--execPre']
    if scripts:
        for script in scripts.split(','):
            exec(open(script).read())
    
    for expr in args['--lotesEnt'].split(';'):
        lotesEnt = eval(expr)

    for expr in args['--lotesDev'].split(';'):
        lotesDev = eval(expr)

    for expr in args['--modelo'].split(';'):
        modelo = eval(expr)

    entorch(modelo=modelo, nomMod=nomMod, lotesEnt=lotesEnt, lotesDev=lotesDev, numEpo=numEpo)

