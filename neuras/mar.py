<<<<<<< HEAD
import re

def cogeTrn(ficMar):
    """
    Devuelve el contenido del cuarto campo de la primera etiqueta LBO presente en el
    fichero de Marcas 'ficMar'.
    """

    reLBO = re.compile(r'LBO:(\s*\d*[.]?\d*,){3}(?P<trn>\w+)')

    with open(ficMar) as fpMar:
        for linea in fpMar:
            if (lbo := reLBO.match(linea)): return lbo['trn']
=======
import re

def cogeTrn(ficMar):
    """
    Devuelve el contenido del cuarto campo de la primera etiqueta LBO presente en el
    fichero de Marcas 'ficMar'.
    """

    reLBO = re.compile(r'LBO:(\s*\d*[.]?\d*,){3}(?P<trn>\w+)')

    with open(ficMar) as fpMar:
        for linea in fpMar:
            if (lbo := reLBO.match(linea)): return lbo['trn']
>>>>>>> 0f30eff0258a3b1533738d9012b9fc950e5e3f82
