##################################
##### DEFINE EL PROBLEMA #########
##### NO LA SOLUCION #############
##################################


class River:

    # Definimos el constructor.
    def __init__(self):
        self.nombre = 'River'

        # Estados
        self.estados = self.generar_estados()

        #####
        # [2, 2, 2, 2, 0, 2, 1, 1, 1]
        # Papa derecha, hijo1 derecha, hijo2 derecha
        # Mama derecha, hija1 izquierda, hija2 derecha
        # Policía bote
        # Ladron Bote
        # Bote a la navegando (inválido) Bote = 0,2 (izquierda, derecha)
        self.estado_inicial = self.estados[0]

        # Papá ==> El papa sube al bote.
        # Hija1 ==> Hija1 se sube al bote
        # Bote ==> El bote se mueve a la orilla, opuesta y baja a los pasajeros
        self.acciones = ['PA', 'H1', 'H2', 'MA', 'M1', 'M2', 'PO', 'LA', 'BO']

    def goal_test(self, estado):
        return estado == 'Washington'

    def acciones_posibles(self, estado):
        lista_acciones = []

        # Si el bote está lleno entonces no puede subir nadie más.
        if cantidad_en_bote(estado) <= 1:
            for persona in range(0, 8):
                if estado[persona] == estado[8]:  # Si la persona y el bote están del mismo lado, se puede subir.
                    lista_acciones.append(self.acciones[persona])
        return lista_acciones

    def transicion(self, estado, accion):
        copia_estado = estado
        if accion == 'LA':  # El ladron sube al bote.
            copia_estado[7] = 1
        return (copia_estado, 1)

    # Genera estados.
    def generar_estados(self):
        estados = []
        for n in range(0, 19683):
            estado = numberToBase(n, 3)
            if estado[8] == 1:
                continue
            estados.append(estado)
        return estados


# Auxiliares.
def numberToBase(n, b):
    if n == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    modificado = digits
    for agregado in range(0, 9-len(modificado)):
        modificado.append(0)
    modificado = digits[:: -1]
    return modificado


def cantidad_en_bote(estado):
    cantidad = 0
    for persona in estado:
        if persona == 1:
            cantidad += 1
    return cantidad
