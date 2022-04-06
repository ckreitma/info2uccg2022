##################################
##### DEFINE EL PROBLEMA #########
##### NO LA SOLUCION #############
##################################

class Mapa:

    # Definimos el constructor.
    def __init__(self):
        self.nombre = 'Mapa'

        # Estados
        self.estados = [
            'Seattle',
            'SanFrancisco',
            'LosAngeles',
            'Riverside',
            'Phoenix',
            'Houston',
            'Dallas',
            'Miami',
            'Atlanta',
            'Chicago',
            'Detroit',
            'Boston',
            'NewYork',
            'Philadelphia',
            'Washington',
        ]

        self.estado_inicial = self.estados[0]

        self.acciones = []
        for e in self.estados:
            self.acciones.append(f'Ir {e}')

        # Fijarse en el archivo mapa_con_numero
        self.transiciones = []
        self.transiciones.append((0, 1, 678))
        self.transiciones.append((0, 9, 1737))
        self.transiciones.append((1, 2, 348))
        self.transiciones.append((1, 3, 386))
        self.transiciones.append((2, 4, 357))
        self.transiciones.append((2, 3, 50))
        self.transiciones.append((3, 4, 307))
        self.transiciones.append((4, 5, 1015))
        self.transiciones.append((4, 6, 887))
        self.transiciones.append((5, 6, 225))
        self.transiciones.append((5, 7, 968))
        self.transiciones.append((5, 8, 702))
        self.transiciones.append((6, 8, 721))
        self.transiciones.append((6, 9, 805))
        self.transiciones.append((7, 8, 604))
        self.transiciones.append((7, 14, 923))
        self.transiciones.append((8, 9, 588))
        self.transiciones.append((8, 14, 543))
        self.transiciones.append((9, 10, 238))
        self.transiciones.append((10, 11, 613))
        self.transiciones.append((10, 12, 482))
        self.transiciones.append((10, 14, 396))
        self.transiciones.append((11, 12, 190))
        self.transiciones.append((12, 13, 81))
        self.transiciones.append((13, 14, 123))

    def transicion(self, estado, accion):
        origen = estado
        destino = accion[3:40]
        for (o, d, c) in self.transiciones:
            #print(f'Buscando transiciones {origen}->{destino} ({o},{d},{c})')
            if (self.estados[o] == origen and self.estados[d] == destino) or (self.estados[d] == origen and self.estados[o] == destino):
                return (destino, c)
        return (False, 10000)

    def goal_test(self, estado):
        return estado == 'Washington'

    def acciones_posibles(self, estado):
        lista_acciones = []
        for accion in self.acciones:
            if self.transicion(estado, accion)[0]:
                lista_acciones.append(accion)
        return lista_acciones
