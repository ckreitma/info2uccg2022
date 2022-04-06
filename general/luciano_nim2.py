#import sys
fosforo = int(input("Cuantos fosforos habra en mesa?:"))
cantidad = 0
turno = 1


def siguiente(fosforo, turno):

    global cantidad
    print(f'cantidad = {cantidad}  Juega: = {turno} fosforo={fosforo}')
    if fosforo == 0:
        print(f'Jugador {3 - turno} perdió')  # no hay mas fosforos
        return

    for fosforoAquitar in range(1, 4):
        if fosforo >= fosforoAquitar:
            fosforo -= fosforoAquitar
            cantidad += 1
            siguiente(fosforo, 3-turno)
            fosforo += fosforoAquitar


siguiente(fosforo, turno)
