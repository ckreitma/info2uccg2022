#import sys
fosforo = int(input("Cuantos fosforos habra en mesa?:"))
cantidad = 0
turno = 1
fosforoAquitar = 1


def siguiente(fosforo, turno):

    global cantidad
    print(f'cantidad = {cantidad}  Juega: = {turno} fosforo={fosforo}')
    if fosforo == 0:
        print(f'Jugador {3 - turno} perdió')  # no hay mas fosforos
        return

    if fosforo >= 3:
        fosforo -= 3
        cantidad += 1
        siguiente(fosforo, 3-turno)
        fosforo += 3
    if fosforo >= 2:
        fosforo -= 2
        cantidad += 1
        siguiente(fosforo, 3-turno)
        fosforo += 2
    if fosforo >= 1:
        fosforo -= 1
        cantidad += 1
        siguiente(fosforo, 3-turno)
        fosforo += 1


siguiente(fosforo, turno)
