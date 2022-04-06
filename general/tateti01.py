
t = [0, 0, 0, 0, 0, 0, 0, 0, 0]
cantidad = 0

n = 6

# Recibe el tablero, y quién juega: turno = 1 (X), turno = 2 (O)


def siguiente(t, turno):
    global cantidad
    # imprimir(t)
    if cantidad % 10000 == 0:
        print(f'cantidad = {cantidad}')
    ocupadas = 0
    for i in range(9):
        if t[i] != 0:
            ocupadas += 1
    # Todas las casillas están ocupadas
    if ocupadas == 9:
        return

    # Ponemos la jugada en el tablero.
    for i in range(9):
        if t[i] == 0:  # Casilla libre.
            t[i] = turno
            cantidad += 1
            siguiente(t, 3-turno)
            t[i] = 0


# Imprime el tablero. 1 significa X, 2 significa O
def imprimir(t):
    for i in range(9):
        if i == 3 or i == 6:
            print('\n')
        j = '*'
        if t[i] == 1:
            j = 'X'
        if t[i] == 2:
            j = 'O'
        print(f'{j}', end='')
    print('')


siguiente(t, 1)
