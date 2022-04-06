from csv import reader


grafo = []


def posicion(nodo):
    if nodo == 'A':
        return 0
    if nodo == 'B':
        return 1
    if nodo == 'C':
        return 2
    if nodo == 'D':
        return 3
    if nodo == 'E':
        return 4
    if nodo == 'F':
        return 5
    if nodo == 'G':
        return 6
    if nodo == 'H':
        return 7
    if nodo == 'I':
        return 8
    if nodo == 'J':
        return 9
    if nodo == 'K':
        return 10


def exito(grafo, d):
    # print(f'grafo={grafo}')
    for lado in grafo:
        # print(f'lado={lado}')
        # print(f'lado0={lado[0]} lado1={lado[1]}')
        if d[posicion(lado[0])] == d[posicion(lado[1])]:
            return False
    return True


def asignar_color(nodos, pos, colores, d):
    # print(f'Intentando.... {nodos}, {pos},{colores},{d}')
    if pos > nodos:
        print(f'Distribución de colores {d}')
        if exito(grafo, d):
            print(f'Campeón')
            print(f'25 en base 4 {numberToBase(25,4)}')
            exit()

        return
    for color in range(colores):
        d[pos] = color
        asignar_color(nodos, pos+1, colores, d)

# Plus... esto es para contar en cualquier base.


def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]


# read csv file as a list of lists
with open('grafo.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)
    print(list_of_rows)

    nodos = []
    for arco in list_of_rows:
        nodo_origen = arco[0]
        nodo_destino = arco[1]
        if nodo_origen not in nodos:
            nodos.append(nodo_origen)
        if nodo_destino not in nodos:
            nodos.append(nodo_destino)

    print(f'arco={nodo_origen} <-> {nodo_destino} nodos={nodos}')

    grafo = list_of_rows
    d = nodos
    nodos = 8
    pos = 0
    colores = 3
    asignar_color(nodos, pos, colores, d)
