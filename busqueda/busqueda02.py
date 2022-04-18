from river01 import River


class Nodo:
    def __init__(self, problema=None, padre=None, accion=None):
        self.padre = padre
        self.accion = accion

        if accion and padre:
            self.estado = problema.transicion(padre.estado, accion)[0]
            self.costo = padre.costo + problema.transicion(padre.estado, accion)[1]
        else:
            self.estado = None
            self.costo = 0

    def __str__(self):
        if self.padre:
            return f'<{self.estado},{self.padre},{self.accion},{self.costo}>'
        else:
            return f'<{self.estado},N/A,N/A,{self.costo}>'

    def solucion(self):
        # Imprime la lista de padres hasta llegar a la raiz.
        camino = []
        camino.append(self.estado)
        nodo = self
        while nodo.padre:
            padre = nodo.padre
            camino.append(padre.estado)
            nodo = padre
        camino.reverse()
        print(f'Camino=', end='')
        for c in camino:
            print(f'{c}', end='->')


"""
function BREADTH-FIRST-SEARCH(problem) returns a solution, or failure
	node ← a node with STATE = problem.INITIAL-STATE
	PATH-COST = 0
	if problem.GOAL-TEST(node.STATE) then return SOLUTION(node)
	frontier ← a FIFO queue with node as the only element
	explored ← an empty set
	loop do
		if EMPTY?( frontier) then return failure
		node ← POP( frontier) /* chooses the shallowest node in frontier */
		add node.STATE to explored
		for each action in problem.ACTIONS(node.STATE) do
			child ← CHILD-NODE(problem, node, action)
			if child.STATE is not in explored or frontier then
				if problem.GOAL-TEST(child.STATE) then return SOLUTION(child)
				frontier ← INSERT(child, frontier)
"""


def esta_en(lista, hijo):
    resultado = False
    for nodo in lista:
        if nodo.estado == hijo.estado:
            return True
    return resultado


def imprimir(lista, nombre):
    print(f'Inicio = {nombre}', end=' ')
    for nodo in lista:
        print(f'{nodo.estado}', end='###')
    print(f'Fin = {nombre}')


def bfs(problema):
    raiz = Nodo()
    raiz.padre = None
    raiz.accion = None
    raiz.costo = 0
    raiz.estado = problema.estado_inicial

    print(f'raiz={raiz}')

    if problema.goal_test(raiz.estado):
        raiz.solucion()
    frontera = []
    frontera.append(raiz)
    explorados = []

    while True:
        if len(frontera) <= 0:
            return "Sin Solucion"
        nodo = frontera.pop()
        explorados.append(nodo)
        for accion in problema.acciones_posibles(nodo.estado):
            #print(f'Procesando accion={accion}')
            hijo = Nodo(problema, nodo, accion)

            # Verificamos que el hijo no esté entre los explorados ni en la frontera.
            if (not esta_en(frontera, hijo)) and (not esta_en(explorados, hijo)):
                if problema.goal_test(hijo.estado):
                    hijo.solucion()
                #print(f'Agregando {hijo.estado} a frontera')
                frontera.append(hijo)

            #imprimir(frontera, 'frontera')
            #imprimir(explorados, 'explorados')


if __name__ == "__main__":
    river = River()
    print(f'Estado inicial = {river.estado_inicial}')

    for accion_posible in river.acciones_posibles(river.estados[0]):
        nuevo_estado = river.transicion(river.estados[0], accion_posible)[0]
        print(f'Nuevo estado={nuevo_estado} accion={accion_posible}')

    #print(f'Acciones posibles = {river.estados[5]} {river.acciones_posibles(river.estados[5])}')
    # print(f'{e}')
    # print(f'Acciones={mapa.acciones_posibles("Seattle")}')
    # bfs(mapa)
