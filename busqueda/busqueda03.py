from mapa03 import Mapa


class Nodo:
    def __init__(self, problema=None, padre=None, accion=None):
        self.padre = padre
        self.accion = accion

        if accion and padre:
            self.estado = problema.transicion(padre.estado, accion)[0]
            self.costo = padre.costo + problema.transicion(padre.estado, accion)[1]

            # Función de evaluación. Es el costo total al nodo + la función heuristica
            self.f = self.costo + problema.h(self.estado)
        else:
            self.estado = None
            self.costo = 0

    def __str__(self):
        if self.padre:
            return f'<{self.estado},{self.padre},{self.accion},{self.costo},{self.f}>'
        else:
            return f'<{self.estado},N/A,N/A,{self.costo},{self.f}>'

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


def a_star(problema):
    raiz = Nodo(problema=problema)
    raiz.padre = None
    raiz.accion = None
    raiz.costo = 0
    raiz.estado = problema.estado_inicial
    raiz.f = problema.h(problema.estado_inicial)

    # print(f'raiz={raiz}')

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
        mejor_evaluacion = 100000
        mejor_hijo = None
        for accion in problema.acciones_posibles(nodo.estado):
            hijo = Nodo(problema, nodo, accion)
            if hijo.f < mejor_evaluacion:
                mejor_hijo = hijo
                mejor_evaluacion = hijo.f

        # Verificamos que el hijo no esté entre los explorados ni en la frontera.
        print(f'Mejor hijo: {mejor_hijo}')
        if (not esta_en(frontera, mejor_hijo)) and (not esta_en(explorados, mejor_hijo)):
            if problema.goal_test(mejor_hijo.estado):
                mejor_hijo.solucion()
            frontera.append(mejor_hijo)

            #imprimir(frontera, 'frontera')
            #imprimir(explorados, 'explorados')


if __name__ == "__main__":
    mapa = Mapa()
    #print(f'Estado inicial = {mapa.estado_inicial}')
    # print(f'Acciones={mapa.acciones_posibles("Seattle")}')
    a_star(mapa)
