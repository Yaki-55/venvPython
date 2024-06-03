import osmnx as ox
import random
import heapq
import geopandas as gpd
from collections import deque


# extraer el grafo de la red de carreteras de la ubicación en place_name.
#G = ox.graph_from_place(place_name, network_type="drive")


def main(algorithm, start, end):
    place_name = "Huajuapan de León, Oaxaca, México"
    point_start = start[0], start[1]
    point_end = end[0], end[1]
    G = ox.graph_from_place(place_name, network_type="drive")
    # ox.plot_graph(G, node_size=1, node_color="red")

    for edge in G.edges:
        # Limpiando el atributo "maxspeed", algunos valores son listas, algunos son cadenas, algunos son None
        maxspeed = 40  # Establecer un valor de velocidad máxima predeterminado en 40 km/h
        if "maxspeed" in G.edges[edge]:  # Comprobando si el borde tiene el atributo "maxspeed"
            maxspeed = G.edges[edge]["maxspeed"]  # Obteniendo el valor de "maxspeed" para el borde
            if type(maxspeed) == list:  # Comprobando si maxspeed es una lista
                speeds = [int(speed) for speed in maxspeed]  # Convertir cada elemento de la lista a un número entero
                maxspeed = min(speeds)  # Tomando la velocidad mínima de la lista.
            elif type(maxspeed) == str:  # Comprobando si maxspeed es una cadena
                maxspeed = int(maxspeed)  # Convertir la cadena a un número entero
        G.edges[edge]["maxspeed"] = maxspeed  # Actualizando el atributo "maxspeed" del borde
        # Agregando el atributo "weight" (tiempo = distancia / velocidad)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed  # Calculando y asignando el atributo de peso

    # Convertir start y end a nodos

    # Encontrar los nodos más cercanos a las coordenadas de inicio y fin
    start_node = ox.nearest_nodes(G, point_start[1], point_start[0])
    end_node = ox.nearest_nodes(G, point_end[1], point_end[0])

    if algorithm == 'busqueda_amplitud':
        busqueda = busqueda_amplitud(G, start_node, end_node)
        resultado_final = reconstruct_path(busqueda, start_node, end_node)
        return resultado_final
    elif algorithm == 'busqueda_profundidad':
        busqueda = busqueda_profundidad(G, start_node, end_node)
        resultado_final = reconstruct_path(busqueda, start_node, end_node)
        return resultado_final
    elif algorithm == 'busqueda_limitada':
        busqueda = busqueda_limitada_profundidad(G, start_node, end_node)
        resultado_final = reconstruct_path(busqueda, start_node, end_node)
        return resultado_final
    elif algorithm == 'busqueda_uniforme':
        busqueda = busqueda_costo_uniforme(G, start_node, end_node)
        resultado_final = reconstruct_path(busqueda, start_node, end_node)
        return resultado_final

def style_unvisited_edge(G, edge):
    # Estilo para un borde no visitado
    G.edges[edge]["color"] = "#d36206"  # Color del borde
    G.edges[edge]["alpha"] = 0.2  # Transparencia del borde
    G.edges[edge]["linewidth"] = 0.5  # Grosor del borde


def style_visited_edge(G, edge):
    # Estilo para un borde visitado
    G.edges[edge]["color"] = "#d36206"  # Color del borde
    G.edges[edge]["alpha"] = 1  # Transparencia del borde
    G.edges[edge]["linewidth"] = 1  # Grosor del borde


def style_active_edge(G, edge):
    # Estilo para un borde activo (en proceso de visita)
    G.edges[edge]["color"] = '#e8a900'  # Color del borde
    G.edges[edge]["alpha"] = 1  # Transparencia del borde
    G.edges[edge]["linewidth"] = 1  # Grosor del borde


def style_path_edge(G, edge):
    # Estilo para un borde que forma parte del camino encontrado
    G.edges[edge]["color"] = "white"  # Color del borde
    G.edges[edge]["alpha"] = 1  # Transparencia del borde
    G.edges[edge]["linewidth"] = 1  # Grosor del borde


def plot_graph(G):
    # Función para trazar el grafo
    ox.plot_graph(
        G,  # Grafo
        # Tamaño de los nodos del grafo, se basa en el atributo "size" de los nodos
        node_size=[G.nodes[node]["size"] for node in G.nodes],
        # Color de los bordes del grafo, se basa en el atributo "color" de los bordes
        edge_color=[G.edges[edge]["color"] for edge in G.edges],
        # Transparencia de los bordes del grafo, se basa en el atributo "alpha" de los bordes
        edge_alpha=[G.edges[edge]["alpha"] for edge in G.edges],
        # Grosor de los bordes del grafo, se basa en el atributo "linewidth" de los bordes
        edge_linewidth=[G.edges[edge]["linewidth"] for edge in G.edges],
        # Color de los nodos del grafo
        node_color="white",
        # Color de fondo del gráfico
        bgcolor="#18080e"
    )


# --------------------------------------------------------------------------------
def quitar_frente(lista_espera):
    # Quitar el primer elemento de la cola
    return lista_espera.popleft()


# Función para realizar la búsqueda en amplitud (BFS) en el grafo
def busqueda_amplitud(G, orig, dest, plot=False):
    # Inicialización de los nodos del grafo
    for node in G.nodes:
        G.nodes[node]["visited"] = False  # Marcando todos los nodos como no visitados
        G.nodes[node]["previous"] = None  # Estableciendo el nodo anterior como None
        G.nodes[node]["size"] = 0  # Estableciendo el tamaño del nodo como 0

    # Inicialización de los bordes del grafo
    # for edge in G.edges:
    #    style_unvisited_edge(G, edge)  # Aplicando el estilo para bordes no visitados

    # Estableciendo el tamaño del nodo de origen y destino como 50
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    cola = deque([orig])  # Creando una cola y agregando el nodo de origen
    step = 0  # Inicializando el contador de pasos en 0

    while cola:  # Mientras la cola no esté vacía
        node = quitar_frente(cola)  # Sacando el primer nodo de la cola

        if node == dest:  # Si el nodo actual es el nodo de destino
            if plot:
                print("Iteraciones:", step)
                plot_graph(G)  # Visualizando el grafo si se establece plot como True

            return G  # Terminando la función

        if G.nodes[node]["visited"]:  # Si el nodo ya ha sido visitado, continuamos con el siguiente nodo en la cola
            continue

        G.nodes[node]["visited"] = True  # Marcando el nodo como visitado

        for edge in G.out_edges(node):  # Iterando sobre los bordes salientes del nodo
            # style_visited_edge(G, (edge[0], edge[1], 0))  # Aplicando el estilo para un borde visitado
            neighbor = edge[1]  # Obteniendo el vecino al otro lado del borde
            if not G.nodes[neighbor]["visited"]:  # Si el vecino no ha sido visitado
                #  style_active_edge(G, (edge[0], edge[1], 0))  # Aplicando el estilo para un borde activo
                G.nodes[neighbor]["previous"] = node  # Estableciendo el nodo actual como el nodo anterior del vecino
                cola.append(neighbor)  # Agregando el vecino a la cola para su posterior exploración

        step += 1  # Incrementando el contador de pasos


# --------------------------------------------------------------------------------
# Función para realizar la búsqueda en profundidad
def busqueda_profundidad(G, orig, dest, plot=False):
    # Inicialización de los nodos del grafo
    for node in G.nodes:
        G.nodes[node]["visited"] = False  # nodos como no visitados
        G.nodes[node]["previous"] = None  # nodo anterior como None
        G.nodes[node]["size"] = 0  # tamaño del nodo como 0

    # Inicialización de los bordes del grafo
    #for edge in G.edges:
    #    style_unvisited_edge(edge)  # Aplicando el estilo para bordes no visitados

    # Estableciendo el tamaño del nodo de origen y destino como 50
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    pila = [orig]  # Creando una pila y agregando el nodo de origen
    step = 0  # Inicializando el contador de pasos en 0

    while pila:  # Mientras la pila no esté vacía
        node = pila.pop()  # Sacando el último nodo de la pila

        if node == dest:  # Si el nodo actual es el nodo de destino
            if plot:
                print("Iteraciones:", step)
                # plot_graph()  # Visualizando el grafo si se establece plot como True
            return G  # Terminando la función

        if G.nodes[node]["visited"]:  # Si el nodo ya ha sido visitado, continuamos con el siguiente nodo en la pila
            continue

        G.nodes[node]["visited"] = True  # Marcando el nodo como visitado

        for edge in G.out_edges(node):  # Iterando sobre los bordes salientes del nodo
           # style_visited_edge((edge[0], edge[1], 0))  # Aplicando el estilo para un borde visitado
            neighbor = edge[1]  # Obteniendo el vecino al otro lado del borde

            if not G.nodes[neighbor]["visited"]:  # Si el vecino no ha sido visitado
             #   style_active_edge((edge[0], edge[1], 0))  # Aplicando el estilo para un borde activo
                G.nodes[neighbor]["previous"] = node  # Estableciendo el nodo actual como el nodo anterior del vecino
                pila.append(neighbor)  # Agregando el vecino a la pila para su posterior exploración

        step += 1  # Incrementando el contador de pasos


# --------------------------------------------------------------------------------

# Función para realizar la búsqueda en profundidad limitada en el grafo
def busqueda_limitada_profundidad(G, orig, dest, limite, plot=False):
    # Inicialización de los nodos del grafo
    for node in G.nodes:
        G.nodes[node]["visited"] = False  # Marcando todos los nodos como no visitados
        G.nodes[node]["previous"] = None  # Estableciendo el nodo anterior como None
        G.nodes[node]["size"] = 0  # Estableciendo el tamaño del nodo como 0

    # Inicialización de los bordes del grafo
    #for edge in G.edges:
    #    style_unvisited_edge(edge)  # Aplicando el estilo para bordes no visitados

    # Estableciendo el tamaño del nodo de origen y destino como 50
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50

    pila = [(orig, 0)]  # Creando una pila y guardando la profundidad actual
    step = 0  # Inicializando el contador de pasos en 0

    while pila:  # Mientras la pila no esté vacía
        node, depth = pila.pop()  # Sacando el nodo y su profundidad actual de la pila

        if node == dest:  # Si el nodo actual es el nodo de destino
            if plot:
                print("Iteraciones:", step)
                #plot_graph()  # Visualizando el grafo si se establece plot como True
            return G  # Terminando la función

        if depth >= limite:  # Si la profundidad actual es mayor o igual al límite establecido
            continue  # Pasando al siguiente nodo en la pila

        if G.nodes[node]["visited"]:  # Si el nodo ya ha sido visitado, continuamos con el siguiente nodo en la pila
            continue

        G.nodes[node]["visited"] = True  # Marcando el nodo como visitado

        for edge in G.out_edges(node):  # Iterando sobre los bordes salientes del nodo
            #style_visited_edge((edge[0], edge[1], 0))  # Aplicando el estilo para un borde visitado
            neighbor = edge[1]  # Obteniendo el vecino al otro lado del borde

            if not G.nodes[neighbor]["visited"]:  # Si el vecino no ha sido visitado
                #style_active_edge((edge[0], edge[1], 0))  # Aplicando el estilo para un borde activo
                G.nodes[neighbor]["previous"] = node  # Estableciendo el nodo actual como el nodo anterior del vecino
                pila.append((neighbor, depth + 1))  # Agregando el vecino a la pila con una profundidad incrementada

        step += 1  # Incrementando el contador de pasos


# --------------------------------------------------------------------------------

# Función para realizar la búsqueda de costo uniforme
def busqueda_costo_uniforme(G, orig, dest, plot=False):
    # Inicialización de los nodos del grafo
    for node in G.nodes:
        G.nodes[node]["visited"] = False  # Marcando todos los nodos como no visitados
        G.nodes[node]["distance"] = float("inf")  # Estableciendo la distancia a cada nodo como infinito
        G.nodes[node]["previous"] = None  # Estableciendo el nodo anterior como None
        G.nodes[node]["size"] = 0  # Estableciendo el tamaño del nodo como 0

    # Inicialización de los bordes del grafo
    #for edge in G.edges:
    #    style_unvisited_edge(edge)  # Aplicando el estilo para bordes no visitados

    G.nodes[orig]["distance"] = 0  # Estableciendo la distancia al nodo de origen como 0
    G.nodes[orig]["size"] = 50  # Estableciendo el tamaño del nodo de origen como 50
    G.nodes[dest]["size"] = 50  # Estableciendo el tamaño del nodo de destino como 50

    pq = [(0, orig)]  # Creando una cola de prioridad con el nodo de origen y su distancia como prioridad
    step = 0  # Inicializando el contador de pasos en 0

    while pq:  # Mientras la cola de prioridad no esté vacía
        _, node = heapq.heappop(pq)  # Sacando el nodo con la menor distancia de la cola

        if node == dest:  # Si el nodo actual es el nodo de destino
            if plot:
                print("Iteraciones:", step)  # Imprimiendo el número de iteraciones si se establece plot como True
               # plot_graph()  # Visualizando el grafo si se establece plot como True
            return G # Terminando la función

        if G.nodes[node]["visited"]:  # Si el nodo ya ha sido visitado, continuamos con el siguiente nodo en la cola
            continue

        G.nodes[node]["visited"] = True  # Marcando el nodo como visitado

        for edge in G.out_edges(node):  # Iterando sobre los bordes salientes del nodo
            # style_visited_edge((edge[0], edge[1], 0))  # Aplicando el estilo para un borde visitado
            neighbor = edge[1]  # Obteniendo el vecino al otro lado del borde
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]  # Obteniendo el peso del borde

            if G.nodes[neighbor]["distance"] > G.nodes[node][
                "distance"] + weight:  # Si la distancia al vecino es mayor que la suma de la distancia al nodo actual y el peso del borde
                G.nodes[neighbor]["distance"] = G.nodes[node][
                                                    "distance"] + weight  # Actualizando la distancia al vecino
                G.nodes[neighbor]["previous"] = node  # Estableciendo el nodo actual como el nodo anterior del vecino
                heapq.heappush(pq, (G.nodes[neighbor]["distance"],
                                    neighbor))  # Agregando el vecino a la cola de prioridad con su nueva distancia como prioridad
                #for edge2 in G.out_edges(neighbor):  # Iterando sobre los bordes salientes del vecino
                    #style_active_edge((edge2[0], edge2[1], 0))  # Aplicando el estilo para un borde activo

        step += 1  # Incrementando el contador de pasos


# --------------------------------------------------------------------------------

def busqueda_bidireccional(orig, dest, plot=False):
    # Inicialización de las estructuras de datos para ambas direcciones
    for node in G.nodes:
        G.nodes[node][
            "visited_forward"] = False  # Marcando todos los nodos como no visitados en la búsqueda hacia adelante
        G.nodes[node][
            "visited_backward"] = False  # Marcando todos los nodos como no visitados en la búsqueda hacia atrás
        G.nodes[node][
            "previous_forward"] = None  # Estableciendo el nodo anterior como None en la búsqueda hacia adelante
        G.nodes[node]["previous_backward"] = None  # Estableciendo el nodo anterior como None en la búsqueda hacia atrás
        G.nodes[node]["distance_forward"] = float(
            "inf")  # Estableciendo la distancia a cada nodo como infinito en la búsqueda hacia adelante
        G.nodes[node]["distance_backward"] = float(
            "inf")  # Estableciendo la distancia a cada nodo como infinito en la búsqueda hacia atrás
        G.nodes[node]["size"] = 0  # Estableciendo el tamaño del nodo como 0

    for edge in G.edges:
        style_unvisited_edge(edge)  # Aplicando el estilo para bordes no visitados

    # Inicialización de los nodos de inicio y fin
    G.nodes[orig][
        "distance_forward"] = 0  # Estableciendo la distancia al nodo de origen como 0 en la búsqueda hacia adelante
    G.nodes[dest][
        "distance_backward"] = 0  # Estableciendo la distancia al nodo de destino como 0 en la búsqueda hacia atrás
    G.nodes[orig]["size"] = 50  # Estableciendo el tamaño del nodo de origen como 50
    G.nodes[dest]["size"] = 50  # Estableciendo el tamaño del nodo de destino como 50

    # Colas de prioridad para las búsquedas hacia adelante y hacia atrás
    pq_forward = [(0,
                   orig)]  # Cola de prioridad para la búsqueda hacia adelante con el nodo de origen y su distancia como prioridad
    pq_backward = [(0,
                    dest)]  # Cola de prioridad para la búsqueda hacia atrás con el nodo de destino y su distancia como prioridad

    step = 0  # Inicializando el contador de pasos en 0

    while pq_forward or pq_backward:  # Mientras alguna de las colas de prioridad no esté vacía
        # Búsqueda hacia adelante
        if pq_forward:
            _, node_forward = heapq.heappop(
                pq_forward)  # Sacando el nodo con la menor distancia de la cola hacia adelante

            if G.nodes[node_forward][
                "visited_backward"]:  # Si el nodo hacia adelante ya ha sido visitado por la búsqueda hacia atrás
                if plot:
                    print("Iteraciones:", step)  # Imprimiendo el número de iteraciones si se establece plot como True
                    plot_graph()  # Visualizando el grafo si se establece plot como True
                return  # Terminando la función

            if G.nodes[node_forward][
                "visited_forward"]:  # Si el nodo hacia adelante ya ha sido visitado por la búsqueda hacia adelante, continuamos con el siguiente nodo en la cola
                continue

            G.nodes[node_forward][
                "visited_forward"] = True  # Marcando el nodo como visitado en la búsqueda hacia adelante

            for edge in G.out_edges(node_forward):  # Iterando sobre los bordes salientes del nodo hacia adelante
                style_visited_edge((edge[0], edge[1], 0))  # Aplicando el estilo para un borde visitado
                neighbor = edge[1]  # Obteniendo el vecino al otro lado del borde
                weight = G.edges[(edge[0], edge[1], 0)]["weight"]  # Obteniendo el peso del borde

                if G.nodes[neighbor]["distance_forward"] > G.nodes[node_forward][
                    "distance_forward"] + weight:  # Si la distancia al vecino desde el nodo de origen es mayor que la suma de la distancia al nodo actual y el peso del borde
                    G.nodes[neighbor]["distance_forward"] = G.nodes[node_forward][
                                                                "distance_forward"] + weight  # Actualizando la distancia al vecino desde el nodo de origen
                    G.nodes[neighbor][
                        "previous_forward"] = node_forward  # Estableciendo el nodo actual como el nodo anterior del vecino en la búsqueda hacia adelante
                    heapq.heappush(pq_forward, (G.nodes[neighbor]["distance_forward"],
                                                neighbor))  # Agregando el vecino a la cola de prioridad hacia adelante con su nueva distancia como prioridad
                    for edge2 in G.out_edges(neighbor):  # Iterando sobre los bordes salientes del vecino
                        style_active_edge((edge2[0], edge2[1], 0))  # Aplicando el estilo para un borde activo

        # Búsqueda hacia atrás
        if pq_backward:
            _, node_backward = heapq.heappop(
                pq_backward)  # Sacando el nodo con la menor distancia de la cola hacia atrás

            if G.nodes[node_backward][
                "visited_forward"]:  # Si el nodo hacia atrás ya ha sido visitado por la búsqueda hacia adelante
                if plot:
                    print("Iteraciones:", step)  # Imprimiendo el número de iteraciones si se establece plot como True
                    plot_graph()  # Visualizando el grafo si se establece plot como True
                return  # Terminando la función

            if G.nodes[node_backward][
                "visited_backward"]:  # Si el nodo hacia atrás ya ha sido visitado por la búsqueda hacia atrás, continuamos con el siguiente nodo en la cola
                continue

            G.nodes[node_backward][
                "visited_backward"] = True  # Marcando el nodo como visitado en la búsqueda hacia atrás

            for edge in G.in_edges(node_backward):  # Iterando sobre los bordes entrantes del nodo hacia atrás
                style_visited_edge((edge[0], edge[1], 0))  # Aplicando el estilo para un borde visitado
                neighbor = edge[0]  # Obteniendo el vecino al otro lado del borde
                weight = G.edges[(edge[0], edge[1], 0)]["weight"]  # Obteniendo el peso del borde

                if G.nodes[neighbor]["distance_backward"] > G.nodes[node_backward][
                    "distance_backward"] + weight:  # Si la distancia al vecino desde el nodo de destino es mayor que la suma de la distancia al nodo actual y el peso del borde
                    G.nodes[neighbor]["distance_backward"] = G.nodes[node_backward][
                                                                 "distance_backward"] + weight  # Actualizando la distancia al vecino desde el nodo de destino
                    G.nodes[neighbor][
                        "previous_backward"] = node_backward  # Estableciendo el nodo actual como el nodo anterior del vecino en la búsqueda hacia atrás
                    heapq.heappush(pq_backward, (G.nodes[neighbor]["distance_backward"],
                                                 neighbor))  # Agregando el vecino a la cola de prioridad hacia atrás con su nueva distancia como prioridad
                    for edge2 in G.in_edges(neighbor):  # Iterando sobre los bordes entrantes del vecino
                        style_active_edge((edge2[0], edge2[1], 0))  # Aplicando el estilo para un borde activo

        step += 1  # Incrementando el contador de pasos


# --------------------------------------------------------------------------------

def reconstruct_path(G, orig, dest, plot=False, algorithm=None):
    #for edge in G.edges:
    #    style_unvisited_edge(edge)  # Restaurando el estilo de todos los bordes a no visitados
    path_coords = []

    dist = 0  # Inicializando la distancia total del recorrido
    speeds = []  # Lista para almacenar las velocidades de los bordes
    curr = dest  # Inicializando el nodo actual como el nodo de destino

    while curr != orig:  # Mientras no lleguemos al nodo de origen
        prev = G.nodes[curr]["previous"]  # Obteniendo el nodo anterior al nodo actual
        dist += G.edges[(prev, curr, 0)]["length"]  # Sumando la longitud del borde al total de la distancia
        speeds.append(
            G.edges[(prev, curr, 0)]["maxspeed"])  # Añadiendo la velocidad del borde a la lista de velocidades
        #style_path_edge((prev, curr, 0))  # Aplicando el estilo de camino al borde

        try:
            path_coords.append((G.nodes[curr]['x'], G.nodes[curr]['y']))
        except KeyError:
            print(f'Key error: {KeyError}')
            path_coords = []

        if algorithm:  # Si se especifica un algoritmo
            # Incrementando el contador de uso del algoritmo para el borde actual
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1

        curr = prev  # Moviendo el nodo actual al nodo anterior

    dist /= 1000  # Convirtiendo la distancia total a kilómetros (11752786689, 6336532308, 0)

    return path_coords


# --------------------------------------------------------------------------------
def obtener_coordenadas(G):
    # Convertir el grafo a GeoDataFrames
    nodos, bordes = ox.graph_to_gdfs(G)

    # Extraer las coordenadas de los nodos
    coordenadas = nodos[['y', 'x']]  # 'y' son las latitudes, 'x' son las longitudes

    # Convertir las coordenadas a una lista de tuplas
    lista_coordenadas = [(lat, lon) for lat, lon in zip(coordenadas['y'], coordenadas['x'])]

    return lista_coordenadas
