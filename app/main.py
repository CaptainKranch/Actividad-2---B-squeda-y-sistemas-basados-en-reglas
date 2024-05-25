import heapq
import gradio as gr


class Graph:
    def __init__(self):
        self.edges = {}  # Diccionario de todas las rutas
        self.weights = {}  # Diccionario con los pesos de cada ruta

    def add_edge(self, from_node, to_node, weight):
        # Añadir la ruta bidireccional
        if from_node not in self.edges:
            self.edges[from_node] = []
        self.edges[from_node].append(to_node)

        if to_node not in self.edges:
            self.edges[to_node] = []
        self.edges[to_node].append(from_node)

        # Añadir el peso de la ruta
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight


def dijkstra(graph, start, end):
    queue = []
    heapq.heappush(queue, (0, start))
    distances = {node: float("infinity") for node in graph.edges}
    distances[start] = 0
    previous_nodes = {node: None for node in graph.edges}

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor in graph.edges[current_node]:
            weight = graph.weights[(current_node, neighbor)]
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))

    path, current_node = [], end
    while previous_nodes[current_node] is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    path.append(start)
    path.reverse()

    return path


# Crear el grafo del sistema de transporte
graph = Graph()
# Añadir más estaciones y rutas con intersecciones
graph.add_edge("A", "B", 1)
graph.add_edge("A", "C", 4)
graph.add_edge("B", "C", 2)
graph.add_edge("B", "D", 5)
graph.add_edge("C", "D", 1)
graph.add_edge("C", "E", 3)
graph.add_edge("D", "E", 2)
graph.add_edge("D", "F", 3)
graph.add_edge("E", "F", 1)
graph.add_edge("E", "G", 2)
graph.add_edge("F", "G", 2)
graph.add_edge("F", "H", 3)
graph.add_edge("G", "H", 1)
graph.add_edge("G", "I", 3)
graph.add_edge("H", "I", 2)
graph.add_edge("H", "J", 4)
graph.add_edge("I", "J", 1)
graph.add_edge("I", "K", 2)
graph.add_edge("J", "K", 3)
graph.add_edge("J", "L", 2)
graph.add_edge("K", "L", 1)
graph.add_edge("K", "M", 4)
graph.add_edge("L", "M", 2)


def find_route(start, end):
    path = dijkstra(graph, start, end)
    return " -> ".join(path)


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            start_input = gr.Textbox(label="Punto de inicio")
            end_input = gr.Textbox(label="Punto de destino")
            submit_button = gr.Button("Encontrar ruta")
        with gr.Column():
            output = gr.Textbox(label="Ruta más corta")

    submit_button.click(fn=find_route, inputs=[start_input, end_input], outputs=output)

demo.launch()
