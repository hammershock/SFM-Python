import networkx as nx

if __name__ == "__main__":
    graph = nx.DiGraph()

    graph.add_node(1, a=1, b=2)
    graph.add_node(2, a=2, b=6)
    graph.add_node(3, a=3, b=68)
    graph.add_edge(1, 3, data='data')
    # print(graph.nodes(data=True))
    print(type(graph.nodes(data=True)))
    print(graph[1][3])
