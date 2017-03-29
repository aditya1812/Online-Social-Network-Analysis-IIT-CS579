import networkx as nx
import matplotlib.pyplot as plt
import pickle

def create_graph():
    graph = nx.read_edgelist('Friends.txt',delimiter='\t')
    nx.draw(graph)
    #plt.show()
    plt.savefig('cluster.png')
    return graph

def girvan_newman(G, most_valuable_edge=None):

    if G.number_of_edges() == 0:
        yield tuple(nx.connected_components(G))
        return
    # If no function is provided for computing the most valuable edge,
    # use the edge betweenness centrality.
    if most_valuable_edge is None:
        def most_valuable_edge(G):

            betweenness = nx.edge_betweenness_centrality(G)
            return max(betweenness, key=betweenness.get)
    # The copy of G here must include the edge weight data.
    g = G.copy().to_undirected()
    # Self-loops must be removed because their removal has no effect on
    # the connected components of the graph.
    g.remove_edges_from(g.selfloop_edges())
    while g.number_of_edges() > 0:
        yield _without_most_central_edges(g, most_valuable_edge)



def _without_most_central_edges(G, most_valuable_edge):

    original_num_components = nx.number_connected_components(G)
    num_new_components = original_num_components
    while num_new_components <= original_num_components:
        edge = most_valuable_edge(G)
        G.remove_edge(*edge)
        new_components = tuple(nx.connected_components(G))
        num_new_components = len(new_components)
    return new_components

def check_graph(graph):
    cluster_count = 0
    graph_components = nx.connected_component_subgraphs(graph)
    for i in graph_components:
        graph_tuple = tuple(sorted(c) for c in next(girvan_newman(i)))
        cluster_count =cluster_count + len(graph_tuple)
    print(cluster_count)
    return cluster_count

def main():
    graph = create_graph()
    cluster_count = check_graph(graph)
    print(nx.number_of_nodes(graph))

    count = []
    count.append(cluster_count)
    count.append(nx.number_of_nodes(graph))
    data = open('Cluster_Statistics.txt', 'wb')
    pickle.dump(count, data)
    data.close()


if __name__ == '__main__':
    main()