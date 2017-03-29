from collections import Counter, defaultdict, deque
from datetime import datetime
import copy
import math
import networkx as nx
import urllib.request


## Community Detection

def example_graph():
    """
    Create the example graph from class. Used for testing.
    Do not modify.
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.
    You may use these two classes to help with this implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque
    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.
    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    In the doctests below, we first try with max_depth=5, then max_depth=2.
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 5)
    >>> sorted(node2distances.items())
    [('A', 3), ('B', 2), ('C', 3), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('A', 1), ('B', 1), ('C', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('A', ['B']), ('B', ['D']), ('C', ['B']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    >>> node2distances, node2num_paths, node2parents = bfs(example_graph(), 'E', 2)
    >>> sorted(node2distances.items())
    [('B', 2), ('D', 1), ('E', 0), ('F', 1), ('G', 2)]
    >>> sorted(node2num_paths.items())
    [('B', 1), ('D', 1), ('E', 1), ('F', 1), ('G', 2)]
    >>> sorted((node, sorted(parents)) for node, parents in node2parents.items())
    [('B', ['D']), ('D', ['E']), ('F', ['E']), ('G', ['D', 'F'])]
    """
    ###TODO
    pass
 
    traversed = {}
    for i in graph:
        traversed[i] = False
    queue = [root]
    q = [root]
    distance = {}
    num_paths = defaultdict(int)
    num_paths[root] = 1
    parents = defaultdict(list)
   
    
    distance[root] = 0
    traversed[root] = True
    
    while(queue != []):
        
        curr_v = queue.pop(0)
        dis = distance[curr_v]
        paths = num_paths[curr_v]
        
        for i in graph.neighbors(curr_v):
            if (dis < max_depth):    
            
                if(traversed[i] == False):
                    
                    traversed[i] = True
                    distance[i] = distance[curr_v] + 1
                if(distance[i] == dis + 1 and paths != 0):
                    num_paths[i] = num_paths[i] + paths
                    parents[i].append(curr_v)
                    
                    queue.append(i)
                        
    return distance, num_paths, parents




def main():
    start_time = datetime.now()
    bfs(example_graph(), 'E', 5)
    end_time = datetime.now()
    print ("Time...............", end_time,start_time)
    
    
if __name__ == '__main__':
    main()
