import plotting

def find_shortest_path_with_heap(
        graph: dict[int, dict[int, float]],
        source: int,
        target: int
) -> tuple[list[int], float]:
    """
    Find the shortest (least-cost) path from `source` to `target` in `graph`
    using the heap-based algorithm.

    Return:
        - the list of nodes (including `source` and `target`)
        - the cost of the path
    """

    ## When inserting, if any child has a smaller value than the parent, swap them
    ## When removing, remove the root and replace it with the last element
    ## Then, if the new root is larger than any of its children
    ## Swap it with the smallest child
    ## Continue until the heap property is satisfied

    ## The heap property is that the parent is smaller than both children
    ## The root is the smallest element in the heap

    ## The heap is a binary tree, so we can represent it as an array
    ## The root is at index 0
    ## The left child of a node at index i is at index 2i + 1
    ## The right child of a node at index i is at index 2i + 2
    ## The parent of a node at index i is at index (i - 1) // 2

    heap = []

    dist = {
        'A': 'inf',
        'B': 'inf'
    }

    heap_dict = {
        1: 'A',
        2: 'B',
        3: 'C',
        4: 'D',
        5: 'E',
        6: 'F',
        7: 'G',
    }

    pointer_dict = {
        'A': '-',
        'B': 2,
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': '-',
    }


def find_shortest_path_with_array(
        graph: dict[int, dict[int, float]],
        source: int,
        target: int
) -> tuple[list[int], float]:
    """
    Find the shortest (least-cost) path from `source` to `target` in `graph`
    using the array-based (linear lookup) algorithm.

    Return:
        - the list of nodes (including `source` and `target`)
        - the cost of the path
    """

    # print(f"Graph: {graph}")
    # print(f"Source: {source}")
    # print(f"Target: {target}")

    visited = [source]
    # print(f"Visited: {visited}")

    distances = {node: float('inf') for node in graph.keys()}
    distances[source] = 0
    # print(f"Distances: {distances}")

    previous = {node: None for node in graph.keys()}
    # print(f"Previous: {previous}")

    curr_node = source

    while len(visited) < len(graph):
        
        # print(f"Now processing node: {curr_node}")

        for neighbor in graph[curr_node]:
            if distances[curr_node] + graph[curr_node][neighbor] < distances[neighbor]:
                distances[neighbor] = distances[curr_node] + graph[curr_node][neighbor]
                previous[neighbor] = curr_node
        
        # print(f"Distances: {distances}")
        # print(f"Previous: {previous}")


        # Find the node with the smallest distance
        smallest_dist = float('inf')
        for node in visited:
            for neighbor in graph[node]:
                if neighbor not in visited and graph[node][neighbor] < smallest_dist:
                    smallest_dist = graph[node][neighbor]
                    curr_node = neighbor
        
        visited.append(curr_node)
        # print(f"Visited: {visited}")

    # plotting.plot_points(graph)

    
    path = trace_previous_list(previous, source, target)

    return path, distances[target]

    # Q array
    dist = {
        'A': 'inf',
        'B': 'inf'
    }

    return [[0, 2, 5, 9], 1.0]  # Dummy return value

    # Q = 

def trace_previous_list(previous: dict[int, int], source: int, target: int) -> list[int]:
    """
    Trace the path from `source` to `target` using the `previous` dictionary.

    Return:
        - the list of nodes (including `source` and `target`)
    """
    path = [target]
    while path[-1] != source:
        path.append(previous[path[-1]])
    return list(reversed(path))