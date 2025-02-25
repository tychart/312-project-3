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

    return dijkstra_algorithm(graph, source, target, pq_class=HeapPriorityQueue)

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

    # print(f"Started find_shortest_path_with_array")

    return dijkstra_algorithm(graph, source, target, pq_class=ArrayPriorityQueue)


    # # print(f"Graph: {graph}")
    # # print(f"Source: {source}")
    # # print(f"Target: {target}")

    # visited = [source]
    # # print(f"Visited: {visited}")

    # distances = {node: float('inf') for node in graph.keys()}
    # distances[source] = 0
    # # print(f"Distances: {distances}")

    # previous = {node: None for node in graph.keys()}
    # # print(f"Previous: {previous}")

    # curr_node = source


    # pq = pq_class()
    # pq.insert(source, 0)
    # visited = set()




    # while not pq.is_empty():
    #     current_dist, curr_node = pq.extract_min()
        
    #     if curr_node in visited:
    #         continue
    #     visited.add(curr_node)
        
    #     if curr_node == target:
    #         break
            
    #     for neighbor, weight in graph[curr_node].items():
    #         if neighbor in visited:
    #             continue
    #         new_dist = current_dist + weight
    #         if new_dist < distances[neighbor]:
    #             distances[neighbor] = new_dist
    #             previous[neighbor] = curr_node
    #             pq.insert(neighbor, new_dist)












    # while len(visited) < len(graph):
        
    #     # print(f"Now processing node: {curr_node}")


    #     for neighbor in graph[curr_node]:
    #         if distances[curr_node] + graph[curr_node][neighbor] < distances[neighbor]:
    #             distances[neighbor] = distances[curr_node] + graph[curr_node][neighbor]
    #             previous[neighbor] = curr_node
        
    #     # print(f"Distances: {distances}")
    #     # print(f"Previous: {previous}")


    #     # Find the node with the smallest distance
    #     smallest_dist = float('inf')
    #     for node in visited:
    #         for neighbor in graph[node]:
    #             if neighbor not in visited and graph[node][neighbor] < smallest_dist:
    #                 smallest_dist = graph[node][neighbor]
    #                 curr_node = neighbor
        
    #     visited.append(curr_node)
    #     # print(f"Visited: {visited}")

    # # plotting.plot_points(graph)

    
    # path = trace_previous_list(previous, source, target)

    # return path, distances[target]

    # # return [[0, 2, 5, 9], 1.0]  # Dummy return value


class Node:
    def __init__(self, name, distance):
        self.name = name
        self.distance = distance


class BasePriorityQueue:
    def insert(self, node: int, distance: float):
        raise NotImplementedError
        
    def extract_min(self) -> tuple[int, float]:
        raise NotImplementedError
        
    def is_empty(self) -> bool:
        raise NotImplementedError

class ArrayPriorityQueue(BasePriorityQueue):
    def __init__(self):
        self.nodes = []
        
    def insert(self, node, distance):
        self.nodes.append((node, distance))
        
    def extract_min(self):
        minIndex = 0
        # minWeight = self.nodes[minIndex][1]

        for i in range(0, len(self.nodes)):
            if self.nodes[i][1] < self.nodes[minIndex][1]:
                minIndex = i
        retNode = self.nodes.pop(minIndex)
        return retNode[0], retNode[1]
        # return min(self.nodes, key=lambda x: x[0])
    
    def is_empty(self) -> bool:
        return not self.nodes

class HeapPriorityQueue(BasePriorityQueue):
    def __init__(self):
        self.heap = []
        
    def insert(self, name, distance):
        
        self.heap.append(Node(name, distance))

    ## The heap is a binary tree, so we can represent it as an array
    ## The root is at index 0
    ## The left child of a node at index i is at index 2i + 1
    ## The right child of a node at index i is at index 2i + 2
    ## The parent of a node at index i is at index (i - 1) // 2

        curr_index = len(self.heap) - 1  # Start from the last inserted node

        while curr_index > 0:  # Ensure we're not at the root
            parent_index = (curr_index - 1) // 2
            parent = self.heap[parent_index]

            if parent.distance > self.heap[curr_index].distance:
                self.heap[parent_index], self.heap[curr_index] = self.heap[curr_index], self.heap[parent_index]
                curr_index = parent_index
            else:
                break  # If the heap property is satisfied, stop the loop


        # Bubble up
        # Check if the parent is smaller than the child (node just added)
        # If not, swap them
        # while True:
        #     parent = (node - 1) // 2
        #     if parent < 0 or self.heap[parent][1] <= distance:
        #         break
        #     self.heap[parent], self.heap[node] = self.heap[node], self.heap[parent]
        #     node = parent

        # heapq.heappush(self.heap, (distance, node))
        
    def extract_min(self):
        
        retNode, retDist = self.heap[0]
        

        
        # Bubble down
        # Compare the 2 children to find the minimum
        # Compare the minimum child to the parent
        # If the parent is smaller, swap them
        # Repeat until the parent is smaller than both children
        # while True:
        #     left = 2 * node + 1
        #     right = 2 * node + 2
        #     if left >= len(self.heap):
        #         break
        #     if right >= len(self.heap) or self.heap[left][1] < self.heap[right][1]:
        #         child = left
        #     else:
        #         child = right
        #     if self.heap[node][1] <= self.heap[child][1]:
        #         break
        #     self.heap[node], self.heap[child] = self.heap[child], self.heap[node]
        #     node = child
        #self.heap.pop(0)
        #return retNodeDist[0], retNodeDist[1]
        # return heapq.heappop(self.heap)
        
        return heapq.heappop(self.heap)








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

def dijkstra_algorithm(
        graph: dict[int, dict[int, float]],
        source: int,
        target: int,
        pq_class: type[BasePriorityQueue] = ArrayPriorityQueue
) -> tuple[list[int], float]:
    
    print(f"Started dijkstra_algorithm using pq_class: {pq_class}")

    distances = {node: float('inf') for node in graph}
    distances[source] = 0
    previous = {node: None for node in graph}
    
    pq = pq_class()
    pq.insert(source, 0)
    visited = set()

    while not pq.is_empty():
        curr_node, current_dist = pq.extract_min()
        
        print(f"Current distance: {current_dist} Current node: {curr_node}")

        if curr_node in visited:
            continue
        visited.add(curr_node)
        
        if curr_node == target:
            break
            
        for neighbor, weight in graph[curr_node].items():
            print(f"Neighbor: {neighbor} Weight: {weight}")
            if neighbor in visited:
                continue
            new_dist = current_dist + weight
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = curr_node
                pq.insert(neighbor, new_dist)
    
    return trace_previous_list(previous, source, target), distances[target]


