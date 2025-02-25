from main import generate_graph
from network_routing import find_shortest_path_with_array, find_shortest_path_with_heap, BasePriorityQueue, HeapPriorityQueue

# def test_binary_heap():
#     # Create a new BinaryHeap instance
#     heap = HeapPriorityQueue()

#     # Add elements to the heap
#     heap.insert(1, 10)
#     heap.insert(2, 4)
#     heap.insert(3, 15)
#     heap.insert(4, 20)
#     heap.insert(5, 0)

#     # Print the heap after insertions
#     print("Heap after insertions:", heap)

#     # Delete the minimum element
#     min_elem = heap.delete_min()
#     print("Deleted minimum element:", min_elem)

#     # Print the heap after deletion
#     print("Heap after deleting the minimum element:", heap)

# if __name__ == "__main__":
#     test_binary_heap()




_, graph = generate_graph(312, 10, 0.3, 0.05)

print(find_shortest_path_with_array(graph, 0, 9))
print(find_shortest_path_with_heap(graph, 0, 9))
