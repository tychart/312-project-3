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

    # Q array
    dist = {
        'A': 'inf',
        'B': 'inf'
    }

    Q = 
