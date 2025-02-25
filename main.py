import argparse
import math
import random
from math import inf
from time import time

from plotting import plot_points, draw_path, circle_point, title, show_plot, plot_weights
from network_routing import find_shortest_path_with_array, find_shortest_path_with_heap


def rand1to1():
    return (random.random() - 0.5) * 2  # -1 to 1


def dist(p1, p2, noise):
    if noise == -1:
        return random.random()

    raw_dist = math.dist(p1, p2)

    return max(0.0, raw_dist + random.normalvariate(mu=0, sigma=noise))


def generate_graph(seed, size, density, noise) -> tuple[
    list[tuple[float, float]],   # The positions
    dict[int, dict[int, float]]  # The graph
]:
    random.seed(seed)

    positions = [
        (rand1to1(), rand1to1())
        for _ in range(size)
    ]

    edges_per_node = int(round((size - 1) * density))

    weights = {}
    for source in range(size):
        weights[source] = {}
        for target in random.sample(range(size), edges_per_node):
            weights[source][target] = dist(positions[source], positions[target], noise)

    return positions, weights


def main(seed: int, size: int, density: float, noise: float, source: int, target: int):
    start = time()
    positions, weights = generate_graph(seed, size, density, noise)
    end = time()

    num_edges = sum(len(edges) for edges in weights.values())
    print(f'Time to generate network of {size} nodes and {num_edges} edges: {round(end - start, 4)}')

    print(f'Direct cost from {source} to {target}: {weights[source].get(target, math.inf)}')

#    plot_points(positions)
#    if num_edges < 50:
#        # If the number of non-inf edges is < 50
#        plot_weights(positions, weights)

    circle_point(positions[source], c='r')
    circle_point(positions[target], c='b')

    # start = time()
    # path, cost = find_shortest_path_with_heap(weights, source, target)
    # end = time()
    # heap_time = end - start
    # print()
    # print('-- Heap --')
    # print('Path:', path)
    # print('Cost:', cost)
    # print('Time:', heap_time)

    # draw_path(positions, path)

    start = time()
    # path, cost = find_shortest_path_with_array(weights, source, target)

    test_weights_graph = {
        0: {4: 0.6368876671279146, 5: 1.1897652785076027, 3: 0.4374171348162189, 0: 0.0, 1: 1.600589011692469, 7: 0.7244746277575653, 8: 0.982352350964381}, 
        1: {3: 1.4348439157107238, 7: 2.2880594269524646, 1: 0.0, 6: 1.527710732133722, 8: 1.1714087659454089, 4: 2.1602968957083597, 2: 1.8792800673720298}, 
        2: {3: 0.7543319323750056, 7: 1.1451080859807352, 4: 0.9938336928276987, 1: 1.8792800673720298, 5: 1.2792080530640813, 6: 1.8602213667924192, 0: 1.1281617186905575}, 
        3: {6: 1.1094450078879563, 4: 0.7483139369043482, 7: 0.8945083091439281, 8: 1.1865355932193704, 9: 1.1262140067174637, 3: 0.0, 1: 1.4348439157107238}, 
        4: {7: 0.16103862217505918, 0: 0.6368876671279146, 9: 1.4454987880291832, 3: 0.7483139369043482, 5: 1.6608577494026757, 2: 0.9938336928276987, 6: 1.3379806698258363}, 
        5: {5: 0.0, 2: 1.2792080530640813, 7: 1.8037881077682616, 3: 0.9129627118538306, 8: 1.1802298228007837, 1: 0.6004303988348423, 9: 1.3007401254761848}, 
        6: {8: 0.4153590611559747, 6: 0.0, 1: 1.527710732133722, 9: 0.17494510519454698, 7: 1.366216241423186, 4: 1.3379806698258363, 3: 1.1094450078879563}, 
        7: {1: 2.2880594269524646, 8: 1.668462279768978, 6: 1.366216241423186, 4: 0.16103862217505918, 7: 0.0, 5: 1.8037881077682616, 9: 1.488432391484682}, 
        8: {1: 1.1714087659454089, 7: 1.668462279768978, 9: 0.2404390205542289, 8: 0.0, 2: 1.9284980653982196, 4: 1.6089810912976688, 0: 0.982352350964381}, 
        9: {4: 1.4454987880291832, 2: 1.8803415840785553, 7: 1.488432391484682, 0: 0.8448086431169819, 5: 1.3007401254761848, 9: 0.0, 8: 0.2404390205542289}
    }

    test_weights_graph = {
        0: {4: 6, 5: 11, 3: 4, 0: 0, 1: 16, 7: 7, 8: 9}, 
        1: {3: 14, 7: 22, 1: 0.0, 6: 15, 8: 11, 4: 21, 2: 18}, 
        2: {3: 7, 7: 11, 4: 9, 1: 18, 5: 12, 6: 18, 0: 11}, 
        3: {6: 11, 4: 7, 7: 8, 8: 11, 9: 11, 3: 0.0, 1: 14}, 
        4: {7: 1, 0: 6, 9: 14, 3: 7, 5: 16, 2: 9, 6: 13}, 
        5: {5: 0.0, 2: 12, 7: 18, 3: 9, 8: 11, 1: 6, 9: 13}, 
        6: {8: 4, 6: 0.0, 1: 15, 9: 1, 7: 13, 4: 13, 3: 11}, 
        7: {1: 22, 8: 16, 6: 13, 4: 1, 7: 0.0, 5: 18, 9: 14}, 
        8: {1: 11, 7: 16, 9: 2, 8: 0.0, 2: 19, 4: 16, 0: 9}, 
        9: {4: 14, 2: 18, 7: 14, 0: 8, 5: 13, 9: 0.0, 8: 2}
    }

    book_positions = [
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1)
    ]

    book_graph = {
        0: {0: 0, 1: 4, 2: 2},
        1: {1: 0, 2: 3, 3: 2, 4: 3},
        2: {1: 1, 2: 0, 3: 4, 4: 5},
        3: {3: 0},
        4: {3: 1, 4: 0}
    }

    book_source = 0
    book_target = 3

    # positions, test_graph = generate_graph(312, 1000, 0.2, 0.05)

    print(f"Positions: {positions}")
    plot_weights(book_positions, book_graph)

    path, cost = find_shortest_path_with_array(book_graph, book_source, book_target)
    # path, cost = find_shortest_path_with_array(test_graph, 2, 9)
    
    
    end = time()
    array_time = end - start
    print()
    print('-- Array --')
    print('Path:', path)
    print('Cost:', cost)
    print('Time:', array_time)

    # draw_path(positions, path)  ## This is temp while the heap is not implemented
    heap_time = 0 ## This is temp while the heap is not implemented

    title(f'Cost: {cost}, Heap: {round(heap_time, 4)}, Array: {round(array_time, 4)}')
    show_plot()


if __name__ == '__main__':
    # To debug or run in your IDE
    # you can uncomment the lines below and modify the arguments as needed
    # import sys
    # sys.argv = ['main.py', '-n', '100000', '--seed', '312', '--density', '0.0001', '--noise', '0.05']

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, help='The number of points to generate', default=10)
    parser.add_argument('--seed', type=int, default=312, help='Random seed')
    parser.add_argument('--density', type=float, default=0.8, help='Fraction of non-inf edges')
    parser.add_argument('--noise', type=float, default=0, help='How non-euclidean are the edge weights')
    parser.add_argument('--source', type=int, default=0, help='Starting node')
    parser.add_argument('--target', type=int, default=None, help='Target node')
    parser.add_argument('--debug', action='store_true', help='Turn on debug plotting')
    args = parser.parse_args()

    if args.debug:
        # To debug your algorithm with incremental plotting:
        # - run this script with --debug (e.g. add '--debug' to the sys.argv above)
        # - set breakpoints
        # As you step through your code, you will see the plot update as you go
        import matplotlib.pyplot as plt

        plt.switch_backend('QtAgg')
        plt.ion()

    if args.target is None:
        args.target = args.n - 1

    main(args.seed, args.n, args.density, args.noise, args.source, args.target)

    # You can use a loop like the following to generate data for your tables:
    # for n in [100, 200, 400, 800, 1600, 3200, 6400]:
    #     main(312, n, 1, 0.05, 2, 9)
