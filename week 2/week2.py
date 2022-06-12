"""
You can create any other helper funtions.
Do not modify the given functions
"""


def A_star_Traversal(cost, heuristic, start_point, goals):
    """
    Perform A* Traversal and find the optimal path 
    Args:
        cost: cost matrix (list of floats/int)
        heuristic: heuristics for A* (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from A*(list of ints)
    """
    path = []
    frontier = [start_point]

    while len(frontier):
        node = frontier.pop()
        path.append(node)
        
        if(node in goals):
            return path
        else:
            s = float("inf")
            i = -1
            m = len(cost) - 1
            while m > 0:
                if(cost[node][m] > 0 and (m not in path)):
                    c = cost[node][m] + heuristic[m]
                    if(s > c):
                        s = c
                        i = m
                m -= 1
            frontier.append(i)

    return []


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path = []
    frontier = [start_point]
    explored = set()
    p, z = 1, []

    while len(frontier):
        node = frontier.pop()
        p -= 1
        n = len(z)
        if n > 1 and z[n - 2] > z[n - 1]:
            path.pop()
        path.append(node)
        explored.add(node)

        if node in goals:
            return path
        else:
            m = len(cost) - 1
            while m > 0:
                if not(m in explored or cost[node][m] < 1):
                    frontier.append(m)
                    p += 1
                m -= 1
            z.append(p)

    return []
