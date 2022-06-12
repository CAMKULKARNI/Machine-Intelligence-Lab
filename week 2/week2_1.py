"""
You can create any other helper funtions.
Do not modify the given functions
"""
import sys as s

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
    path =[]
    f=[]
    f.append(start_point)
    while(len(f)!=0):
        curr_node=f.pop()
        path.append(curr_node)
        if(curr_node in goals):
            return path
        else:
            m=s.maxsize
            ind=-1
            for i in range(len(cost)-1,0,-1):
                if(cost[curr_node][i]!=-1 and cost[curr_node][i]!=0 and (i not in path)):
                    if(m>cost[curr_node][i]+heuristic[i]):
                        m=cost[curr_node][i]+heuristic[i]
                        ind=i
            f.append(ind)
    path=[]
    return path


def DFS_Traversal(cost, start_point, goals):
    """
    Perform DFS Traversal and find the optimal path 
        cost: cost matrix (list of floats/int)
        start_point: Staring node (int)
        goals: Goal states (list of ints)
    Returns:
        path: path to goal state obtained from DFS(list of ints)
    """
    path=[]
    f=[]
    visit=[]
    x=1
    z=[]
    f.append(start_point)
    while(len(f)!=0):     
        curr_node=f.pop()
        x-=1
        if(len(z)>=2 and z[len(z)-1]<z[len(z)-2]):
            path.pop()
        path.append(curr_node)
        visit.append(curr_node)
        if(curr_node in  goals):
            
            return path
        else:
            for i in range(len(cost)-1,0,-1):
                if ((i in visit) or cost[curr_node][i]<=0):
                    pass
                else:
                    f.append(i)
                    x+=1
            z.append(x)
    path=[]
    return path
