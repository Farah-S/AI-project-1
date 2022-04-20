# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import searchAgents
import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem): #problem is type PositionSearchProblem i think, it can be found in searchAgents.py
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print ("Start:", problem.getStartState())
    print ("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    sucs=problem.getSuccessors(problem.getStartState()) #list of tuples where each tuple is (x,y pos tuple,direction,cost)
    #((34,2),'North',1)
    print ("Start's successors:", problem.getSuccessors(problem.getStartState()))
    print ("Start's successors[0][0][0]:", sucs[0][0][0]) #getting the x
    print ("Start's successors[0][1]:", sucs[0][1])#getting the direction
    """
    
    #REMEMBER For each algorithm ; calculate the searching time , Explored node and path on the figure
    
    "*** YOUR CODE HERE ***"
    #import util
    st = [] #stack
    visited = [] 
    sol=[] #solution steps
    st.append((problem.getStartState(),'',0)) #push starting "node"
    visited.append(problem.getStartState())
    while st: #while stack is not empty
        current = st[-1] #grab last element/top of stack (not pop)
        
        if(problem.isGoalState(current[0])): #check if we reached the goal
            return sol #return solution steps
        
        if current[0] not in visited: #check if the node has been visited before      
            visited.append(current[0]) 
        
        allNeighbours = problem.getSuccessors(current[0]) #get all "successors" aka neighbours of this node
        
        for neighbour in allNeighbours: #cycle through all the neighbours
            if neighbour[0] not in visited: #check if the neighbour has been visited
                st.append(neighbour) #if not visited add it to stack
                sol.append(neighbour[1]) #add it's movement direction to the sol
                break  
        else: 
            st.pop() #if ALL neighbours were visited(and none were the goal) backtrack by popping the top of stack
            sol.pop() #remove the last movement to backtrack
    return sol
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #sucs=problem.getSuccessors(problem.getStartState())
    #print(sucs)
    #q = util.Queue()
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def GreedyBestFirstSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    pqueue=[] #priority queue
    visited = [] 
    sol=[] #solution steps
    
    pqueue.append([0,(problem.getStartState(),'',0),[]]) #[heuristic,((pos),direction,cost),[path]]
    visited.append(problem.getStartState())

    while pqueue:  #while queue is not empty
        current = pqueue.pop(0) #grab first element
        sol=current[2] #save current path as solution

        if current[1][0] not in visited: #check if the node has been visited before      
            visited.append(current[1][0])

        if(problem.isGoalState(current[1][0])): #check if we reached the goal
            #sol=current[2]
            return sol #return solution steps

        allNeighbours = problem.getSuccessors(current[1][0]) #get all "successors"/neighbours of this node
        
        for neighbour in allNeighbours: #cycle through all the neighbours
            if neighbour[0] not in visited: #check if the neighbour has been visited
                heuristic=searchAgents.euclideanHeuristic(neighbour[0],problem) #calculate the heuristic for this neighbour
                path=[]
                path=current[2].copy() #copy the parent node path
                path.append(neighbour[1]) #add this node to the path
                pqueue.append([heuristic,neighbour,path]) #if not visited add the neighbour to pqueue
                
        pqueue.sort(key = lambda x: x[0]) #sort the priority queue based on heuristic
        topNeighbour=pqueue[0] #take the top of the queue (least heuristic)
        visited.append(topNeighbour[0]) #add the neighbour to the visited list
    
    return sol
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
gbfs = GreedyBestFirstSearch