import React, { useState, useRef, useEffect } from "react";

const singleAgentDictionary = {
  "queue": "A First-In-First-Out (FIFO) data structure used in BFS.",
  "queues": "First-In-First-Out (FIFO) data structures used in BFS.",
  "stack": "A Last-In-First-Out (LIFO) data structure used in DFS.",
  "stacks": "Last-In-First-Out (LIFO) data structures used in DFS.",
  "priority queue": "A queue where elements with higher priority are served before elements with lower priority.",
  "priority queues": "Queues where elements with higher priority are served before elements with lower priority.",
  "enqueue": "Adding an element to the end of a queue.",
  "enqueues": "Adds an element to the end of a queue.",
  "dequeue": "Removing an element from the front of a queue.",
  "dequeues": "Removes an element from the front of a queue.",
  "push": "Adding an element to the top of a stack.",
  "pushes": "Adds an element to the top of a stack.",
  "pop": "Removing an element from the top of a stack.",
  "pops": "Removes an element from the top of a stack.",
  "breadth first": "A traversal strategy that explores all nodes at the current distance before moving further.",
  "breadth-first": "A traversal strategy that explores all nodes at the current distance before moving further.",
  "depth first": "A traversal strategy that explores as far as possible along a branch before backtracking.",
  "depth-first": "A traversal strategy that explores as far as possible along a branch before backtracking.",
  "greedy": "An algorithm that makes locally optimal choices at each step.",
  "backtracking": "Abandoning a path when it's determined not to lead to a solution.",
  "level by level": "BFS's characteristic exploration pattern expanding outward in concentric 'rings'.",
  "directed": "A movement pattern that tends to favor certain directions over others.",
  "manhattan distance": "The sum of absolute differences between coordinates, used as a heuristic.",
  "euclidean distance": "The straight-line distance between two points, used as a heuristic.",
  "recursive": "An approach where a function calls itself to solve smaller instances of the same problem.",
  "dynamic programming": "Breaking complex problems into simpler subproblems and storing their solutions.",
  "admissible heuristic": "A heuristic that never overestimates the actual cost to reach the goal.",
  "admissible heuristics": "Heuristics that never overestimate the actual cost to reach the goal."
};

const multiAgentDictionary = {
  "agent": "An entity that can perceive its environment and take actions.",
  "agents": "Entities that can perceive their environment and take actions.",
  "conflict": "A situation where two or more agents try to occupy the same space at the same time.",
  "conflicts": "Situations where two or more agents try to occupy the same space at the same time.",
  "constraint": "A restriction on an agent's movement to prevent conflicts.",
  "constraints": "Restrictions on agents' movements to prevent conflicts.",
  "conflict-based search": "A multi-agent pathfinding algorithm that resolves conflicts through constraints.",
  "increasing cost tree": "A search tree used in ICTS where each level represents a specific cost for each agent.",
  "constraint tree": "A tree structure used in CBS where each node represents a set of constraints.",
  "subproblem": "A smaller part of the original problem that can be solved independently.",
  "subproblems": "Smaller parts of the original problem that can be solved independently."
};

const sharedDictionary = {
  "graph": "A data structure consisting of nodes (vertices) connected by edges.",
  "graphs": "Data structures consisting of nodes (vertices) connected by edges.",
  "node": "A point or vertex in a graph, representing a location.",
  "nodes": "Points or vertices in a graph, representing locations.",
  "edge": "A connection between two nodes in a graph.",
  "edges": "Connections between nodes in a graph.",
  "vertex": "A point or node in a graph, representing a location.",
  "vertices": "Points or locations in a graph (singular: vertex).",
  "neighbor": "A node directly connected to the current node by an edge.",
  "neighbors": "Nodes directly connected to the current node by an edge.",
  "adjacent": "Nodes that are directly connected by an edge.",
  "traversal": "The process of visiting all nodes in a graph in a specific order.",
  "frontier": "The collection of nodes that have been discovered but not yet explored.",
  "visited": "Nodes that have already been processed during traversal.",
  "path": "A sequence of nodes connected by edges.",
  "paths": "Sequences of nodes connected by edges.",
  "cycle": "A path that starts and ends at the same node.",
  "cycles": "Paths that start and end at the same node.",
  "heuristic": "An educated guess or estimate that helps guide search algorithms toward the goal.",
  "heuristics": "Educated guesses or estimates that help guide search algorithms toward the goal.",
  "optimal": "Guaranteeing the shortest or lowest-cost path to the goal.",
  "complete": "Guaranteeing to find a solution if one exists.",
  "weighted": "A graph where edges have different costs or distances associated with them.",
  "weighted graph": "A graph where edges have different costs or distances associated with them.",
  "weighted graphs": "Graphs where edges have different costs or distances associated with them.",
  "unweighted": "A graph where all edges have the same cost or distance.",
  "unweighted graph": "A graph where all edges have the same cost or distance.",
  "unweighted graphs": "Graphs where all edges have the same cost or distance."
};

const isMultiAgentAlgorithm = (algorithm) => {
  return ['cbs', 'icts', 'mstar', 'pushandrotate'].includes(algorithm);
};

const getTooltipDictionary = (algorithm) => {
  if (!algorithm) return {};
  
  if (isMultiAgentAlgorithm(algorithm)) {
    return { ...sharedDictionary, ...multiAgentDictionary };
  } else {
    return { ...sharedDictionary, ...singleAgentDictionary };
  }
};

const TermTooltip = ({ term, selectedAlgorithm, children }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({
    verticalPlacement: 'top',
    horizontalPlacement: 'center',
    left: '50%',
    top: 'auto',
    bottom: '125%',
    transform: 'translateX(-50%)'
  });
  const tooltipRef = useRef(null);
  const termRef = useRef(null);
  
  const tooltipDictionary = getTooltipDictionary(selectedAlgorithm);
  const definition = tooltipDictionary[term.toLowerCase()];
  
  useEffect(() => {
    if (isVisible && termRef.current && tooltipRef.current && definition) {
      const termRect = termRef.current.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      const tooltipWidth = tooltipRect.width || 220;
      const tooltipHeight = tooltipRect.height || 60;
      
      let container = termRef.current.closest('.overflow-y-auto') || document.body;
      const containerRect = container.getBoundingClientRect();
      
      const spaceAbove = termRect.top - containerRect.top;
      const spaceBelow = containerRect.bottom - termRect.bottom;
      const spaceLeft = termRect.left - containerRect.left;
      const spaceRight = containerRect.right - termRect.right;
      
      let verticalPlacement, top, bottom;
      if (spaceAbove >= tooltipHeight || spaceAbove > spaceBelow) {
        verticalPlacement = 'top';
        top = 'auto';
        bottom = '125%';
      } else {
        verticalPlacement = 'bottom';
        top = '125%';
        bottom = 'auto';
      }
      
      let horizontalPlacement, left, transform;
      if (spaceLeft < tooltipWidth / 2) {
        horizontalPlacement = 'left';
        left = '0';
        transform = 'translateX(0)';
      } else if (spaceRight < tooltipWidth / 2) {
        horizontalPlacement = 'right';
        left = '100%';
        transform = 'translateX(-100%)';
      } else {
        horizontalPlacement = 'center';
        left = '50%';
        transform = 'translateX(-50%)';
      }
      
      setTooltipPosition({
        verticalPlacement,
        horizontalPlacement,
        top,
        bottom,
        left,
        transform
      });
    }
  }, [isVisible, definition]);
  
  if (!definition) {
    return <>{children || term}</>;
  }
  
  const tooltipStyle = {
    position: 'absolute',
    top: tooltipPosition.top,
    bottom: tooltipPosition.bottom,
    left: tooltipPosition.left,
    transform: tooltipPosition.transform,
    backgroundColor: 'rgba(51, 51, 51, 0.95)',
    color: 'white',
    textAlign: 'center',
    padding: '8px 12px',
    borderRadius: '6px',
    width: '220px',
    fontSize: '10px',
    lineHeight: '1.4',
    zIndex: 100,
    opacity: isVisible ? 1 : 0,
    visibility: isVisible ? 'visible' : 'hidden',
    transition: 'opacity 0.3s, visibility 0.3s',
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
    pointerEvents: 'none'
  };
  
  const getArrowStyle = () => {
    const vertical = tooltipPosition.verticalPlacement;
    const horizontal = tooltipPosition.horizontalPlacement;
    
    const style = {
      position: 'absolute',
      borderWidth: '6px',
      borderStyle: 'solid',
    };
    
    if (vertical === 'top') {
      style.top = '100%';
      style.borderColor = 'rgba(51, 51, 51, 0.95) transparent transparent transparent';
    } else {
      style.bottom = '100%';
      style.borderColor = 'transparent transparent rgba(51, 51, 51, 0.95) transparent';
    }
    
    if (horizontal === 'left') {
      style.left = '10%';
    } else if (horizontal === 'right') {
      style.right = '10%';
    } else {
      style.left = '50%';
      style.transform = 'translateX(-50%)';
    }
    
    return style;
  };
  
  return (
    <span 
      ref={termRef}
      style={{ 
        position: 'relative',
        borderBottom: '1px dotted #666',
        cursor: 'help',
        display: 'inline-block'
      }}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children || term}
      <span
        ref={tooltipRef}
        style={tooltipStyle}
      >
        {definition}
        <span style={getArrowStyle()} />
      </span>
    </span>
  );
};

const withTooltips = (text, selectedAlgorithm) => {
  const spanRegex = /\*\*<span style='color: ([^']+)'>([^<]+)<\/span>\*\*/g;
  
  const paragraphs = text.split('\n\n');
  
  return paragraphs.map((paragraph, paragraphIndex) => {
    const parts = [];
    let lastIndex = 0;
    let match;

    while ((match = spanRegex.exec(paragraph)) !== null) {
      if (match.index > lastIndex) {
        parts.push(paragraph.substring(lastIndex, match.index));
      }
      
      parts.push(
        <span key={`span-${paragraphIndex}-${match.index}`} style={{color: match[1], fontWeight: 'bold'}}>
          {match[2]}
        </span>
      );
      
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < paragraph.length) {
      parts.push(paragraph.substring(lastIndex));
    }

    const processedParts = parts.map((part, index) => {
      if (typeof part !== 'string') return part;

      const tooltipDictionary = getTooltipDictionary(selectedAlgorithm);
      
      const termPattern = Object.keys(tooltipDictionary)
        .map(term => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('|');
      
      if (!termPattern) return part;  
      
      const termRegex = new RegExp(`\\b(${termPattern})\\b`, 'gi');
      
      const tooltipParts = [];
      let lastTermIndex = 0;
      let termMatch;

      while ((termMatch = termRegex.exec(part)) !== null) {
        if (termMatch.index > lastTermIndex) {
          tooltipParts.push(part.substring(lastTermIndex, termMatch.index));
        }
        
        tooltipParts.push(
          <TermTooltip 
            key={`tooltip-${index}-${termMatch.index}`} 
            term={termMatch[0].toLowerCase()}
            selectedAlgorithm={selectedAlgorithm}
          >
            {termMatch[0]}
          </TermTooltip>
        );
        
        lastTermIndex = termMatch.index + termMatch[0].length;
      }

      if (lastTermIndex < part.length) {
        tooltipParts.push(part.substring(lastTermIndex));
      }

      return tooltipParts;
    });

    return (
      <p key={paragraphIndex} className="whitespace-pre-line text-left align-left">
        {processedParts.flat()}
      </p>
    );
  });
};

const algorithmDescriptions = {
  bfs: "This explores a graph level by level, visiting all vertices at the current distance before moving outward, like exploring a city by visiting all locations closest to your starting point before venturing further.  It uses a queue to track vertices, ensuring the shortest path in unweighted graphs. \n\n\n At each step, Breadth First Search dequeues a vertex, visits all its unvisited neighbours, enqueues them, and marks them visited. Time complexity is O(V + E) and space complexity is O(V). It is ideal for finding shortest paths or connected components in unweighted graphs. Use Dijkstra's algorithm for weighted graphs and Depth-First Search for exploring deep branches or paths.",
  
  dfs: "This aims to traverse as far along as possible along each branch in a graph before backtracking, similar to retracing a route to the goal when hitting a dead end in a maze.\n\n\n This begins by pushing the starting vertex onto the stack. It then repeatedly pops a vertex from the stack, visits any unvisited neighbours, pushes them onto the stack, and marks them as visited. When all neighbours of the current vertex have been explored, DFS backtracks by popping vertices off the stack, exploring alternative paths. \n\n\n The time complexity is O(V + E) and space complexity of O(V) due to the stack. This is ideal for exploring all possible paths in a graph, but does not guarantee the shortest path.",
  
  dijkstra: "This finds the shortest path in weighted graphs by maintaining a priority queue of vertices sorted by their current shortest distance from the starting node. Imagine a hiker who always chooses the path with the least accumulated effort, considering the terrain difficulty of each trail segment rather than distance alone. \n\n\n\ At each step, it selects the unvisited vertex with the smallest distance, marks it as visited, and updates its neighbours' distances if a shorter path is found through the current vertex. \n\n\n The algorithm has a  O((V + E) log V) time complexity with a priority queue implementation. It is optimal for weighted graphs, but unnecessary for unweighted graphs where Breadth First Search suffices.",
  
  astar: "This is an informed search algorithm that combines Dijkstra's shortest-path approach with heuristic estimations of remaining distance to the goal. Car GPS systems often combine distance travelled with estimated remaining distance based on speed limits and road layout. This guides drivers along the most efficient route, adjusting recommendations based on traffic. \n\n\n It maintains a priority queue of nodes sorted by f(n) = g(n) + h(n), where g(n) is the actual distance from start and h(n) estimates distance to goal. This helps focus exploration towards promising paths. A* guarantees the shortest path when using an admissible heuristic, and is widely used in pathfinding and navigation due to better performance than Dijkstra's.",
  
  randomwalk: "This explores a graph by randomly selecting directions at each step. Imagine a person who does not have the ability to memorise, who needs to move around making spontaneous turns at each intersection towards their destination.  \n\n\n\The algorithm chooses one of four possible directions (up, down, left, right) with equal probability at each step. It can revisit positions multiple times, leading to meandering paths. It is simple but inefficient for pathfinding, and more commonly used in simulations.",

  biasedrandomwalk: "This improves upon the standard random walk by weighting movement choices towards the direction of the goal. Imagine a traveller without a map, but they have a compass pointing towards their destination, so they mostly move in the general direction of their goal, but they occasionally explore interesting side paths. \n\n\n At each step, nodes closer to the goal receive higher selection probabilities. The algorithm also considers visit history to encourage exploration of unvisited nodes. This creates a balance between random exploration and directed movement, typically finding paths more effectively than pure random walk.",

  jps: "This is an optimisation of A* for uniform-cost grid maps that skip over straight lune segments to focus on 'jump points' and make decisions. Imagine a lift that skips floors where passengers do not need to get on or off, reducing travel speeds. \n\n\n This identifies and only expands nodes that may change the optimal path, reducing the number of nodes examined. It preserves the optimality of A* whilst improving performance, and is best in large open areas with many straight paths.",
  
  gbfs: "This a heuristic search algorithm that always expands the node cloest to the goal, based on a heuristic function. Imagine a chess player who always makes a move that captures the most valuable piece, without considering strategic effects in the long term. \n\n\n At each step, the algorithm expands the node with the lowest heuristic value, which is aslo the one appearing the closest to the goal. This continues until the goal is reached or no more nodes remain. This is faster than A* but less accurate as the path cost so far is ignored, only considering estimated distance to the goal. It cannot guarantee the shortest path as it can get stuck in local minima, so can be used when a good enough solution is acceptable.",

  pushandrotate: "This is a rule-based algorithm that moves agents toward their goals one at a time. Imagine a crowded train station where one person moves toward their platform while others temporarily step aside, with everyone eventually reaching their destinations through a sequence of coordinated movements. \n\n\n The algorithm uses basic operations like 'push' (moving agents out of the way) and 'rotate' (swapping positions in cycles) to systematically resolve conflicts. This guarantees solution completeness on graphs with at least two empty vertices, though not optimality. Its polynomial time complexity makes it practical for large agent numbers where optimal solutions aren't required.",

  cbs: "This algorithm plans paths for all agents while resolving conflicts through constraints. Imagine an orchestra conductor who ensures each musician plays their part without conflicting with each other, and with constraints imposed on specific musicians when they need to resolve dissonance.\n\n\n It uses a two level approach: a high-level search in the constraint tree and low-level searches to find paths satisfying the constraints. The algorithm branches and creates new constraints when conflicts occur between agents. This guarantees optimal solutions for the sum of path costs, though computational complexity increases with the number of conflicts that need resolution.",

  icts: "This algorithm searches the space of possible path costs rather than paths themselves. Imagine a gardener planning a flower arrangement by considering various combinations of flower prices to create a design that is attractive and affordable, rather than focusing on flower species. \n\n\n This constructs an increasing cost tree where each node represents a combination of path costs for all agents. For each node, it checks if compatible paths with those costs exist. This approach guarantees optimal solutions but can have exponential complexity in the number of agents, and performs well when paths have few interactions.",

  mstar: "This algorithm combines individual policies with collaborative planning. Imagine many autonomous vehicles navigating a city, where each vehicle follows its own optimal path, but when they encounter potential collisions they must communicate and collaboratively plan to avoid conflicts while maintaining efficiency. \n\n\n It starts with optimal single-agent policies and only considers joint actions at states where following individual policies would lead to collisions. This focused exploration dramatically reduces the search space, and the algorithm dynamically expands the joint state space as needed, offering better scalability than naïve approaches while still guaranteeing optimal solutions."
};

const runningDescriptions = {
  bfs: "Breadth First Search is exploring the grid level by level:\n• Visited nodes: each node is only visited once, and the ripple pattern shows how all nodes are explored at one level before moving outward\n• Current node: the node being processed in this step\n• Unvisited nodes: nodes that have not been reached yet\n• Walls: obstacles that block the path and cannot be traversed",
  
  dfs: "Depth First Search is exploring the grid in depth:\n• Visited nodes: nodes may get progressively darker as they are visited multiple times when backtracking from dead ends\n• Considered nodes: nodes that were evaluated but not visited\n• Current node: the node being processed, with a prioritization on exploring down and right directions before up and left\n• Walls: obstacles that force backtracking to find alternative paths",
  
  dijkstra: "Dijkstra's algorithm is exploring based on path costs:\n• Parent connections: forms a tree of shortest paths from the start to each visited node\n• Current node: the unvisited node with the lowest total cost from start\n• Low-cost terrain: easier to traverse\n• High-cost terrain: more difficult to traverse\n• Walls: obstacles that block the path and cannot be traversed",
  
  astar: "A* is exploring using path costs and estimated goal distance:\n• Parent connections: forms a tree tracking the best path found to each visited node\n• Current node: selected based on lowest f(n) = g(n) + h(n) score\n• Low-cost terrain: easier to traverse\n• High-cost terrain: more difficult to traverse\n• Walls: obstacles that block the path and cannot be traversed",
  
  randomwalk: "Random Walk is exploring using pure chance:\n• Visited positions: darker colors indicate more frequent visits as positions are revisited\n• Current position: the node being processed in this step\n• Unvisited nodes: nodes that have not been reached yet\n• Walls: obstacles that block the path and cannot be traversed",
  
  biasedrandomwalk: "Biased Random Walk is exploring using weighted chance:\n• Visited positions: darker colors indicate more frequent visits as positions are revisited\n• Current position: the node being processed in this step\n• Unvisited nodes: nodes that have not been reached yet\n• Walls: obstacles that block the path and cannot be traversed",
  
  gbfs: "Greedy Best First Search is exploring the grid using heuristics only:\n• Visited nodes: nodes that have been processed based purely on their estimated distance to the goal\n• Current node: the node with the lowest estimated distance to the goal\n• Walls: obstacles that block the path and cannot be traversed",
  
  jps: "Jump Point Search is exploring the grid with optimized pathfinding:\n• Visited nodes: nodes that have been processed, including special jump points where the search changes direction or encounters obstacles\n• Current node: the node being processed in this step\n• Walls: obstacles that block the path and cannot be traversed",
  
  cbs: "Conflict-Based Search is coordinating multiple agents:\n• Visited nodes: nodes that have been explored by each agent\n• Considered nodes: nodes that were evaluated but not visited yet\n• Walls: obstacles that block the path and cannot be traversed",
  
  icts: "Increasing Cost Tree Search is finding optimal multi-agent paths:\n• Visited nodes: nodes that have been explored by each agent\n• Considered nodes: nodes that were evaluated but not visited yet\n• Walls: obstacles that block the path and cannot be traversed",
  
  mstar: "M* is exploring a joint configuration space for multiple agents:\n• Visited nodes: nodes that have been explored by each agent\n• Considered nodes: nodes that were evaluated but not visited yet\n• Walls: obstacles that block the path and cannot be traversed",
  
  pushandrotate: "Push and Rotate is solving multi-agent pathfinding with local operations:\n• Travelling nodes: nodes that agents are moving through during operations\n• Walls: obstacles that block the path and cannot be traversed"
};

const completionMessages = {
  bfs: "**<span style='color: darkgreen'>Breadth First Search has found a path:</span>**\n• Yellow cells reveal a ripple-like expansion pattern, where each 'ripple' represents nodes at the same distance from start, showing how all nodes are explored at one distance before moving outward\n• Blue cells reveal the discovered route from start to end, and this is guaranteed to be the shortest possible route due to exploration through increasing distances from the source",
  
  dfs: "**<span style='color: darkgreen'>Depth First Search has found a path:</span>**\n• Yellow cells show the algorithm's deep exploration pattern, with lighter shades for initial visits and darker where it had to backtrack\n• Dark yellow cells typically appear near walls, showing where the algorithm hit dead ends and needed to retreat\n• Green cells show neighbors that were considered but never explored as the algorithm found its path elsewhere\n• Blue cells show a valid path from start to end, but may not be the shortest path",
  
  dijkstra: "**<span style='color: darkgreen'>Dijkstra's algorithm has found a path:</span>**\n• Coral lines show the expansion of Dijkstra's search, indicating which nodes were considered to reach others\n• These lines represent how the algorithm tracks the best known path to each node during exploration\n• The light blue path highlights the optimal route from the start node to the goal node\n• This path minimises the total accumulated weight, showing how Dijkstra's algorithm guarantees finding the lowest-cost path by prioritizing nodes with the smallest accumulated cost",  
  
  astar: "**<span style='color: darkgreen'>A* has found a path:</span>**\n• Pink lines show paths A* considered promising, and they tend to point toward the goal\n• The blue path highlights the discovered optimal route, balancing actual path costs with estimated remaining distance.",
  
  randomwalk: "**<span style='color: darkgreen'>Random Walk has found a path:</span>**\n• Yellow cells show visitied cells\n• Darker shades indicate areas where the algorithm repeatedly revisited nodes, as there is no bias or memory of visited nodes recollected\n• There is no guarantee of finding the optimal path.",
  
  biasedrandomwalk: "**<span style='color: darkgreen'>Biased Random Walk has found a path:</span>**\n• Yellow cells show the visited cells, where randomness was balanced alongside directed movement\n• Visited cells tend to cluster toward the goal rather than spreading evenly due to directional bias\n• Varying shades of yellow show where the algorithm revisited positions\n• The path found may not be optimal but is usually more direct than pure random walk.",
  
  gbfs: "**<span style='color: darkgreen'>Greedy Best First Search has found a path:</span>**\n• Yellow cells show visited nodes explored based purely on their estimated distance to the goal, prioritising nodes appearing closest to the goal\n• Blue cells show the discovered path from the start node to the goal node\n• There may be detours when heuristic estimates are misleading, as it does not consider the cost of the path traveled so far and thus does not find an optimal path like A*",
  
  jps: "**<span style='color: darkgreen'>Jump Point Search has found a path:</span>**\n• Yellow cells show visited nodes, with a lack of straight lines due to the algorithm skipping redundant nodes, where exploration is only at points where the optimal path may change\n• Blue cells show the optimal path from the start node to the goal node, exploring fewer nodes than A* whilst still finding the optimal path path",
  
  cbs: "**<span style='color: darkgreen'>Conflict-Based Search has found a path:</span>**\n• Visited nodes are highlighted in a darker shade, and considered nodes are highlighted in a lighter shade\n• The agents are able to reach their goals without collisions at start and goal nodes, and when travelling through the grid",
  
  icts: "**<span style='color: darkgreen'>Increasing Cost Tree Search has found a path:</span>**\n• Visited nodes are highlighted in a darker shade, and considered nodes are highlighted in a lighter shade\n• The sum of all agent path costs is minimised, with exploration focusing on finding compatible paths with increasing costs.",
  
  mstar: "**<span style='color: darkgreen'>M* has found a path:</span>**\n• Visited nodes are highlighted in a darker shade, and considered nodes are highlighted in a lighter shade\n• Individual optimal paths were followed when possible, with joint planning at conflict resolution", 
  
  pushandrotate: "**<span style='color: darkgreen'>Push and Rotate has found a path:</span>**\n• The travelling nodes have reached the goal positions at each node\n• The solution ensures all agents reach their goals without collisions, getting one agent at a time to its goal\n• Some agents may take longer paths to allow others to reach their goals first, in a sequential approach to multi-agent pathfinding."
};

const failureMessages = {
  bfs: "**<span style='color: darkred'>Breadth First Search could not find a path:</span>**\n• Yellow cells show all the reachable nodes from the start position, each visited once\n• The absence of a blue path indicates that the end position is not reachable\n• There must be a wall completely separating the start from the goal",
  
  dfs: "**<span style='color: darkred'>DFS could not find a path:</span>**\n• Yellow cells show all the reachable nodes from the start position, and darker shades of yellow show backtracking\n• The absence of a blue path indicates that the end position is not reachable\n• There could be a wall completely separating the start from the goal",
  
  dijkstra: "**<span style='color: darkred'>Dijkstra's algorithm could not find a path:</span>**\n• Coral lines show the visited nodes from the start node\n• The absence of a blue path indicates that the end position is not reachable\n• There could be a wall completely separating the start from the goal",
  
  astar: "**<span style='color: darkred'>A* could not find a path:</span>**\n•  Coral lines show the visited nodes from the start node\n• The absence of a blue path indicates that the end position is not reachable\n• There could be a wall completely separating the start from the goal",
  
  randomwalk: "**<span style='color: darkred'>Random Walk could not find a path:</span>**\n• Yellow cells show the wandering pattern of exploration, where darker yellows show cells visited more than once\n• The algorithm reached its maximum number of steps without finding the goal\n• Due to its random nature, running it again might yield different results",

  biasedrandomwalk: "**<span style='color: darkred'>Biased Random Walk could not find a path:</span>**\n• Yellow cells show a biased exploration pattern, where darker yellows show cells visited more than once\n• The algorithm reached its maximum number of steps without finding the goal\n• There might be a wall blocking access to the goal, or the goal may require a more complex path than the algorithm could find",
  
  gbfs: "**<span style='color: darkred'>Greedy Best First Search could not find a path:</span>**\n• Yellow cells show visited nodes explored based purely on their estimated distance to the goal, with each node visited once\n• The absence of a blue path indicates that the end position is not reachable\n• There could be a wall completely separating the start from the goal",
  
  jps: "**<span style='color: darkred'>Jump Point Search could not find a path:</span>**\n• Yellow cells show the visited jump points, each visited once\n• The absence of a blue path indicates that the end position is not reachable\n• There could be a wall completely separating the start from the goal",
  
  cbs: "**<span style='color: darkred'>Conflict-Based Search could not find a solution:</span>**\n• The constraints required to avoid conflicts between agents made it impossible to find valid paths for all agents\n• Try repositioning agents or simplifying the environment",
  
  icts: "**<span style='color: darkred'>Increasing Cost Tree Search could not find a solution:</span>**\n• No compatible path combination could be found with the given constraints\n• The algorithm exhausted all possible cost combinations\n• Try repositioning agents or removing some walls",
  
  mstar: "**<span style='color: darkred'>M* could not find a solution:</span>**\n• The algorithm could not find conflict-free paths for all agents\n• The joint configuration space exploration was unable to resolve all conflicts\n• Try repositioning agents or removing some walls",
  
  pushandrotate: "**<span style='color: darkred'>Push and Rotate could not find a solution:</span>**\n• The algorithm could not find a sequence of push and rotate operations to move all agents to their goals\n• Try repositioning agents or removing some walls"
};

const AlgorithmInfo = ({ selectedAlgorithm, isRunning, isFinished, walkData = null, agentMode}) => {
  if (!selectedAlgorithm) {
    return (
      <div className="bg-white border-2 border-black rounded-lg p-3 h-48 overflow-y-auto text-left">
        <p style={{ fontSize: "0.7em", marginBottom: "0.5em" }}>
          Select an algorithm to run and learn how it works. Once selected, hover over underlined terms for definitions.
        </p>
        <p className="text-xs" style={{ fontSize: "0.7em", marginBottom: "0.05em" }}>
          <i>Click and drag on the grid to place walls, or drag the start/end nodes to reposition them.</i>
        </p>
        <p className="text-xs" style={{ fontSize: "0.7em", marginBottom: "1em" }}>
          <i>Use Ctrl+Z to undo node placement and Ctrl+Y to redo.</i>
        </p>
        <p className="text-xs" style={{ fontSize: "0.7em", fontWeight: "bold", marginBottom: "0.1em" }}>
          Interface Control
        </p>
        <ul style={{ fontSize: "0.65em", paddingLeft: "15px" }}>
          <li><b>Clear</b>: Remove all mazes, walls and paths from the grid</li>
          <li><b>Randomise Nodes</b>: Rearrange start and end nodes on the grid</li>
          <li><b>Generate Maze</b>: Create a structured maze with pathways</li>
          {agentMode === "left" ? (
            <li><b>Random Maze</b>: Create a maze with random wall placements</li>
          ) : (
            <li><b>Open Grid</b>: Create a mostly empty grid with few obstacles</li>
          )}
        </ul>
      </div>
    );
  }


  const goalReached = walkData && 
                     walkData.length > 0 && 
                     walkData[walkData.length - 1]?.isGoalReached === true;

  let content;
  if (isRunning) {
    content = runningDescriptions[selectedAlgorithm];
  } else if (isFinished) {
    if (goalReached) {
      content = completionMessages[selectedAlgorithm];
    } else {
      content = failureMessages[selectedAlgorithm];
    }
  } else {
    content = algorithmDescriptions[selectedAlgorithm];
  }

  return (
    <div className="bg-white border-2 border-black rounded-lg p-3 h-48 overflow-y-auto">
      <div style={{ fontSize: "0.7em" }}>
        {withTooltips(content, selectedAlgorithm)}
      </div>
    </div>
  );
};

export default AlgorithmInfo;