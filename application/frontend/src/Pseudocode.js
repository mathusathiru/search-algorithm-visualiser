import React from "react";

const algorithmPseudocode = {
  bfs: `ALGORITHM BreadthFirstSearch(start, goal, graph)
    Initialise empty queue Q
    Initialise empty set visited
    Initialise empty map parent
    
    ENQUEUE start into Q
    ADD start to visited
    
    WHILE Q is not empty DO
        current ← DEQUEUE from Q
        
        IF current = goal THEN
            path ← empty list
            WHILE current ≠ start DO
                PREPEND current to path
                current ← parent[current]
            END WHILE
            PREPEND start to path
            RETURN path
        END IF
        
        FOR EACH neighbor adjacent to current in graph DO
            IF neighbor not in visited THEN
                ADD neighbor to visited
                parent[neighbor] ← current
                ENQUEUE neighbor into Q
            END IF
        END FOR
    END WHILE
    
    RETURN "No path exists"
END ALGORITHM`,

  dfs: `ALGORITHM DepthFirstSearch(start, goal, graph)
    Initialise empty stack S
    Initialise empty set visited
    Initialise empty map parent
    
    PUSH start onto S
    
    WHILE S is not empty DO
        current ← TOP of S
        
        IF current = goal THEN
            path ← empty list
            WHILE current ≠ start DO
                PREPEND current to path
                current ← parent[current]
            END WHILE
            PREPEND start to path
            RETURN path
        END IF
        
        IF current not in visited THEN
            ADD current to visited
        END IF
        
        found_unvisited ← false
        FOR EACH neighbor adjacent to current in graph DO
            IF neighbor not in visited THEN
                PUSH neighbor onto S
                parent[neighbor] ← current
                found_unvisited ← true
                BREAK
            END IF
        END FOR
        
        IF not found_unvisited THEN
            POP from S
        END IF
    END WHILE
    
    RETURN "No path exists"
END ALGORITHM`,

  dijkstra: `ALGORITHM Dijkstra(start, goal, graph)
    Initialise map distance with all vertices set to ∞
    distance[start] ← 0
    
    Initialise empty map parent
    Initialise empty set visited
    
    Initialise priority queue Q ordered by distance values
    INSERT start into Q with priority 0
    
    WHILE Q is not empty DO
        current ← EXTRACT-MIN from Q
        
        IF current = goal THEN
            path ← empty list
            WHILE current ≠ start DO
                PREPEND current to path
                current ← parent[current]
            END WHILE
            PREPEND start to path
            RETURN path
        END IF
        
        IF current in visited THEN
            CONTINUE
        END IF
        
        ADD current to visited
        
        FOR EACH neighbor adjacent to current in graph DO
            IF neighbor not in visited THEN
                alt ← distance[current] + weight(current, neighbor)
                IF alt < distance[neighbor] THEN
                    distance[neighbor] ← alt
                    parent[neighbor] ← current
                    INSERT neighbor into Q with priority alt
                END IF
            END IF
        END FOR
    END WHILE
    
    RETURN "No path exists"
END ALGORITHM`,

  astar: `ALGORITHM AStarSearch(start, goal, graph)
    Initialise empty set closed
    Initialise empty map parent
    
    Initialise map g_score with all vertices set to ∞
    g_score[start] ← 0
    
    Initialise map f_score with all vertices set to ∞
    f_score[start] ← h(start, goal)  // h is the heuristic function
    
    Initialise priority queue open ordered by f_score values
    INSERT start into open with priority f_score[start]
    
    WHILE open is not empty DO
        current ← EXTRACT-MIN from open
        
        IF current = goal THEN
            path ← empty list
            WHILE current ≠ start DO
                PREPEND current to path
                current ← parent[current]
            END WHILE
            PREPEND start to path
            RETURN path
        END IF
        
        ADD current to closed
        
        FOR EACH neighbor adjacent to current in graph DO
            IF neighbor in closed THEN
                CONTINUE
            END IF
            
            tentative_g ← g_score[current] + weight(current, neighbor)
            
            IF neighbor not in open OR tentative_g < g_score[neighbor] THEN
                parent[neighbor] ← current
                g_score[neighbor] ← tentative_g
                f_score[neighbor] ← g_score[neighbor] + h(neighbor, goal)
                
                IF neighbor not in open THEN
                    INSERT neighbor into open with priority f_score[neighbor]
                ELSE
                    UPDATE neighbor in open with priority f_score[neighbor]
                END IF
            END IF
        END FOR
    END WHILE
    
    RETURN "No path exists"
END ALGORITHM`,

  randomwalk: `ALGORITHM RandomWalk(start, goal, max_steps)
    current ← start
    Initialise empty set visited
    ADD start to visited
    
    FOR step FROM 1 TO max_steps DO
        IF current = goal THEN
            RETURN visited
        END IF
        
        neighbors ← empty list
        FOR EACH direction in [UP, RIGHT, DOWN, LEFT] DO
            next_pos ← current + direction
            IF next_pos is traversable THEN
                ADD next_pos to neighbors
            END IF
        END FOR
        
        IF neighbors is empty THEN
            RETURN visited
        END IF
        
        current ← RANDOM selection from neighbors
        ADD current to visited
    END FOR
    
    RETURN visited
END ALGORITHM`,

  biasedrandomwalk: `ALGORITHM BiasedRandomWalk(start, goal, max_steps)
    current ← start
    
    Initialise map visit_count with all positions set to 0
    visit_count[start] ← 1
    
    FOR step FROM 1 TO max_steps DO
        IF current = goal THEN
            RETURN "Goal reached"
        END IF
        
        neighbors ← empty list
        weights ← empty list
        
        FOR EACH direction in [UP, RIGHT, DOWN, LEFT] DO
            next_pos ← current + direction
            IF next_pos is traversable THEN
                ADD next_pos to neighbors
                
                // Calculate bias based on direction to goal
                dx ← goal.x - next_pos.x
                dy ← goal.y - next_pos.y
                direction_weight ← 1 - (√(dx² + dy²) / MAX_DISTANCE)
                
                // Penalise frequently visited positions
                visit_penalty ← 1 / (1 + visit_count[next_pos])
                
                weight ← direction_weight × visit_penalty
                ADD weight to weights
            END IF
        END FOR
        
        IF neighbors is empty THEN
            RETURN "No path possible"
        END IF
        
        current ← SELECT from neighbors with probability proportional to weights
        visit_count[current] ← visit_count[current] + 1
    END FOR
    
    RETURN "Max steps reached"
END ALGORITHM`,

  gbfs: `ALGORITHM GreedyBestFirstSearch(start, goal, graph)
    Initialise empty set visited
    Initialise empty map parent
    
    Initialise priority queue frontier ordered by heuristic values
    INSERT start into frontier with priority h(start, goal)
    
    WHILE frontier is not empty DO
        current ← EXTRACT-MIN from frontier
        
        IF current in visited THEN
            CONTINUE
        END IF
        
        ADD current to visited
        
        IF current = goal THEN
            path ← empty list
            WHILE current ≠ start DO
                PREPEND current to path
                current ← parent[current]
            END WHILE
            PREPEND start to path
            RETURN path
        END IF
        
        FOR EACH neighbor adjacent to current in graph DO
            IF neighbor not in visited THEN
                parent[neighbor] ← current
                INSERT neighbor into frontier with priority h(neighbor, goal)
            END IF
        END FOR
    END WHILE
    
    RETURN "No path exists"
END ALGORITHM`,

  jps: `ALGORITHM JumpPointSearch(start, goal, grid)
    Initialise empty set closed
    Initialise empty map parent
    
    Initialise map g_score with all vertices set to ∞
    g_score[start] ← 0
    
    Initialise map f_score with all vertices set to ∞
    f_score[start] ← h(start, goal)
    
    Initialise priority queue open ordered by f_score values
    INSERT start into open with priority f_score[start]
    
    WHILE open is not empty DO
        current ← EXTRACT-MIN from open
        
        IF current = goal THEN
            path ← empty list
            WHILE current ≠ start DO
                PREPEND current to path
                current ← parent[current]
            END WHILE
            PREPEND start to path
            RETURN path
        END IF
        
        ADD current to closed
        
        successors ← empty list
        
        // Find jump points in all directions
        FOR EACH direction in [cardinal and diagonal directions] DO
            jump_point ← JUMP(current, direction, goal, grid)
            IF jump_point exists THEN
                ADD jump_point to successors
            END IF
        END FOR
        
        FOR EACH successor in successors DO
            IF successor in closed THEN
                CONTINUE
            END IF
            
            tentative_g ← g_score[current] + distance(current, successor)
            
            IF successor not in open OR tentative_g < g_score[successor] THEN
                parent[successor] ← current
                g_score[successor] ← tentative_g
                f_score[successor] ← g_score[successor] + h(successor, goal)
                
                IF successor not in open THEN
                    INSERT successor into open with priority f_score[successor]
                END IF
            END IF
        END FOR
    END WHILE
    
    RETURN "No path exists"
END ALGORITHM`,

    cbs: `ALGORITHM ConflictBasedSearch(starts, goals, grid)
    root.constraints ← empty
    root.solution ← empty
    root.cost ← 0
    
    // Find initial paths for all agents
    FOR i FROM 0 TO number_of_agents - 1 DO
        path ← GetPath(i, starts[i], goals[i], empty constraints)
        IF path does not exist THEN
            RETURN "No solution exists"
        END IF
        root.solution[i] ← path
        root.cost ← root.cost + length(path)
    END FOR
    
    Initialise priority queue open_list with root
    
    WHILE open_list is not empty DO
        node ← EXTRACT-MIN from open_list
        
        // Check for conflicts
        conflicts ← FindConflicts(node.solution)
        IF conflicts is empty THEN
            RETURN node.solution // Found valid solution
        END IF
        
        conflict ← first conflict in conflicts
        
        // Branch on each agent involved in the conflict
        FOR EACH agent in conflict.agents DO
            child ← new node
            child.constraints ← copy of node.constraints
            
            // Add constraint for this agent
            ADD constraint based on conflict to child.constraints
            
            // Copy solutions from parent for all agents except this one
            FOR EACH a in all agents DO
                IF a ≠ agent THEN
                    child.solution[a] ← node.solution[a]
                END IF
            END FOR
            
            // Find new path for this agent with new constraints
            path ← GetPath(agent,starts[agent], goals[agent], child.constraints)
            IF path exists THEN
                child.solution[agent] ← path
                child.cost ← sum of all path lengths in child.solution
                INSERT child into open_list with priority child.cost
            END IF
        END FOR
    END WHILE
    
    RETURN "No solution exists"
END ALGORITHM`,

  icts: `ALGORITHM IncreasingCostTreeSearch(starts, goals, grid)
    // Find cost-optimal path for each agent individually
    optimal_costs ← empty list
    FOR i FROM 0 TO number_of_agents - 1 DO
        path ← FindShortestPath(starts[i], goals[i])
        ADD length(path) to optimal_costs
    END FOR
    
    // Initialise search queue with optimal costs
    Initialise queue Q with optimal_costs
    Initialise set visited with optimal_costs
    
    WHILE Q is not empty DO
        current_costs ← DEQUEUE from Q
        
        // Build Multi-Valued Decision Diagrams for each agent
        mdds ← empty list
        FOR i FROM 0 TO number_of_agents - 1 DO
            mdd ← BuildMDD(i, starts[i], goals[i], current_costs[i])
            ADD mdd to mdds
        END FOR
        
        // Check if joint solution exists with current costs
        joint_solution ← FindJointSolution(mdds)
        IF joint_solution exists THEN
            RETURN joint_solution
        END IF
        
        // Try increasing the cost for each agent
        FOR i FROM 0 TO number_of_agents - 1 DO
            new_costs ← copy of current_costs
            new_costs[i] ← new_costs[i] + 1
            
            IF new_costs not in visited THEN
                ADD new_costs to visited
                ENQUEUE new_costs into Q
            END IF
        END FOR
    END WHILE
    
    RETURN "No solution exists"
END ALGORITHM`,

  mstar: `ALGORITHM MStar(starts, goals, grid)
    // Find individual optimal paths for each agent
    individual_paths ← empty list
    FOR i FROM 0 TO number_of_agents - 1 DO
        path ← FindShortestPath(starts[i], goals[i])
        ADD path to individual_paths
    END FOR
    
    // Initialise search
    root.positions ← starts
    root.g_score ← 0
    root.h_score ← sum of heuristic distances from starts to goals
    root.conflict_set ← empty
    root.parent ← null
    
    Initialise priority queue open with root
    Initialise set closed as empty
    
    WHILE open is not empty DO
        current ← EXTRACT-MIN from open
        
        IF current is in closed THEN
            CONTINUE
        END IF
        
        ADD current to closed
        
        // Check if goal reached
        all_at_goal ← true
        FOR i FROM 0 TO number_of_agents - 1 DO
            IF current.positions[i] ≠ goals[i] THEN
                all_at_goal ← false
                BREAK
            END IF
        END FOR
        
        IF all_at_goal THEN
            RETURN ReconstructPaths(current)
        END IF
        
        // Generate next positions for each agent
        successor_positions ← empty list of lists
        FOR i FROM 0 TO number_of_agents - 1 DO
            IF i in current.conflict_set THEN
                // Consider all possible moves for agents in conflict
                possible_moves ← GetAllPossibleMoves(current.positions[i])
                ADD possible_moves to successor_positions
            ELSE
                // Follow optimal path for non-conflicting agents
                next_pos ← GetNextPos(individual_paths[i], current.g_score)
                ADD [next_pos] to successor_positions
            END IF
        END FOR
        
        // Generate all combinations of next positions
        FOR EACH combination in CartesianProduct(successor_positions) DO
            // Check for conflicts in this combination
            conflicts ← DetectConflicts(combination)
            
            successor.positions ← combination
            successor.g_score ← current.g_score + 1
            successor.h_score ← sum of h-distances from combination to goals
            successor.conflict_set ← union of current.conflict_set and conflicts
            successor.parent ← current
            
            IF successor not in closed THEN
                INSERT successor into open with priority (successor g+h scores)
            END IF
        END FOR
    END WHILE
    
    RETURN "No solution exists"
END ALGORITHM`,

  pushandrotate: `ALGORITHM PushAndRotate(starts, goals, grid)
    // Initialise agents
    agents ← empty list
    FOR i FROM 0 TO number_of_agents - 1 DO
        agent.id ← i
        agent.start ← starts[i]
        agent.goal ← goals[i]
        agent.current ← starts[i]
        agent.at_goal ← (starts[i] = goals[i])
        ADD agent to agents
    END FOR
    
    // Decompose graph into connected components
    graph ← CreateGraph(grid)
    components ← FindConnectedComponents(graph)
    
    // Assign agents to components and determine priority
    AssignAgentsToComponents(agents, components)
    priority_order ← SortAgentsByPriority(agents, components)
    
    displaced_agents ← empty set
    
    // Process agents in priority order
    FOR EACH agent_id in priority_order DO
        agent ← agents[agent_id]
        
        IF agent is at goal THEN
            CONTINUE
        END IF
        
        path ← GetPath(agent.current, agent.goal)
        IF path does not exist THEN
            RETURN "No solution exists"
        END IF
        
        // Move agent along path
        FOR EACH position in path (excluding start) DO
            occupant ← GetAgentAt(position)
            
            IF no occupant THEN
                Move agent to position
            ELSE
                // Try to resolve conflict
                IF occupant is at goal THEN
                    Mark occupant as displaced
                    ADD occupant.id to displaced_agents
                END IF
                
                IF CanPush(agent, occupant) THEN
                    Push occupant away
                    Move agent to position
                ELSE IF CanSwap(agent, occupant) THEN
                    Swap positions of agent and occupant
                ELSE IF CanRotate(agent, occupant, path) THEN
                    Perform rotation to resolve cycle
                ELSE
                    RETURN "Cannot resolve conflict"
                END IF
            END IF
        END FOR
        
        Mark agent as at goal
        
        // Resolve displaced agents
        ResolveDisplacedAgents(displaced_agents)
    END FOR
    
    RETURN agents
END ALGORITHM`
};

const TokenColors = {
    keyword: "#0A58CA",
    dataStructure: "#6F42C1",
    method: "#FD7E14",
    function: "#FD7E14",
    constant: "#DC3545",
    operator: "#6F42C1",
    string: "#198754",
    number: "#20C997",
    variable: "#0D6EFD",
    collection: "#6610F2",
    comment: "#6C757D" 
  };
  
  const Pseudocode = ({ selectedAlgorithm = "" }) => {
    const code = algorithmPseudocode[selectedAlgorithm] || "";
    
    const lines = code.split("\n");
    
    const colorWord = (word, colorClass) => {
      return <span style={{color: TokenColors[colorClass], fontWeight: colorClass === "keyword" ? "bold" : "normal"}}>{word}</span>;
    };
    
    const styleLine = (line) => {
        const commentIndex = line.indexOf("//");
        if (commentIndex !== -1) {
            const beforeComment = line.substring(0, commentIndex);
            const comment = line.substring(commentIndex);
            return [...styleLine(beforeComment), colorWord(comment, "comment")];
        }
      const result = [];
      let remainingText = line;
      
      while (remainingText.length > 0) {
        const keywordMatch = remainingText.match(/^\b(ALGORITHM|END ALGORITHM|WHILE|END WHILE|FOR|END FOR|IF|THEN|ELSE|END IF|CONTINUE|RETURN|FROM|TO|DO|AND|OR|NOT)\b/);
        if (keywordMatch) {
          result.push(colorWord(keywordMatch[0], "keyword"));
          remainingText = remainingText.substring(keywordMatch[0].length);
          continue;
        }
        
        const dataStructureMatch = remainingText.match(/^\b(queue|set|map|list|priority queue|stack)\b/i);
        if (dataStructureMatch) {
          result.push(colorWord(dataStructureMatch[0], "dataStructure"));
          remainingText = remainingText.substring(dataStructureMatch[0].length);
          continue;
        }
        
        const methodMatch = remainingText.match(/^\b(ENQUEUE|DEQUEUE|PUSH|POP|INSERT|EXTRACT-MIN|ADD|REMOVE|PREPEND|RANDOM|SELECT|CartesianProduct|union)\b/);
        if (methodMatch) {
          result.push(colorWord(methodMatch[0], "method"));
          remainingText = remainingText.substring(methodMatch[0].length);
          continue;
        }
        
        const functionMatch = remainingText.match(/^([A-Z][a-zA-Z0-9_]*)\(/);
        if (functionMatch) {
          result.push(colorWord(functionMatch[1], "function"));
          result.push("(");
          remainingText = remainingText.substring(functionMatch[0].length);
          continue;
        }
        
        const constantMatch = remainingText.match(/^\b(true|false|null|infinity|empty)\b/i);
        if (constantMatch) {
          result.push(colorWord(constantMatch[0], "constant"));
          remainingText = remainingText.substring(constantMatch[0].length);
          continue;
        }
        
        const operatorMatch = remainingText.match(/^(=|←|≠|≤|≥|>|<|\+|-|\*|\/|∞|√)/);
        if (operatorMatch) {
          result.push(colorWord(operatorMatch[0], "operator"));
          remainingText = remainingText.substring(operatorMatch[0].length);
          continue;
        }
        
        const stringMatch = remainingText.match(/^("[^"]*"|"[^"]*")/);
        if (stringMatch) {
          result.push(colorWord(stringMatch[0], "string"));
          remainingText = remainingText.substring(stringMatch[0].length);
          continue;
        }
        
        const numberMatch = remainingText.match(/^\b(\d+)\b/);
        if (numberMatch) {
          result.push(colorWord(numberMatch[0], "number"));
          remainingText = remainingText.substring(numberMatch[0].length);
          continue;
        }
        
        const variableMatch = remainingText.match(/^\b(start|goal|current|neighbor|path|node|agent|conflict|grid|position|successor|combination|next_pos|closed|open|root|child|mdds|h_score|g_score|f_score)\b/);
        if (variableMatch) {
          result.push(colorWord(variableMatch[0], "variable"));
          remainingText = remainingText.substring(variableMatch[0].length);
          continue;
        }
        
        const collectionMatch = remainingText.match(/^\b(open|closed|visited|neighbors|constraints|conflicts|successors|priority queue|frontier|components|displaced_agents|individual_paths)\b/);
        if (collectionMatch) {
          result.push(colorWord(collectionMatch[0], "collection"));
          remainingText = remainingText.substring(collectionMatch[0].length);
          continue;
        }
        
        result.push(remainingText.charAt(0));
        remainingText = remainingText.substring(1);
      }
      
      return result;
    };
  
    return (
      <div className="bg-white border-2 border-black rounded-lg p-3 h-48 overflow-y-auto">
        <pre className="whitespace-pre text-xs font-mono text-left">
          {lines.map((line, index) => (
            <React.Fragment key={index}>
              {styleLine(line)}
              {index < lines.length - 1 ? "\n" : ""}
            </React.Fragment>
          ))}
        </pre>
      </div>
    );
  };

export default Pseudocode;