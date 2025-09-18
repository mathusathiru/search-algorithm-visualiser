import React, { useState, useCallback, useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import PropTypes from "prop-types";
import BaseGrid, { GRID_SIZES } from "./BaseGrid";

const AGENT_COLORS = ["#3366FF", "#FF9933", "#66CC66", "#FF66B2"];

const MultiAgentPathOverlay = ({ walkData, currentStep, cellSize, gridSize }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !walkData || currentStep === null) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const step = walkData[currentStep];
    if (!step || !step.paths) return;

    Object.entries(step.paths).forEach(([agentId, path]) => {
      if (!path || path.length < 2) return;
      
      const color = AGENT_COLORS[agentId % AGENT_COLORS.length];
      
      ctx.lineWidth = 3;
      ctx.strokeStyle = color;
      
      const offset = parseInt(agentId) * 0.08;
      
      ctx.beginPath();
      const [startRow, startCol] = path[0];
      ctx.moveTo(
        (startCol + 0.5 + offset) * cellSize.width, 
        (startRow + 0.5 + offset) * cellSize.height
      );
      
      for (let i = 1; i < path.length; i++) {
        const [row, col] = path[i];
        ctx.lineTo(
          (col + 0.5 + offset) * cellSize.width, 
          (row + 0.5 + offset) * cellSize.height
        );
      }
      
      ctx.stroke();
    });
  }, [walkData, currentStep, cellSize, gridSize.cols, gridSize.rows]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 10
      }}
      width={gridSize.cols * cellSize.width}
      height={gridSize.rows * cellSize.height}
    />
  );
};

const MultiAgentGrid = forwardRef(({
  size = "medium",
  walkData,
  currentStep,
  onAgentPositionsChange,
  isRunning = false,
  isFinished = false,
  walls,
  onWallsChange,
  onNodeDragEnd,
  agentStarts = [],
  agentGoals = [],
}, ref) => {
  // Initialise four default agents
  const initialiseAgents = useCallback(() => {
    const { rows, cols } = GRID_SIZES[size];
    
    const startColumn = Math.floor(cols / 5);
    const goalColumn = Math.floor(4 * cols / 5); 
    
    const spacing = Math.floor(rows / 5);
    const startRow = Math.floor(rows / 4);

    const defaultAgents = [
      { id: 0, start: [startRow, startColumn], goal: [startRow, goalColumn], color: AGENT_COLORS[0] },
      { id: 1, start: [startRow + spacing, startColumn], goal: [startRow + spacing, goalColumn], color: AGENT_COLORS[1] },
      { id: 2, start: [startRow + 2 * spacing, startColumn], goal: [startRow + 2 * spacing, goalColumn], color: AGENT_COLORS[2] },
      { id: 3, start: [startRow + 3 * spacing, startColumn], goal: [startRow + 3 * spacing, goalColumn], color: AGENT_COLORS[3] }
    ];
    
    return defaultAgents;
  }, [size]);
  
  const [agents, setAgents] = useState(initialiseAgents);
  const [cellSize, setCellSize] = useState({ width: 0, height: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [draggedNode, setDraggedNode] = useState(null);
  const [dragPosition, setDragPosition] = useState(null);
  
  // Handle node randomization
  const randomizeNodes = useCallback(() => {
    if (isRunning) return;
    
    const { rows, cols } = GRID_SIZES[size];
    const newAgents = [...agents];
    const MAX_ATTEMPTS = 100;
    
    // For each agent
    for (let i = 0; i < newAgents.length; i++) {
      let newStartRow, newStartCol, newGoalRow, newGoalCol;
      let attempts = 0;
      
      // Generate start position
      do {
        newStartRow = Math.floor(Math.random() * rows);
        newStartCol = Math.floor(Math.random() * cols);
        attempts++;
        
        if (attempts > MAX_ATTEMPTS) {
          const startColumn = Math.floor(cols / 5);
          const startRow = Math.floor(rows / 3);
          const spacing = Math.floor(rows / 6);
          
          newStartRow = Math.min(startRow + (i * spacing), rows - 1);
          newStartCol = Math.min(startColumn, cols - 1);
          break;
        }
      } while (
        walls[newStartRow][newStartCol] || 
        newAgents.some((a, idx) => idx !== i && 
          a.start[0] === newStartRow && a.start[1] === newStartCol)
      );
      
      // Reset attempts counter
      attempts = 0;
      
      // Generate goal position
      do {
        newGoalRow = Math.floor(Math.random() * rows);
        newGoalCol = Math.floor(Math.random() * cols);
        attempts++;
        
        if (attempts > MAX_ATTEMPTS) {
          const goalColumn = Math.floor(4 * cols / 5);
          const startRow = Math.floor(rows / 3);
          const spacing = Math.floor(rows / 6);
          
          newGoalRow = Math.min(startRow + (i * spacing), rows - 1);
          newGoalCol = Math.min(goalColumn, cols - 1);
          break;
        }
      } while (
        walls[newGoalRow][newGoalCol] || 
        (newGoalRow === newStartRow && newGoalCol === newStartCol) ||
        newAgents.some((a, idx) => idx !== i && 
          (a.goal[0] === newGoalRow && a.goal[1] === newGoalCol ||
           a.start[0] === newGoalRow && a.start[1] === newGoalCol))
      );
      
      newAgents[i].start = [newStartRow, newStartCol];
      newAgents[i].goal = [newGoalRow, newGoalCol];
    }
    
    setAgents([...newAgents]);
    
    // Creating new arrays for starts and goals
    const newStarts = newAgents.map(a => [...a.start]);
    const newGoals = newAgents.map(a => [...a.goal]);
    
    // Notify parent component
    if (onAgentPositionsChange) {
      onAgentPositionsChange(newStarts, newGoals);
    }
    
    // Recalculate path if walksteps exist
    if (walkData?.length && !isRunning && onNodeDragEnd) {
      onNodeDragEnd(newStarts, newGoals);
    }
  }, [agents, walls, size, isRunning, onAgentPositionsChange, walkData, onNodeDragEnd]);

  // Update agent positions externally (for undo/redo)
  const updateAgentPositions = useCallback((starts, goals) => {
    if (!starts || !goals || starts.length === 0 || goals.length === 0) return;
    
    // Creating new agents array with updated positions
    const updatedAgents = starts.map((start, index) => ({
      id: index,
      start: [...start],
      goal: index < goals.length ? [...goals[index]] : [...start],
      color: AGENT_COLORS[index % AGENT_COLORS.length]
    }));
    
    setAgents(updatedAgents);
  }, []);

  useImperativeHandle(ref, () => ({
    randomizeNodes,
    initialiseAgents,
    updateAgentPositions,
    handleGridSizeChange: (oldSize, newSize) => {
      const oldGridDimensions = GRID_SIZES[oldSize];
      const newGridDimensions = GRID_SIZES[newSize];
      
      // Calculating scaling factors for rows and columns
      const rowScale = newGridDimensions.rows / oldGridDimensions.rows;
      const colScale = newGridDimensions.cols / oldGridDimensions.cols;
      
      // Adjust positions
      const newAgents = [...agents];
      
      // Helper to get unique key for position
      const posKey = (row, col) => `${row},${col}`;
      
      // Helper to find nearby empty position
      const findEmptyPosition = (row, col) => {
        const maxDistance = 3;
        const positionsMap = new Map();
        
        // Mark all current positions as occupied
        newAgents.forEach(agent => {
          positionsMap.set(posKey(agent.start[0], agent.start[1]), `start_${agent.id}`);
          positionsMap.set(posKey(agent.goal[0], agent.goal[1]), `goal_${agent.id}`);
        });
        
        for (let d = 1; d <= maxDistance; d++) {
          for (let dr = -d; dr <= d; dr++) {
            for (let dc = -d; dc <= d; dc++) {
              // Skip if not on sqaure perimeter with distance d
              if (Math.abs(dr) !== d && Math.abs(dc) !== d) continue;
              
              const newRow = row + dr;
              const newCol = col + dc;
              
              // Ensure position within grid bounds
              if (newRow < 0 || newRow >= newGridDimensions.rows || 
                  newCol < 0 || newCol >= newGridDimensions.cols) {
                continue;
              }
              
              // Check if the position is already occupied
              const key = posKey(newRow, newCol);
              if (!positionsMap.has(key)) {
                return [newRow, newCol];
              }
            }
          }
        }
        
        // If no empty position found, return a random position as last resort
        return [
          Math.floor(Math.random() * newGridDimensions.rows),
          Math.floor(Math.random() * newGridDimensions.cols)
        ];
      };
      
      // Process each agent
      for (let agent of newAgents) {
        // Scale the positions and ensure they"re within the new grid boundaries
        let newStartRow = Math.min(Math.floor(agent.start[0] * rowScale), newGridDimensions.rows - 1);
        let newStartCol = Math.min(Math.floor(agent.start[1] * colScale), newGridDimensions.cols - 1);
        let newGoalRow = Math.min(Math.floor(agent.goal[0] * rowScale), newGridDimensions.rows - 1);
        let newGoalCol = Math.min(Math.floor(agent.goal[1] * colScale), newGridDimensions.cols - 1);
        
        agent.start = [newStartRow, newStartCol];
        agent.goal = [newGoalRow, newGoalCol];
      }
      
      // Resolve collisions
      const startPositions = new Map();
      const goalPositions = new Map();
      
      for (let agent of newAgents) {
        // Check start position collision
        let startKey = posKey(agent.start[0], agent.start[1]);
        if (startPositions.has(startKey)) {
          // Find a new position
          const [newRow, newCol] = findEmptyPosition(agent.start[0], agent.start[1]);
          agent.start = [newRow, newCol];
          startKey = posKey(newRow, newCol);
        }
        startPositions.set(startKey, agent.id);
        
        // Check goal position collision
        let goalKey = posKey(agent.goal[0], agent.goal[1]);
        if (goalPositions.has(goalKey)) {
          // Find a new position
          const [newRow, newCol] = findEmptyPosition(agent.goal[0], agent.goal[1]);
          agent.goal = [newRow, newCol];
          goalKey = posKey(newRow, newCol);
        }
        goalPositions.set(goalKey, agent.id);
      }
      
      // Update state
      setAgents(newAgents);
      
      // Return the new positions for the parent component
      return {
        newStarts: newAgents.map(a => [...a.start]),
        newGoals: newAgents.map(a => [...a.goal])
      };
    }
  }));
  
  // Update useEffect to initialise agents properly accounting for external changes
  useEffect(() => {
    // Only initialise if we don"t already have agents
    if (agents.length === 0) {
      const initialAgents = initialiseAgents();
      setAgents(initialAgents);
      
      // Notify parent component
      if (onAgentPositionsChange) {
        const starts = initialAgents.map(a => [...a.start]);
        const goals = initialAgents.map(a => [...a.goal]);
        onAgentPositionsChange(starts, goals);
      }
    }
  }, [agents.length, initialiseAgents, onAgentPositionsChange]);

  useEffect(() => {
    if (!agentStarts || !agentGoals || agentStarts.length === 0 || agentGoals.length === 0) {
      return;
    }

    if (agentStarts.length !== agents.length) {
      const newAgents = agentStarts.map((start, i) => ({
        id: i,
        start: [...start],
        goal: [...(agentGoals[i] || start)], 
        color: AGENT_COLORS[i % AGENT_COLORS.length]
      }));
      
      setAgents(newAgents);
      return;
    }
    
    const positionsChanged = agents.some((a, i) => {
      if (!agentStarts[i] || !agentGoals[i]) return false;
      
      return (
        a.start[0] !== agentStarts[i][0] || 
        a.start[1] !== agentStarts[i][1] || 
        a.goal[0] !== agentGoals[i][0] || 
        a.goal[1] !== agentGoals[i][1]
      );
    });
    
    if (positionsChanged) {
      // Create new agents array from the starts and goals
      const newAgents = agentStarts.map((start, i) => ({
        id: i,
        start: [...start],
        goal: [...(agentGoals[i] || start)],
        color: AGENT_COLORS[i % AGENT_COLORS.length]
      }));
      
      setAgents(newAgents);
    }
  }, [agentStarts, agentGoals, agents]);
  const notifyParentOfChanges = useCallback(() => {
    if (!onAgentPositionsChange) return;
    
    const newStarts = agents.map(a => [...a.start]);
    const newGoals = agents.map(a => [...a.goal]);
    
    onAgentPositionsChange(newStarts, newGoals);
  }, [agents, onAgentPositionsChange]);
  
  useEffect(() => {
    notifyParentOfChanges();
  }, [notifyParentOfChanges]);
  
  const getPathDirection = useCallback((path, rowIndex, colIndex) => {
    if (!path || path.length < 2) return "horizontal";
    
    const pathIndex = path.findIndex(([r, c]) => r === rowIndex && c === colIndex);
    if (pathIndex === -1) return "horizontal";
    
    if (pathIndex > 0 && pathIndex < path.length - 1) {
      const prevCell = path[pathIndex - 1];
      const nextCell = path[pathIndex + 1];
      
      const horizontalMovement = Math.abs(nextCell[1] - prevCell[1]);
      const verticalMovement = Math.abs(nextCell[0] - prevCell[0]);
      
      return horizontalMovement >= verticalMovement ? "horizontal" : "vertical";
    } else if (pathIndex === 0 && path.length > 1) {
      const nextCell = path[1];
      return Math.abs(nextCell[1] - colIndex) >= Math.abs(nextCell[0] - rowIndex) ? "horizontal" : "vertical";
    } else if (pathIndex === path.length - 1 && path.length > 1) {
      const prevCell = path[path.length - 2];
      return Math.abs(prevCell[1] - colIndex) >= Math.abs(prevCell[0] - rowIndex) ? "horizontal" : "vertical";
    }
    
    return "horizontal";
  }, []);
  
const getAgentsWithCellInPath = useCallback((step, rowIndex, colIndex) => {
  const agentsWithPaths = [];
  if (!step || !step.paths) return agentsWithPaths;
  
  for (const agentId in step.paths) {
    const path = step.paths[agentId];
    // Make sure path is an array before calling .some()
    if (path && Array.isArray(path) && path.some(([r, c]) => r === rowIndex && c === colIndex)) {
      agentsWithPaths.push(parseInt(agentId));
    }
  }
  return agentsWithPaths;
}, []);

const getAgentsWithCellInVisited = useCallback((step, rowIndex, colIndex) => {
  const agentsWithVisited = [];
  if (!step || !step.visited) return agentsWithVisited;
  
  for (const agentId in step.visited) {
    const visitedCells = step.visited[agentId];
    // Make sure visitedCells is an array before calling .some()
    if (visitedCells && Array.isArray(visitedCells) && visitedCells.some(([r, c]) => r === rowIndex && c === colIndex)) {
      agentsWithVisited.push(parseInt(agentId));
    }
  }
  return agentsWithVisited;
}, []);

const getAgentsWithCellInConsidered = useCallback((step, rowIndex, colIndex) => {
  const agentsWithConsidered = [];
  if (!step || !step.considered) return agentsWithConsidered;
  
  for (const agentId in step.considered) {
    const consideredCells = step.considered[agentId];
    // Ensuring consideredCells is an array before calling .some()
    if (consideredCells && Array.isArray(consideredCells) && consideredCells.some(([r, c]) => r === rowIndex && c === colIndex)) {
      agentsWithConsidered.push(parseInt(agentId));
    }
  }
  return agentsWithConsidered;
}, []);
  
  const handleNodeMouseDown = useCallback((e, agentId, type, row, col) => {
    if (isRunning) return;
    
    e.preventDefault();
    setIsDragging(true);
    setDraggedNode({ agentId, type, startRow: row, startCol: col });
    setDragPosition({
      x: col * cellSize.width + cellSize.width / 2,
      y: row * cellSize.height + cellSize.height / 2,
    });
  }, [isRunning, cellSize]);
  
  const handleMouseMove = useCallback((e) => {
    if (!isDragging || !dragPosition) return;
    
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    setDragPosition({
      x: Math.max(0, Math.min(rect.width, x)),
      y: Math.max(0, Math.min(rect.height, y)),
    });
  }, [isDragging, dragPosition]);
  
  const handleMouseUp = useCallback(async () => {
    if (!isDragging || !dragPosition || !draggedNode) {
      setIsDragging(false);
      setDraggedNode(null);
      setDragPosition(null);
      return;
    }
    
    const { rows, cols } = GRID_SIZES[size];
    const newCol = Math.min(
      Math.max(0, Math.round(dragPosition.x / cellSize.width - 0.5)),
      cols - 1
    );
    const newRow = Math.min(
      Math.max(0, Math.round(dragPosition.y / cellSize.height - 0.5)),
      rows - 1
    );
        
    if (!walls[newRow][newCol]) {
      // Creating new agents array to ensure proper state update
      const newAgents = agents.map(a => ({
        ...a,
        start: [...a.start],
        goal: [...a.goal]
      }));
      
      const agent = newAgents[draggedNode.agentId];
      
      if (!agent) {
        setIsDragging(false);
        setDraggedNode(null);
        setDragPosition(null);
        return;
      }
      
      const isPositionTaken = newAgents.some((a, idx) => {
        if (idx === draggedNode.agentId) return false;
        return draggedNode.type === "start" ? 
          (a.start[0] === newRow && a.start[1] === newCol) ||
          (a.goal[0] === newRow && a.goal[1] === newCol) :
          (a.start[0] === newRow && a.start[1] === newCol) ||
          (a.goal[0] === newRow && a.goal[1] === newCol);
      });
      
      const selfCollision = draggedNode.type === "start" ?
        (agent.goal[0] === newRow && agent.goal[1] === newCol) :
        (agent.start[0] === newRow && agent.start[1] === newCol);
      
      if (!isPositionTaken && !selfCollision) {
        // Updating position
        if (draggedNode.type === "start") {
          agent.start = [newRow, newCol];
        } else {
          agent.goal = [newRow, newCol];
        }
        
        // Updating local state
        setAgents(newAgents);
        
        // Creating new arrays for starts and goals
        const newStarts = newAgents.map(a => [...a.start]);
        const newGoals = newAgents.map(a => [...a.goal]);
        
        // Updating parent
        if (onAgentPositionsChange) {
          onAgentPositionsChange(newStarts, newGoals);
        }
        
        // Recalculating path if needed
        if (walkData?.length && !isRunning && onNodeDragEnd) {
          try {
            await onNodeDragEnd(newStarts, newGoals, walls);
          } catch (error) {
          }
        }
      }
    }
    
    setIsDragging(false);
    setDraggedNode(null);
    setDragPosition(null);
  }, [isDragging, dragPosition, draggedNode, size, cellSize, walls, agents, onAgentPositionsChange, walkData, isRunning, onNodeDragEnd]);
  
  const handleCellClick = useCallback((rowIndex, colIndex) => {
    if (isRunning) return;
    
    const isOccupied = agents.some(agent => 
      (agent.start[0] === rowIndex && agent.start[1] === colIndex) || 
      (agent.goal[0] === rowIndex && agent.goal[1] === colIndex)
    );
    
    if (isOccupied) return;
    
    const newWalls = walls.map(row => [...row]);
    newWalls[rowIndex][colIndex] = !newWalls[rowIndex][colIndex];
    onWallsChange?.(newWalls);
    
    // Adding recalculation when changing walls
    if (walkData?.length && !isRunning && onNodeDragEnd) {
      const starts = agents.map(a => [...a.start]);
      const goals = agents.map(a => [...a.goal]);
      
      onNodeDragEnd(starts, goals, newWalls);
    }
  }, [isRunning, agents, walls, onWallsChange, walkData, onNodeDragEnd]);
  
  const getCellBackgroundColor = useCallback((rowIndex, colIndex, isWall) => {
    if (isWall) return "black";
    
    if (walkData && currentStep !== null && walkData[currentStep]) {
      const step = walkData[currentStep];
      
      const pathAgents = getAgentsWithCellInPath(step, rowIndex, colIndex);
      const visitedAgents = step.visited ? getAgentsWithCellInVisited(step, rowIndex, colIndex) : [];
      const consideredAgents = step.considered ? getAgentsWithCellInConsidered(step, rowIndex, colIndex) : [];
      
      const visitedAndPathAgents = [...new Set([...pathAgents, ...visitedAgents])];
      const allAgents = [...new Set([...visitedAndPathAgents, ...consideredAgents])];
      
      if (allAgents.length > 1) {
        return "transparent";
      }
      
      for (const agentId in step.positions) {
        const [r, c] = step.positions[agentId];
        const isActivePosition = r === rowIndex && c === colIndex;
        const agent = agents.find(a => a.id === parseInt(agentId));
        const isGoalNode = agent && agent.goal[0] === rowIndex && agent.goal[1] === colIndex;
        
        if (isActivePosition && !isGoalNode) {
          return AGENT_COLORS[agentId % AGENT_COLORS.length];
        }
      }
      
      if (allAgents.length === 1) {
        const agentId = allAgents[0];
        if (visitedAndPathAgents.includes(agentId)) {
          return `${AGENT_COLORS[agentId % AGENT_COLORS.length]}60`;
        } else if (consideredAgents.includes(agentId)) {
          return `${AGENT_COLORS[agentId % AGENT_COLORS.length]}35`;
        }
      }
    }
    
    return "white";
  }, [walkData, currentStep, agents, getAgentsWithCellInPath, getAgentsWithCellInVisited, getAgentsWithCellInConsidered]);
  
  const getCellContent = useCallback((rowIndex, colIndex) => {
    if (!walkData || currentStep === null || !walkData[currentStep]) return null;
  
    const step = walkData[currentStep];
    
    // Safely get paths, visited, and considered agents
    const pathAgents = getAgentsWithCellInPath(step, rowIndex, colIndex);
    const visitedAgents = step.visited ? getAgentsWithCellInVisited(step, rowIndex, colIndex) : [];
    const consideredAgents = step.considered ? getAgentsWithCellInConsidered(step, rowIndex, colIndex) : [];
    
    // Ensure defined values are used, empty arrays as fallbacks for spread operations
    const visitedAndPathAgents = [...new Set([...(pathAgents || []), ...(visitedAgents || [])])];
    const allAgents = [...new Set([...(visitedAndPathAgents || []), ...(consideredAgents || [])])];
    
    if (allAgents.length > 1) {
      const agentInfo = allAgents.map(agentId => {
        let opacity = 0.35;
        let priority = 1;
        
        if (visitedAndPathAgents.includes(agentId)) {
          opacity = 0.6;
          priority = 2;
        }
        
        return { agentId, opacity, priority };
      });
      
      let horizontalCount = 0;
      let verticalCount = 0;
      
      for (const agentId of allAgents) {
        if (pathAgents && pathAgents.includes(agentId) && step.paths && step.paths[agentId]) {
          const direction = getPathDirection(step.paths[agentId], rowIndex, colIndex);
          if (direction === "horizontal") {
            horizontalCount++;
          } else {
            verticalCount++;
          }
        } else {
          // Ensure agents[agentId] exists before accessing properties
          if (agents[agentId]) {
            const [startRow, startCol] = agents[agentId].start;
            const [goalRow, goalCol] = agents[agentId].goal;
            
            const horizontalDist = Math.abs(goalCol - startCol);
            const verticalDist = Math.abs(goalRow - startRow);
            
            if (horizontalDist >= verticalDist) {
              horizontalCount++;
            } else {
              verticalCount++;
            }
          } else {
            horizontalCount++;
          }
        }
      }
      
      const useHorizontalSplit = horizontalCount >= verticalCount;
      const splitSize = 100 / allAgents.length;
      
      return (
        <div style={{ 
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: useHorizontalSplit ? "row" : "column",
          zIndex: 1
        }}>
          {allAgents.map(agentId => {
            const info = agentInfo.find(info => info.agentId === agentId);
            return (
              <div 
                key={agentId}
                style={{
                  width: useHorizontalSplit ? `${splitSize}%` : "100%",
                  height: useHorizontalSplit ? "100%" : `${splitSize}%`,
                  backgroundColor: `${AGENT_COLORS[agentId % AGENT_COLORS.length]}${Math.round(info.opacity * 100)}`,
                }}
              />
            );
          })}
        </div>
      );
    }
    
    return null;
  }, [walkData, currentStep, getAgentsWithCellInPath, getAgentsWithCellInVisited, getAgentsWithCellInConsidered, getPathDirection, agents]);
  
  useEffect(() => {
    const updateCellSize = () => {
      const gridElement = document.querySelector(".grid-wrapper");
      if (gridElement) {
        const rect = gridElement.getBoundingClientRect();
        setCellSize({
          width: rect.width / GRID_SIZES[size].cols,
          height: rect.height / GRID_SIZES[size].rows,
        });
      }
    };
    
    updateCellSize();
    
    const resizeObserver = new ResizeObserver(updateCellSize);
    const gridElement = document.querySelector(".grid-wrapper");
    if (gridElement) {
      resizeObserver.observe(gridElement);
    }
    
    return () => {
      if (gridElement) {
        resizeObserver.unobserve(gridElement);
      }
    };
  }, [size]);
  
  return (
    <BaseGrid
      size={size}
      walls={walls}
      onWallsChange={onWallsChange}
      isRunning={isRunning}
      getCellBackgroundColor={getCellBackgroundColor}
      getCellContent={getCellContent}
      onCellClick={handleCellClick}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {cellSize.width > 0 && walkData && (
        <MultiAgentPathOverlay
          walkData={walkData}
          currentStep={currentStep}
          cellSize={cellSize}
          gridSize={GRID_SIZES[size]}
        />
      )}
      
      {cellSize.width > 0 && agents.map((agent) => {
        const atGoal = walkData && currentStep !== null && walkData[currentStep]?.atGoal?.[agent.id];
        
        const isMoving = isRunning;
        
        const startIsBeingDragged = isDragging && 
          draggedNode?.agentId === agent.id && 
          draggedNode?.type === "start";
          
        const goalIsBeingDragged = isDragging && 
          draggedNode?.agentId === agent.id && 
          draggedNode?.type === "end";
          
        const startCenterX = agent.start[1] * cellSize.width + cellSize.width / 2;
        const startCenterY = agent.start[0] * cellSize.height + cellSize.height / 2;
        
        const goalCenterX = agent.goal[1] * cellSize.width + cellSize.width / 2;
        const goalCenterY = agent.goal[0] * cellSize.height + cellSize.height / 2;
        
        return (
          <React.Fragment key={agent.id}>
            {/* Start node */}
            <div
              className="absolute flex items-center justify-center grid-node start"
              style={{
                backgroundColor: AGENT_COLORS[agent.id % AGENT_COLORS.length],
                left: startIsBeingDragged ? `${dragPosition.x}px` : `${startCenterX}px`,
                top: startIsBeingDragged ? `${dragPosition.y}px` : `${startCenterY}px`,
                transform: "translate(-50%, -50%)",
                cursor: isRunning ? "not-allowed" : "move",
                transition: startIsBeingDragged ? "none" : "all 0.05s ease-out",
                zIndex: 20,
                boxShadow: atGoal ? "0 0 10px rgba(255, 255, 255, 0.5)" : "none",
                animation: isMoving ? "pulse 0.5s ease-in-out" : "none",
                width: "1rem",
                height: "1rem",
                borderRadius: "50%",
                pointerEvents: isRunning ? "none" : "auto"
              }}
              onMouseDown={(e) => handleNodeMouseDown(e, agent.id, "start", agent.start[0], agent.start[1])}
            >
              <svg width="20" height="20" className="pointer-events-none">
                <text
                  x="50%"
                  y="50%"
                  dominantBaseline="middle"
                  textAnchor="middle"
                  fill="white"
                  fontSize="8"
                  fontFamily="Arial"
                >
                  S
                  <tspan baselineShift="sub" fontSize="6">
                    {agent.id + 1}
                  </tspan>
                </text>
              </svg>
            </div>
            
            {/* Goal node */}
            <div
              className="absolute flex items-center justify-center grid-node end"
              style={{
                border: `2px solid ${AGENT_COLORS[agent.id % AGENT_COLORS.length]}`,
                backgroundColor: "white",
                left: goalIsBeingDragged ? `${dragPosition.x}px` : `${goalCenterX}px`,
                top: goalIsBeingDragged ? `${dragPosition.y}px` : `${goalCenterY}px`,
                transform: "translate(-50%, -50%)",
                cursor: isRunning ? "not-allowed" : "move",
                transition: goalIsBeingDragged ? "none" : "all 0.05s ease-out",
                zIndex: 20,
                boxShadow: atGoal ? "0 0 10px rgba(255, 255, 255, 0.5)" : "none",
                width: "1rem",
                height: "1rem",
                borderRadius: "50%",
                pointerEvents: isRunning ? "none" : "auto"
              }}
              onMouseDown={(e) => handleNodeMouseDown(e, agent.id, "end", agent.goal[0], agent.goal[1])}
            >
              <svg width="20" height="20" className="pointer-events-none">
                <text
                  x="50%"
                  y="50%"
                  dominantBaseline="middle"
                  textAnchor="middle"
                  fill={AGENT_COLORS[agent.id % AGENT_COLORS.length]}
                  fontSize="8"
                  fontFamily="Arial"
                >
                  G
                  <tspan baselineShift="sub" fontSize="6">
                    {agent.id + 1}
                  </tspan>
                </text>
              </svg>
            </div>
          </React.Fragment>
        );
      })}
    </BaseGrid>
  );
});

MultiAgentGrid.propTypes = {
  size: PropTypes.oneOf(["small", "medium", "large"]),
  walkData: PropTypes.array,
  currentStep: PropTypes.number,
  onAgentPositionsChange: PropTypes.func,
  isRunning: PropTypes.bool,
  isFinished: PropTypes.bool,
  walls: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.bool)),
  onWallsChange: PropTypes.func,
  onNodeDragEnd: PropTypes.func,
  agentStarts: PropTypes.array,
  agentGoals: PropTypes.array,
};

MultiAgentPathOverlay.propTypes = {
  walkData: PropTypes.array,
  currentStep: PropTypes.number,
  cellSize: PropTypes.shape({
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired,
  }).isRequired,
  gridSize: PropTypes.shape({
    rows: PropTypes.number.isRequired,
    cols: PropTypes.number.isRequired,
  }).isRequired,
};

export default MultiAgentGrid;