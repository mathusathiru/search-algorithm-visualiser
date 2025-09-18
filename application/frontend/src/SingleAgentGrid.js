import React, { useState, useCallback, useEffect, useRef, forwardRef, useImperativeHandle } from "react";
import PropTypes from "prop-types";
import BaseGrid, { GRID_SIZES } from "./BaseGrid";
import { getVisitedColor } from "./ColourConverter";

const WEIGHTED_ALGORITHMS = ["dijkstra", "astar"];

const GridNode = React.memo(({
  isStart,
  rowIndex,
  colIndex,
  isBeingDragged,
  dragPosition,
  centerX,
  centerY,
  isRunning,
  onNodeMouseDown
}) => (
  <div
    className={`
      grid-node 
      ${isStart ? "start" : "end"}
      ${isRunning ? "disabled" : "draggable"}
    `}
    style={{
      left: isBeingDragged ? `${dragPosition.x}px` : `${centerX}px`,
      top: isBeingDragged ? `${dragPosition.y}px` : `${centerY}px`,
      transition: isBeingDragged ? "none" : "all 0.05s ease-out",
      position: "absolute"
    }}
    onMouseDown={(e) => onNodeMouseDown(e, isStart ? "start" : "end", rowIndex, colIndex)}
  />
));

const SingleAgentGrid = forwardRef(({
  size = "medium",
  walkData,
  currentStep,
  onNodePositionsChange,
  isRunning = false,
  walls,
  onWallsChange,
  initialAltitudes = null,
  selectedAlgorithm = "",
  onNodeDragEnd,
}, ref) => {
  const [nodes, setNodes] = useState(() => {
    const { rows, cols } = GRID_SIZES[size];
    const startRow = Math.min(Math.floor(rows / 4), rows - 1);
    const startCol = Math.min(Math.floor(cols / 4), cols - 1);
    const endRow = Math.min(Math.floor((3 * rows) / 4), rows - 1);
    const endCol = Math.min(Math.floor((3 * cols) / 4), cols - 1);

    return Array(rows).fill().map((_, row) =>
      Array(cols).fill().map((_, col) => ({
        row,
        col,
        isStart: row === startRow && col === startCol,
        isEnd: row === endRow && col === endCol,
      }))
    );
  });

  const [isDragging, setIsDragging] = useState(false);
  const [draggedNode, setDraggedNode] = useState(null);
  const [dragPosition, setDragPosition] = useState(null);
  const [cellSize, setCellSize] = useState({ width: 0, height: 0 });
  
  const pathCanvasRef = useRef(null);

  // Draw the path visualisation when walkData or currentStep changes
  useEffect(() => {
    const canvas = pathCanvasRef.current;
    if (!canvas || !walkData || currentStep === null || !cellSize.width) return;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const step = walkData[currentStep];
    if (!step) return;

    const isWeightedAlgorithm = WEIGHTED_ALGORITHMS.includes(selectedAlgorithm);

    // Draw parent connections for weighted algorithms
    if (isWeightedAlgorithm && step.parent) {
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = "#f88379";
      Object.entries(step.parent).forEach(([nodeStr, parentStr]) => {
        if (parentStr === "null" || !parentStr) return;
        
        try {
          const [currentRow, currentCol] = JSON.parse(nodeStr);
          const [parentRow, parentCol] = JSON.parse(parentStr);
          
          const startX = (parentCol + 0.5) * cellSize.width;
          const startY = (parentRow + 0.5) * cellSize.height;
          const endX = (currentCol + 0.5) * cellSize.width;
          const endY = (currentRow + 0.5) * cellSize.height;

          ctx.beginPath();
          ctx.moveTo(startX, startY);
          ctx.lineTo(endX, endY);
          ctx.stroke();
        } catch (error) {
          // Handle potential JSON parse errors gracefully
          console.warn("Error parsing node coordinates:", error);
        }
      });
    }

    // Draw final path if available
    if (step.path && step.isGoalReached) {
      ctx.lineWidth = 3.5;
      ctx.strokeStyle = "#4DC6FF";
      
      // For weighted algorithms, draw a brighter path when we reach the final node
      if (isWeightedAlgorithm) {
        // Add a subtle glow effect
        ctx.shadowColor = "#4DC6FF";
        ctx.shadowBlur = 5;
      }
      
      // Draw the path connections
      for (let i = 1; i < step.path.length; i++) {
        const [prevRow, prevCol] = step.path[i - 1];
        const [currentRow, currentCol] = step.path[i];

        const startX = (prevCol + 0.5) * cellSize.width;
        const startY = (prevRow + 0.5) * cellSize.height;
        const endX = (currentCol + 0.5) * cellSize.width;
        const endY = (currentRow + 0.5) * cellSize.height;

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      }

      // Reset shadow for performance
      if (isWeightedAlgorithm) {
        ctx.shadowBlur = 0;
      }
      
      // Draw direction arrows on the path for all algorithms
      ctx.strokeStyle = "black";
      ctx.fillStyle = "black";
      ctx.lineWidth = 2;
      
      for (let i = 0; i < step.path.length - 1; i++) {
        const [currentRow, currentCol] = step.path[i];
        const [nextRow, nextCol] = step.path[i + 1];
        
        // Calculate center points
        const startX = (currentCol + 0.5) * cellSize.width;
        const startY = (currentRow + 0.5) * cellSize.height;
        const endX = (nextCol + 0.5) * cellSize.width;
        const endY = (nextRow + 0.5) * cellSize.height;
        
        // Calculate direction and midpoint
        const dx = endX - startX;
        const dy = endY - startY;
        const midX = startX + dx * 0.5;
        const midY = startY + dy * 0.5;
        
        // Calculate arrow properties
        const arrowLength = Math.min(cellSize.width, cellSize.height) * 0.25;
        const angle = Math.atan2(dy, dx);
        const arrowWidth = Math.min(cellSize.width, cellSize.height) * 0.15;
        
        // Draw arrow
        ctx.save();
        ctx.translate(midX, midY);
        ctx.rotate(angle);
        
        // Draw arrowhead
        ctx.beginPath();
        ctx.moveTo(arrowLength, 0);
        ctx.lineTo(arrowLength - arrowWidth, -arrowWidth);
        ctx.lineTo(arrowLength - arrowWidth, arrowWidth);
        ctx.closePath();
        ctx.fill();
        
        ctx.restore();
      }
    }
  }, [walkData, currentStep, cellSize, selectedAlgorithm]);

  const randomizeNodes = useCallback(() => {
    if (isRunning) return;
    
    const { rows, cols } = GRID_SIZES[size];
    let newStartRow, newStartCol, newEndRow, newEndCol;
    let attempts = 0;
    const MAX_ATTEMPTS = 100;
    
    do {
      newStartRow = Math.floor(Math.random() * rows);
      newStartCol = Math.floor(Math.random() * cols);
      attempts++;
      
      if (attempts > MAX_ATTEMPTS) {
        newStartRow = Math.floor(rows / 4);
        newStartCol = Math.floor(cols / 4);
        break;
      }
    } while (walls[newStartRow][newStartCol]);
    
    attempts = 0;
    
    do {
      newEndRow = Math.floor(Math.random() * rows);
      newEndCol = Math.floor(Math.random() * cols);
      attempts++;
      
      if (attempts > MAX_ATTEMPTS) {
        newEndRow = Math.floor(3 * rows / 4);
        newEndCol = Math.floor(3 * cols / 4);
        break;
      }
    } while (walls[newEndRow][newEndCol] || 
             (newEndRow === newStartRow && newEndCol === newStartCol));
    
    const newNodes = nodes.map(row => 
      row.map(node => ({
        ...node, 
        isStart: false, 
        isEnd: false
      }))
    );
    
    newNodes[newStartRow][newStartCol].isStart = true;
    newNodes[newEndRow][newEndCol].isEnd = true;
    
    setNodes(newNodes);
    
    const newStart = [newStartRow, newStartCol];
    const newEnd = [newEndRow, newEndCol];
    onNodePositionsChange?.(newStart, newEnd);
    
    // No recalculation for random walk algorithms
    const isRandomWalkAlgorithm = ['randomwalk', 'biasedrandomwalk'].includes(selectedAlgorithm);
    
    if (walkData?.length && !isRunning && onNodeDragEnd && !isRandomWalkAlgorithm) {
      onNodeDragEnd(newStart, newEnd);
    }
  }, [nodes, walls, size, isRunning, onNodePositionsChange, walkData, onNodeDragEnd, selectedAlgorithm]);

  // Method to update node positions externally (for undo/redo)
  const updateNodePositions = useCallback((start, end) => {
    if (!start || !end) return;
    
    const newNodes = nodes.map(row => 
      row.map(node => ({
        ...node, 
        isStart: false, 
        isEnd: false
      }))
    );
    
    // Set start position
    if (start && start.length === 2) {
      const [startRow, startCol] = start;
      if (newNodes[startRow] && newNodes[startRow][startCol]) {
        newNodes[startRow][startCol].isStart = true;
      }
    }
    
    // Set end position
    if (end && end.length === 2) {
      const [endRow, endCol] = end;
      if (newNodes[endRow] && newNodes[endRow][endCol]) {
        newNodes[endRow][endCol].isEnd = true;
      }
    }
    
    setNodes(newNodes);
  }, [nodes]);

  useImperativeHandle(ref, () => ({
    randomizeNodes,
    updateNodePositions
  }));

  useEffect(() => {
    const { rows, cols } = GRID_SIZES[size];
    const startRow = Math.min(Math.floor(rows / 4), rows - 1);
    const startCol = Math.min(Math.floor(cols / 4), cols - 1);
    const endRow = Math.min(Math.floor((3 * rows) / 4), rows - 1);
    const endCol = Math.min(Math.floor((3 * cols) / 4), cols - 1);

    const newNodes = Array(rows).fill().map((_, row) =>
      Array(cols).fill().map((_, col) => ({
        row,
        col,
        isStart: row === startRow && col === startCol,
        isEnd: row === endRow && col === endCol,
      }))
    );
    setNodes(newNodes);
  }, [size]);

  const findNodePosition = useCallback((nodes, nodeType) => {
    for (let row = 0; row < nodes.length; row++) {
      for (let col = 0; col < nodes[row].length; col++) {
        if (nodes[row][col][nodeType]) return [row, col];
      }
    }
    return null;
  }, []);

  const isCellVisited = useCallback((row, col) => {
    if (!walkData?.[currentStep]) return 0;
    return walkData[currentStep].visitCounts[`[${row}, ${col}]`] ||
           walkData[currentStep].visitCounts[JSON.stringify([row, col])] || 0;
  }, [walkData, currentStep]);

const isCellConsidered = useCallback((row, col) => {
  if (!walkData?.[currentStep]) return false;
  const considered = walkData[currentStep].considered;
  // Make sure considered is an array before calling .some()
  return Array.isArray(considered) && considered.some(([r, c]) => r === row && c === col);
}, [walkData, currentStep]);

const isCellInPath = useCallback((row, col) => {
  if (!walkData?.[currentStep]?.path) return false;
  const path = walkData[currentStep].path;
  // Make sure path is an array before calling .some()
  return Array.isArray(path) && path.some(([pathRow, pathCol]) => pathRow === row && pathCol === col);
}, [walkData, currentStep]);

  const isCurrentPosition = useCallback((row, col) => {
    if (!walkData?.[currentStep]) return false;
    const [currentRow, currentCol] = walkData[currentStep].position;
    
    if (isCellInPath(row, col)) return false;

    if (selectedAlgorithm === "jps") {
      const isJumpPoint = walkData[currentStep].jumpPoints?.some(
        ([jr, jc]) => jr === row && jc === col
      );
      return (currentRow === row && currentCol === col) || isJumpPoint;
    }

    return (currentRow === row && currentCol === col);
  }, [walkData, currentStep, selectedAlgorithm, isCellInPath]);

  const getCellBackgroundColor = useCallback(
    (rowIndex, colIndex, isWall) => {
      if (isWall) return "black";
      
      const isPath = isCellInPath(rowIndex, colIndex);
      const visitCount = isCellVisited(rowIndex, colIndex);
      const isConsidered = isCellConsidered(rowIndex, colIndex);
      const isCurrent = isCurrentPosition(rowIndex, colIndex);

      // For weighted algorithms, maintain the altitude colors even for cells in the path
      if (WEIGHTED_ALGORITHMS.includes(selectedAlgorithm)) {
        const altitudes = walkData?.[currentStep]?.altitudes || initialAltitudes;
        if (altitudes) {
          const altitude = altitudes[rowIndex][colIndex];
          const minAltitude = Math.min(...altitudes.flat());
          const maxAltitude = Math.max(...altitudes.flat());
          const normalizedValue = (altitude - minAltitude) / (maxAltitude - minAltitude);
          return `rgba(176, 230, 220, ${normalizedValue * 0.6 + 0.2})`;
        }
      }
      
      // For non-weighted algorithms, apply blue background for path cells
      if (isPath && !WEIGHTED_ALGORITHMS.includes(selectedAlgorithm)) {
        return "rgba(133, 227, 255, 0.6)";
      }
      
      if (visitCount > 0 && selectedAlgorithm === "dfs") return getVisitedColor(visitCount);
      if (selectedAlgorithm === "dfs" && isConsidered) return "rgba(144, 238, 144, 0.3)";
      if (visitCount > 0 && !WEIGHTED_ALGORITHMS.includes(selectedAlgorithm)) return getVisitedColor(visitCount);
      if (isCurrent) return "#FFA07A";

      return "white";
    },
    [selectedAlgorithm, initialAltitudes, walkData, currentStep, isCellVisited, isCellInPath, isCellConsidered, isCurrentPosition]
  );

  const getCellContent = useCallback((rowIndex, colIndex) => {
    const isCurrent = isCurrentPosition(rowIndex, colIndex);
    if (isCurrent && walls && !walls[rowIndex][colIndex]) {
      return <div className="current-cell" />;
    }
    return null;
  }, [isCurrentPosition, walls]);

  const handleCellClick = useCallback((rowIndex, colIndex) => {
    if (isRunning || !walls) return;

    const node = nodes[rowIndex][colIndex];
    if (node.isStart || node.isEnd) return;

    const newWalls = walls.map(row => [...row]);
    newWalls[rowIndex][colIndex] = !newWalls[rowIndex][colIndex];
    onWallsChange(newWalls);

    if (walkData?.length && !isRunning) {
      const startPos = findNodePosition(nodes, "isStart");
      const endPos = findNodePosition(nodes, "isEnd");
      if (startPos && endPos && onNodeDragEnd) {
        onNodeDragEnd(startPos, endPos, newWalls);
      }
    }
  }, [isRunning, nodes, walls, onWallsChange, walkData, findNodePosition, onNodeDragEnd]);

  const handleNodeMouseDown = useCallback((e, nodeType, row, col) => {
    if (isRunning) return;

    e.preventDefault();
    setIsDragging(true);
    setDraggedNode({ type: nodeType, startRow: row, startCol: col });
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
    if (!isDragging || !dragPosition || !draggedNode || !walls) {
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
      const newNodes = nodes.map(row =>
        row.map(node => ({
          ...node,
          isStart: false,
          isEnd: false,
        }))
      );

      newNodes[newRow][newCol][draggedNode.type === "start" ? "isStart" : "isEnd"] = true;

      const otherNodeType = draggedNode.type === "start" ? "isEnd" : "isStart";
      const otherPos = findNodePosition(nodes, otherNodeType);
      if (otherPos) {
        newNodes[otherPos[0]][otherPos[1]][otherNodeType] = true;
      }

      setNodes(newNodes);

      const startPos = draggedNode.type === "start"
        ? [newRow, newCol]
        : findNodePosition(nodes, "isStart");
      const endPos = draggedNode.type === "end"
        ? [newRow, newCol]
        : findNodePosition(nodes, "isEnd");

      onNodePositionsChange?.(startPos, endPos);

      if (walkData?.length && onNodeDragEnd) {
        await onNodeDragEnd(startPos, endPos);
      }
    }

    setIsDragging(false);
    setDraggedNode(null);
    setDragPosition(null);
  }, [isDragging, dragPosition, draggedNode, size, cellSize, walls, nodes, findNodePosition, onNodePositionsChange, onNodeDragEnd, walkData]);

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

  useEffect(() => {
    const startNode = findNodePosition(nodes, "isStart");
    const endNode = findNodePosition(nodes, "isEnd");
    if (startNode && endNode) {
      onNodePositionsChange?.(startNode, endNode);
    }
  }, [nodes, findNodePosition, onNodePositionsChange]);

  return (
    <BaseGrid
      size={size}
      walls={walls}
      onWallsChange={onWallsChange}
      isRunning={isRunning}
      getCellContent={getCellContent}
      getCellBackgroundColor={getCellBackgroundColor}
      onCellClick={handleCellClick}
      className={`${isRunning ? "running" : ""}`}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Integrated path visualisation canvas */}
      {cellSize.width > 0 && walkData && (
        <canvas
          ref={pathCanvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
            pointerEvents: "none",
            zIndex: 10
          }}
          width={GRID_SIZES[size].cols * cellSize.width}
          height={GRID_SIZES[size].rows * cellSize.height}
        />
      )}

      {nodes.map((row, rowIndex) =>
        row.map((node, colIndex) => {
          if (!node.isStart && !node.isEnd) return null;

          const isBeingDragged = isDragging &&
            draggedNode?.startRow === rowIndex &&
            draggedNode?.startCol === colIndex;

          const centerX = Math.min(
            Math.max(
              cellSize.width / 2,
              colIndex * cellSize.width + cellSize.width / 2
            ),
            GRID_SIZES[size].cols * cellSize.width - cellSize.width / 2
          );

          const centerY = Math.min(
            Math.max(
              cellSize.height / 2,
              rowIndex * cellSize.height + cellSize.height / 2
            ),
            GRID_SIZES[size].rows * cellSize.height - cellSize.height / 2
          );

          return (
            <GridNode
              key={`node-${rowIndex}-${colIndex}`}
              isStart={node.isStart}
              isEnd={node.isEnd}
              rowIndex={rowIndex}
              colIndex={colIndex}
              isBeingDragged={isBeingDragged}
              dragPosition={dragPosition}
              centerX={centerX}
              centerY={centerY}
              isRunning={isRunning}
              onNodeMouseDown={handleNodeMouseDown}
            />
          );
        })
      )}
    </BaseGrid>
  );
});

SingleAgentGrid.propTypes = {
  size: PropTypes.oneOf(["small", "medium", "large"]),
  walkData: PropTypes.array,
  currentStep: PropTypes.number,
  onNodePositionsChange: PropTypes.func,
  isRunning: PropTypes.bool,
  isFinished: PropTypes.bool,
  walls: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.bool)),
  onWallsChange: PropTypes.func,
  initialAltitudes: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number)),
  selectedAlgorithm: PropTypes.string,
  onNodeDragEnd: PropTypes.func,
};

GridNode.propTypes = {
  isStart: PropTypes.bool.isRequired,
  isEnd: PropTypes.bool.isRequired,
  rowIndex: PropTypes.number.isRequired,
  colIndex: PropTypes.number.isRequired,
  isBeingDragged: PropTypes.bool.isRequired,
  dragPosition: PropTypes.shape({
    x: PropTypes.number,
    y: PropTypes.number,
  }),
  centerX: PropTypes.number.isRequired,
  centerY: PropTypes.number.isRequired,
  isRunning: PropTypes.bool.isRequired,
  onNodeMouseDown: PropTypes.func.isRequired,
};

export default SingleAgentGrid;