import React, {
  useState,
  useRef,
  useEffect,
  useMemo,
  useCallback,
} from "react";
import PropTypes from "prop-types";
import PathVisualisationOverlay from "./PathVisualisationOverlay.js";
import { getVisitedColor } from "./ColourConverter";
import "./css/Grid.css";

const GRID_SIZES = {
  small: { rows: 20, cols: 50, height: "400px" },
  medium: { rows: 15, cols: 40, height: "400px" },
  large: { rows: 10, cols: 30, height: "400px" },
};

const WEIGHTED_ALGORITHMS = ["dijkstra", "astar"];

const GridCell = React.memo(
  ({
    node,
    rowIndex,
    colIndex,
    backgroundColor,
    isRunning,
    isCurrent,
    isWall,
    onCellClick,
  }) => (
    <div
      key={`${rowIndex}-${colIndex}`}
      className={`
      grid-cell 
      ${!isRunning && !node.isStart && !node.isEnd ? "interactive" : "disabled"}
    `}
      style={{
        backgroundColor,
        opacity: 1,
        animation: "none",
      }}
      onClick={() => onCellClick(rowIndex, colIndex)}
    >
      {isCurrent && !node.isStart && !node.isEnd && !isWall && (
        <div className="current-cell" />
      )}
    </div>
  )
);

const GridNode = React.memo(
  ({
    node,
    rowIndex,
    colIndex,
    isBeingDragged,
    dragPosition,
    centerX,
    centerY,
    isRunning,
    onNodeMouseDown,
  }) => (
    <div
      key={`node-${rowIndex}-${colIndex}`}
      className={`
      grid-node 
      ${node.isStart ? "start" : "end"}
      ${isRunning ? "disabled" : "draggable"}
    `}
      style={{
        left: isBeingDragged ? `${dragPosition.x}px` : `${centerX}px`,
        top: isBeingDragged ? `${dragPosition.y}px` : `${centerY}px`,
        transition: isBeingDragged ? "none" : "all 0.05s ease-out",
      }}
      onMouseDown={(e) =>
        onNodeMouseDown(e, node.isStart ? "start" : "end", rowIndex, colIndex)
      }
    />
  )
);

const Grid = ({
  size = "medium",
  walkData,
  currentStep,
  onNodePositionsChange,
  isRunning = false,
  isFinished = false,
  walls,
  onWallsChange,
  initialAltitudes = null,
  selectedAlgorithm = "",
  onNodeDragEnd,
}) => {
  const initializeNodes = (currentSize) => {
    const { rows, cols } = GRID_SIZES[currentSize];
    const startRow = Math.min(Math.floor(rows / 4), rows - 1);
    const startCol = Math.min(Math.floor(cols / 4), cols - 1);
    const endRow = Math.min(Math.floor((3 * rows) / 4), rows - 1);
    const endCol = Math.min(Math.floor((3 * cols) / 4), cols - 1);

    return Array(rows)
      .fill()
      .map((_, row) =>
        Array(cols)
          .fill()
          .map((_, col) => ({
            row,
            col,
            isStart: row === startRow && col === startCol,
            isEnd: row === endRow && col === endCol,
          }))
      );
  };

  const [cellSize, setCellSize] = useState({ width: 0, height: 0 });
  const [isGridReady, setIsGridReady] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [draggedNode, setDraggedNode] = useState(null);
  const [dragPosition, setDragPosition] = useState(null);
  const [nodes, setNodes] = useState(() => initializeNodes(size));

  const gridRef = useRef(null);
  const prevSizeRef = useRef(size);

  const memoizedInitializeNodes = useCallback((currentSize) => {
    return initializeNodes(currentSize);
  }, []);

  const findNodePosition = useCallback((nodes, nodeType) => {
    for (let row = 0; row < nodes.length; row++) {
      for (let col = 0; col < nodes[row].length; col++) {
        if (nodes[row][col][nodeType]) return [row, col];
      }
    }
    return null;
  }, []);

  const isCellVisited = useCallback(
    (row, col) => {
      if (!walkData?.[currentStep]) return 0;
      return (
        walkData[currentStep].visitCounts[`[${row}, ${col}]`] ||
        walkData[currentStep].visitCounts[JSON.stringify([row, col])] ||
        0
      );
    },
    [walkData, currentStep]
  );

  const isCurrentPosition = useCallback(
    (row, col) => {
      if (!walkData?.[currentStep]) return false;
      const [currentRow, currentCol] = walkData[currentStep].position;
      const path = walkData[currentStep].path;
      const lastPathNode = path?.slice(-1)[0];

      if (selectedAlgorithm === "jps") {
        const isJumpPoint = walkData[currentStep].jumpPoints?.some(
          ([jr, jc]) => jr === row && jc === col
        );
        return (
          (currentRow === row && currentCol === col) ||
          (lastPathNode &&
            lastPathNode[0] === row &&
            lastPathNode[1] === col) ||
          isJumpPoint
        );
      }

      return (
        (currentRow === row && currentCol === col) ||
        (lastPathNode && lastPathNode[0] === row && lastPathNode[1] === col)
      );
    },
    [walkData, currentStep, selectedAlgorithm]
  );

  const isCellConsidered = useCallback(
    (row, col) => {
      if (!walkData?.[currentStep]) return false;
      return walkData[currentStep].considered?.some(
        ([r, c]) => r === row && c === col
      );
    },
    [walkData, currentStep]
  );

  const isCellInPath = useCallback(
    (row, col) => {
      if (!walkData?.[currentStep]?.path) return false;
      return walkData[currentStep].path.some(
        ([pathRow, pathCol]) => pathRow === row && pathCol === col
      );
    },
    [walkData, currentStep]
  );

  const getCellBackgroundColor = useCallback(
    (
      rowIndex,
      colIndex,
      { isWall, isPath, visitCount, isConsidered, step }
    ) => {
      if (isWall) return "black";
      if (isPath && !WEIGHTED_ALGORITHMS.includes(selectedAlgorithm))
        return "#85E3FF";
      if (visitCount > 0 && selectedAlgorithm === "dfs")
        return getVisitedColor(visitCount);
      if (selectedAlgorithm === "dfs" && isConsidered)
        return "rgba(144, 238, 144, 0.3)";
      if (visitCount > 0 && !WEIGHTED_ALGORITHMS.includes(selectedAlgorithm))
        return getVisitedColor(visitCount);

      if (WEIGHTED_ALGORITHMS.includes(selectedAlgorithm)) {
        const altitudes = step?.altitudes || initialAltitudes;
        if (altitudes) {
          const altitude = altitudes[rowIndex][colIndex];
          const minAltitude = Math.min(...altitudes.flat());
          const maxAltitude = Math.max(...altitudes.flat());
          const normalizedValue =
            (altitude - minAltitude) / (maxAltitude - minAltitude);
          return `rgba(176, 230, 220, ${normalizedValue * 0.6 + 0.2})`;
        }
      }

      return "white";
    },
    [selectedAlgorithm, initialAltitudes]
  );

  const handleCellClick = useCallback(
    (rowIndex, colIndex) => {
      if (isRunning) return;

      const node = nodes[rowIndex][colIndex];
      if (node.isStart || node.isEnd) return;

      const newWalls = walls.map((row) => [...row]);
      newWalls[rowIndex][colIndex] = !newWalls[rowIndex][colIndex];
      onWallsChange(newWalls);

      if (walkData?.length && !isRunning) {
        const startPos = findNodePosition(nodes, "isStart");
        const endPos = findNodePosition(nodes, "isEnd");
        if (startPos && endPos && onNodeDragEnd) {
          onNodeDragEnd(startPos, endPos, newWalls);
        }
      }
    },
    [
      isRunning,
      nodes,
      walls,
      onWallsChange,
      walkData,
      findNodePosition,
      onNodeDragEnd,
    ]
  );

  const handleNodeMouseDown = useCallback(
    (e, nodeType, row, col) => {
      if (isRunning) return;

      e.preventDefault();
      if (gridRef.current) {
        setIsDragging(true);
        setDraggedNode({ type: nodeType, startRow: row, startCol: col });
        setDragPosition({
          x: col * cellSize.width + cellSize.width / 2,
          y: row * cellSize.height + cellSize.height / 2,
        });
      }
    },
    [isRunning, isFinished, cellSize]
  );

  const handleMouseMove = useCallback(
    (e) => {
      if (!isDragging || !gridRef.current || !dragPosition) return;

      const rect = gridRef.current.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      setDragPosition({
        x: Math.max(0, Math.min(rect.width, x)),
        y: Math.max(0, Math.min(rect.height, y)),
      });
    },
    [isDragging, dragPosition]
  );

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
      const newNodes = nodes.map((row) =>
        row.map((node) => ({
          ...node,
          isStart: false,
          isEnd: false,
        }))
      );

      newNodes[newRow][newCol][
        draggedNode.type === "start" ? "isStart" : "isEnd"
      ] = true;

      const otherNodeType = draggedNode.type === "start" ? "isEnd" : "isStart";
      const otherPos = findNodePosition(nodes, otherNodeType);
      if (otherPos) {
        newNodes[otherPos[0]][otherPos[1]][otherNodeType] = true;
      }

      setNodes(newNodes);

      const startPos =
        draggedNode.type === "start"
          ? [newRow, newCol]
          : findNodePosition(nodes, "isStart");
      const endPos =
        draggedNode.type === "end"
          ? [newRow, newCol]
          : findNodePosition(nodes, "isEnd");

      onNodePositionsChange?.(startPos, endPos);

      if (onNodeDragEnd) {
        await onNodeDragEnd(startPos, endPos);
      }
    }

    setIsDragging(false);
    setDraggedNode(null);
    setDragPosition(null);
  }, [
    isDragging,
    dragPosition,
    draggedNode,
    size,
    cellSize,
    walls,
    nodes,
    findNodePosition,
    onNodePositionsChange,
    onNodeDragEnd,
  ]);

  const gridStyles = useMemo(
    () => ({
      gridTemplateColumns: `repeat(${GRID_SIZES[size].cols}, 1fr)`,
      gridTemplateRows: `repeat(${GRID_SIZES[size].rows}, 1fr)`,
      height: GRID_SIZES[size].height,
      backgroundSize: `${100 / GRID_SIZES[size].cols}% ${
        100 / GRID_SIZES[size].rows
      }%`,
    }),
    [size]
  );

  useEffect(() => {
    const updateCellSize = () => {
      if (gridRef.current) {
        const rect = gridRef.current.getBoundingClientRect();
        setCellSize({
          width: rect.width / GRID_SIZES[size].cols,
          height: rect.height / GRID_SIZES[size].rows,
        });
        setIsGridReady(true);
      }
    };

    updateCellSize();

    const resizeObserver = new ResizeObserver(updateCellSize);
    if (gridRef.current) {
      resizeObserver.observe(gridRef.current);
    }

    return () => {
      resizeObserver.disconnect();
      setIsGridReady(false);
    };
  }, [size]);

  useEffect(() => {
    if (prevSizeRef.current !== size) {
      const prevGrid = GRID_SIZES[prevSizeRef.current];
      const newGrid = GRID_SIZES[size];

      const startPos = findNodePosition(nodes, "isStart");
      const endPos = findNodePosition(nodes, "isEnd");

      if (startPos && endPos) {
        const newStartRow = Math.min(
          Math.floor((startPos[0] * newGrid.rows) / prevGrid.rows),
          newGrid.rows - 1
        );
        const newStartCol = Math.min(
          Math.floor((startPos[1] * newGrid.cols) / prevGrid.cols),
          newGrid.cols - 1
        );
        const newEndRow = Math.min(
          Math.floor((endPos[0] * newGrid.rows) / prevGrid.rows),
          newGrid.rows - 1
        );
        const newEndCol = Math.min(
          Math.floor((endPos[1] * newGrid.cols) / prevGrid.cols),
          newGrid.cols - 1
        );

        const newNodes = initializeNodes(size);
        setNodes(newNodes);
        onNodePositionsChange?.(
          [newStartRow, newStartCol],
          [newEndRow, newEndCol]
        );
      }

      prevSizeRef.current = size;
    }
  }, [size, onNodePositionsChange, nodes, findNodePosition, initializeNodes]);

  useEffect(() => {
    if (isGridReady) {
      const startNode = findNodePosition(nodes, "isStart");
      const endNode = findNodePosition(nodes, "isEnd");
      if (startNode && endNode) {
        onNodePositionsChange?.(startNode, endNode);
      }
    }
  }, [isGridReady, nodes, findNodePosition, onNodePositionsChange]);

  return (
    <div className="grid-container">
      <div
        ref={gridRef}
        className="grid-wrapper"
        style={gridStyles}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {nodes.map((row, rowIndex) =>
          row.map((node, colIndex) => {
            const visitCount = isCellVisited(rowIndex, colIndex);
            const isWall = walls[rowIndex][colIndex];
            const isCurrent = isCurrentPosition(rowIndex, colIndex);
            const isConsidered = isCellConsidered(rowIndex, colIndex);
            const isPath = isCellInPath(rowIndex, colIndex);

            const backgroundColor = getCellBackgroundColor(rowIndex, colIndex, {
              isWall,
              isPath,
              visitCount,
              isConsidered,
              step: walkData?.[currentStep],
            });

            return (
              <GridCell
                key={`${rowIndex}-${colIndex}`}
                node={node}
                rowIndex={rowIndex}
                colIndex={colIndex}
                backgroundColor={backgroundColor}
                isRunning={isRunning}
                isCurrent={isCurrent}
                isWall={isWall}
                onCellClick={handleCellClick}
              />
            );
          })
        )}

        {isGridReady && cellSize.width > 0 && (
          <>
            {walkData && WEIGHTED_ALGORITHMS.includes(selectedAlgorithm) && (
              <PathVisualisationOverlay
                walkData={walkData}
                currentStep={currentStep}
                cellSize={cellSize}
                gridSize={GRID_SIZES[size]}
                selectedAlgorithm={selectedAlgorithm}
              />
            )}

            {nodes.map((row, rowIndex) =>
              row.map((node, colIndex) => {
                if (!node.isStart && !node.isEnd) return null;

                const isBeingDragged =
                  isDragging &&
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
                    node={node}
                    rowIndex={rowIndex}
                    colIndex={colIndex}
                    isBeingDragged={isBeingDragged}
                    dragPosition={dragPosition}
                    cellSize={cellSize}
                    centerX={centerX}
                    centerY={centerY}
                    isRunning={isRunning}
                    isFinished={isFinished}
                    onNodeMouseDown={handleNodeMouseDown}
                  />
                );
              })
            )}
          </>
        )}
      </div>
    </div>
  );
};

Grid.propTypes = {
  size: PropTypes.oneOf(["small", "medium", "large"]),
  walkData: PropTypes.array,
  currentStep: PropTypes.number,
  onNodePositionsChange: PropTypes.func,
  isRunning: PropTypes.bool,
  isFinished: PropTypes.bool,
  walls: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.bool)).isRequired,
  onWallsChange: PropTypes.func.isRequired,
  initialAltitudes: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number)),
  selectedAlgorithm: PropTypes.string,
  onNodeDragEnd: PropTypes.func,
};

GridCell.propTypes = {
  node: PropTypes.shape({
    row: PropTypes.number.isRequired,
    col: PropTypes.number.isRequired,
    isStart: PropTypes.bool.isRequired,
    isEnd: PropTypes.bool.isRequired,
  }).isRequired,
  rowIndex: PropTypes.number.isRequired,
  colIndex: PropTypes.number.isRequired,
  backgroundColor: PropTypes.string.isRequired,
  isRunning: PropTypes.bool.isRequired,
  isCurrent: PropTypes.bool.isRequired,
  isWall: PropTypes.bool.isRequired,
  onCellClick: PropTypes.func.isRequired,
};

GridNode.propTypes = {
  node: PropTypes.shape({
    row: PropTypes.number.isRequired,
    col: PropTypes.number.isRequired,
    isStart: PropTypes.bool.isRequired,
    isEnd: PropTypes.bool.isRequired,
  }).isRequired,
  rowIndex: PropTypes.number.isRequired,
  colIndex: PropTypes.number.isRequired,
  isBeingDragged: PropTypes.bool.isRequired,
  dragPosition: PropTypes.shape({
    x: PropTypes.number.isRequired,
    y: PropTypes.number.isRequired,
  }),
  cellSize: PropTypes.shape({
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired,
  }).isRequired,
  centerX: PropTypes.number.isRequired,
  centerY: PropTypes.number.isRequired,
  isRunning: PropTypes.bool.isRequired,
  isFinished: PropTypes.bool.isRequired,
  onNodeMouseDown: PropTypes.func.isRequired,
};

export default Grid;
