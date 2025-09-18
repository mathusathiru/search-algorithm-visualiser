import React, { useState, useRef, useEffect, useMemo, useCallback } from "react";
import PropTypes from "prop-types";
import "./css/Grid.css";

export const GRID_SIZES = {
  small: { rows: 20, cols: 50, height: "400px" },
  medium: { rows: 15, cols: 40, height: "400px" },
  large: { rows: 10, cols: 30, height: "400px" },
};

function initializeWalls(size) {
  const { rows, cols } = GRID_SIZES[size];
  return Array(rows).fill().map(() => Array(cols).fill(false));
}

const BaseGridCell = React.memo(({
  rowIndex,
  colIndex,
  backgroundColor,
  isInteractive,
  onCellClick,
  children
}) => (
  <div
    key={`${rowIndex}-${colIndex}`}
    className={`grid-cell ${isInteractive ? "interactive" : "disabled"}`}
    style={{
      backgroundColor,
      opacity: 1,
      animation: "none",
    }}
    onClick={() => onCellClick(rowIndex, colIndex)}
  >
    {children}
  </div>
));

const BaseGrid = ({
  size = "medium",
  walls: externalWalls,
  onWallsChange,
  isRunning = false,
  getCellContent,
  getCellBackgroundColor,
  onCellClick: externalOnCellClick,
  className,
  style,
  children,
  onMouseMove,
  onMouseUp,
  onMouseLeave
}) => {
  const [internalWalls, setInternalWalls] = useState(() => initializeWalls(size));
  const [cellSize, setCellSize] = useState({ width: 0, height: 0 });
  const [isGridReady, setIsGridReady] = useState(false);
  const gridRef = useRef(null);
  const prevSizeRef = useRef(size);

  useEffect(() => {
    if (prevSizeRef.current !== size) {
      const newWalls = initializeWalls(size);
      setInternalWalls(newWalls);
      onWallsChange?.(newWalls);
      prevSizeRef.current = size;
    }
  }, [size, onWallsChange]);

  const activeWalls = useMemo(() => {
    const { rows, cols } = GRID_SIZES[size];
    if (externalWalls?.length === rows && externalWalls[0]?.length === cols) {
      return externalWalls;
    }
    return internalWalls;
  }, [size, externalWalls, internalWalls]);

  const handleCellClick = useCallback((rowIndex, colIndex) => {
    if (isRunning) return;

    if (externalOnCellClick) {
      externalOnCellClick(rowIndex, colIndex);
    } else if (activeWalls) {
      const newWalls = activeWalls.map(row => [...row]);
      newWalls[rowIndex][colIndex] = !newWalls[rowIndex][colIndex];
      onWallsChange?.(newWalls);
    }
  }, [isRunning, externalOnCellClick, activeWalls, onWallsChange]);

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

  const gridStyles = useMemo(() => ({
    gridTemplateColumns: `repeat(${GRID_SIZES[size].cols}, 1fr)`,
    gridTemplateRows: `repeat(${GRID_SIZES[size].rows}, 1fr)`,
    height: GRID_SIZES[size].height,
    backgroundSize: `${100 / GRID_SIZES[size].cols}% ${100 / GRID_SIZES[size].rows}%`,
    ...style
  }), [size, style]);

  if (!activeWalls) {
    return null;
  }

  return (
    <div className="grid-container">
      <div
        ref={gridRef}
        className={`grid-wrapper ${className || ""}`}
        style={gridStyles}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
      >
        {isGridReady && GRID_SIZES[size].rows > 0 && GRID_SIZES[size].cols > 0 && (
          <>
            {Array.from({ length: GRID_SIZES[size].rows }, (_, rowIndex) =>
              Array.from({ length: GRID_SIZES[size].cols }, (_, colIndex) => {
                const isWall = activeWalls[rowIndex]?.[colIndex] || false;
                const backgroundColor = getCellBackgroundColor?.(rowIndex, colIndex, isWall) || 
                  (isWall ? "black" : "white");

                return (
                  <BaseGridCell
                    key={`${rowIndex}-${colIndex}`}
                    rowIndex={rowIndex}
                    colIndex={colIndex}
                    backgroundColor={backgroundColor}
                    isInteractive={!isRunning}
                    onCellClick={handleCellClick}
                  >
                    {getCellContent?.(rowIndex, colIndex)}
                  </BaseGridCell>
                );
              })
            )}
            <div className="grid-overlay">
              {children}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

BaseGrid.propTypes = {
  size: PropTypes.oneOf(["small", "medium", "large"]),
  walls: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.bool)),
  onWallsChange: PropTypes.func,
  isRunning: PropTypes.bool,
  getCellContent: PropTypes.func,
  getCellBackgroundColor: PropTypes.func,
  onCellClick: PropTypes.func,
  className: PropTypes.string,
  style: PropTypes.object,
  children: PropTypes.node,
  onMouseMove: PropTypes.func,
  onMouseUp: PropTypes.func,
  onMouseLeave: PropTypes.func
};

BaseGridCell.propTypes = {
  rowIndex: PropTypes.number.isRequired,
  colIndex: PropTypes.number.isRequired,
  backgroundColor: PropTypes.string.isRequired,
  isInteractive: PropTypes.bool.isRequired,
  onCellClick: PropTypes.func.isRequired,
  children: PropTypes.node
};

export default BaseGrid;