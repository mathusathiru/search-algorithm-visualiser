import React, {useCallback } from "react";

class MazeGenerator {
  constructor(rows, cols) {
    this.rows = rows;
    this.cols = cols;
    this.parent = {};
    this.rank = {};
  }

  find(cell) {
    let cellKey;
    if (typeof cell === "string") {
      cellKey = cell;
    } else if (Array.isArray(cell) && cell.length === 2) {
      cellKey = `${cell[0]},${cell[1]}`;
    } else {
      return "0,0";
    }
    
    if (!this.parent[cellKey]) {
      this.parent[cellKey] = cellKey;
      return cellKey;
    }
    
    if (this.parent[cellKey] !== cellKey) {
      this.parent[cellKey] = this.find(this.parent[cellKey]);
      return this.parent[cellKey];
    }
    
    return cellKey;
  }

  union(cell1, cell2) {
    const root1 = this.find(cell1);
    const root2 = this.find(cell2);

    if (root1 === root2) {
      return false;
    }
    
    if (!this.rank[root1]) this.rank[root1] = 0;
    if (!this.rank[root2]) this.rank[root2] = 0;
    
    if (this.rank[root1] < this.rank[root2]) {
      this.parent[root1] = root2;
    } else {
      this.parent[root2] = root1;
      if (this.rank[root1] === this.rank[root2]) {
        this.rank[root1]++;
      }
    }
    
    return true;
  }

  ensureAgentPositionsAreClear(walls, agentStarts, agentGoals) {
    if (!agentStarts || !agentGoals || agentStarts.length === 0) return;
    
    const positions = [...agentStarts, ...agentGoals];
    for (const pos of positions) {
      if (!pos || pos.length !== 2) continue;
      
      const [row, col] = pos;
      if (row >= 0 && row < this.rows && col >= 0 && col < this.cols) {
        walls[row][col] = false;
        this.clearArea(walls, row, col, 1);
      }
    }
  }

  generateMaze(loopFactor = 0.1, isMultiAgent = false, agentStarts = [], agentGoals = []) {
    let walls = Array(this.rows).fill().map(() => Array(this.cols).fill(true));
    
    if (isMultiAgent && agentStarts?.length > 0) {
      this.ensureAgentPositionsAreClear(walls, agentStarts, agentGoals);
    }
    
    this.parent = {};
    this.rank = {};
    
    // Create grid of passages
    const gridRows = Math.floor(this.rows / 2);
    const gridCols = Math.floor(this.cols / 2);
    
    for (let row = 0; row < gridRows; row++) {
      for (let col = 0; col < gridCols; col++) {
        const cellKey = `${row},${col}`;
        this.parent[cellKey] = cellKey;
        this.rank[cellKey] = 0;
        
        const actualRow = row * 2 + 1;
        const actualCol = col * 2 + 1;
        if (actualRow < this.rows && actualCol < this.cols) {
          walls[actualRow][actualCol] = false;
        }
      }
    }
    
    // Create potential edges between passages
    const edges = [];
    for (let row = 0; row < gridRows; row++) {
      for (let col = 0; col < gridCols; col++) {
        if (col < gridCols - 1) {
          edges.push({
            cell1: [row, col],
            cell2: [row, col + 1],
            wallRow: row * 2 + 1,
            wallCol: col * 2 + 2,
            weight: Math.random()
          });
        }
        
        if (row < gridRows - 1) {
          edges.push({
            cell1: [row, col],
            cell2: [row + 1, col],
            wallRow: row * 2 + 2,
            wallCol: col * 2 + 1,
            weight: Math.random()
          });
        }
      }
    }
    
    // Sort edges by random weight to create randomized spanning tree
    edges.sort((a, b) => a.weight - b.weight);
    
    // Use union-find to create a perfect maze
    const mstEdges = [];
    for (const edge of edges) {
      const { cell1, cell2, wallRow, wallCol } = edge;
      
      if (this.union(cell1, cell2)) {
        if (wallRow < this.rows && wallCol < this.cols) {
          walls[wallRow][wallCol] = false;
          mstEdges.push(edge);
        }
      }
    }
    
    // Add loops to the maze based on loopFactor
    const effectiveLoopFactor = isMultiAgent ? Math.max(loopFactor, 0.3) : loopFactor;
    
    if (effectiveLoopFactor > 0) {
      const remainingWalls = edges.filter(edge => {
        const { wallRow, wallCol } = edge;
        return wallRow < this.rows && wallCol < this.cols && walls[wallRow][wallCol];
      });
      
      // Shuffle remaining walls
      for (let i = remainingWalls.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [remainingWalls[i], remainingWalls[j]] = [remainingWalls[j], remainingWalls[i]];
      }
      
      // Remove some walls to create loops
      const wallsToRemove = Math.floor(remainingWalls.length * effectiveLoopFactor);
      for (let i = 0; i < wallsToRemove && i < remainingWalls.length; i++) {
        const { wallRow, wallCol } = remainingWalls[i];
        walls[wallRow][wallCol] = false;
      }
    }
    
    // Add outer walls
    for (let row = 0; row < this.rows; row++) {
      walls[row][0] = true;
      walls[row][this.cols - 1] = true;
    }
    
    for (let col = 0; col < this.cols; col++) {
      walls[0][col] = true;
      walls[this.rows - 1][col] = true;
    }
    
    // Set start and end positions
    const startRow = Math.floor(this.rows / 4);
    const startCol = Math.floor(this.cols / 4);
    const endRow = Math.floor(3 * this.rows / 4);
    const endCol = Math.floor(3 * this.cols / 4);
    
    walls[startRow][startCol] = false;
    walls[endRow][endCol] = false;
    
    // Only clear areas around start/goal positions for multi-agent scenarios
    if (isMultiAgent) {
      this.clearArea(walls, startRow, startCol, 1);
      this.clearArea(walls, endRow, endCol, 1);
      
      if (agentStarts?.length > 0) {
        this.ensureAgentPaths(walls, agentStarts, agentGoals);
        this.widenCorridors(walls);
        this.clearAgentPositionAreas(walls, agentStarts, agentGoals);
        this.ensureAgentPositionsAreClear(walls, agentStarts, agentGoals);
      }
    }
    
    return walls;
  }
  
  clearArea(walls, centerRow, centerCol, radius) {
    for (let r = Math.max(1, centerRow - radius); r <= Math.min(this.rows - 2, centerRow + radius); r++) {
      for (let c = Math.max(1, centerCol - radius); c <= Math.min(this.cols - 2, centerCol + radius); c++) {
        walls[r][c] = false;
      }
    }
  }
  
  generateRandomMaze(wallDensity = 0.3, isMultiAgent = false, agentStarts = [], agentGoals = []) {
    const effectiveDensity = isMultiAgent ? Math.min(wallDensity, 0.25) : wallDensity;
    
    let walls = Array(this.rows).fill().map(() => Array(this.cols).fill(false));
    
    // Always ensure agent positions are clear if provided (for both single and multi-agent)
    if (agentStarts?.length > 0 && agentGoals?.length > 0) {
      this.ensureAgentPositionsAreClear(walls, agentStarts, agentGoals);
    }
    
    for (let row = 0; row < this.rows; row++) {
      walls[row][0] = true;
      walls[row][this.cols - 1] = true;
    }
    
    for (let col = 0; col < this.cols; col++) {
      walls[0][col] = true;
      walls[this.rows - 1][col] = true;
    }
    
    for (let row = 1; row < this.rows - 1; row++) {
      for (let col = 1; col < this.cols - 1; col++) {
        // Check agent positions for both single and multi-agent modes
        const isAgentPosition = agentStarts?.length > 0 && 
                               (agentStarts.some(pos => pos[0] === row && pos[1] === col) || 
                                agentGoals.some(pos => pos[0] === row && pos[1] === col));
        
        if (!isAgentPosition && Math.random() < effectiveDensity) {
          walls[row][col] = true;
        }
      }
    }
    
    // Only use default positions if none are provided
    if (agentStarts.length === 0 || agentGoals.length === 0) {
      const startRow = Math.floor(this.rows / 4);
      const startCol = Math.floor(this.cols / 4);
      const endRow = Math.floor(3 * this.rows / 4);
      const endCol = Math.floor(3 * this.cols / 4);
      
      walls[startRow][startCol] = false;
      walls[endRow][endCol] = false;
      
      this.clearArea(walls, startRow, startCol, isMultiAgent ? 2 : 1);
      this.clearArea(walls, endRow, endCol, isMultiAgent ? 2 : 1);
    } else {
      // Clear areas around all provided agent positions
      for (const pos of [...agentStarts, ...agentGoals]) {
        this.clearArea(walls, pos[0], pos[1], isMultiAgent ? 2 : 1);
      }
    }
    
    // Final check to ensure all agent positions are clear
    if (agentStarts?.length > 0 && agentGoals?.length > 0) {
      this.ensureAgentPositionsAreClear(walls, agentStarts, agentGoals);
    }
    
    return walls;
  }
    
  ensureAgentPaths(walls, agentStarts, agentGoals) {
    if (!agentStarts || !agentGoals || agentStarts.length === 0) {
      return;
    }
    
    for (let i = 0; i < agentStarts.length; i++) {
      const start = agentStarts[i];
      const goal = agentGoals[i];
      
      if (!this.pathExists(start, goal, walls)) {
        this.createDirectPath(start, goal, walls);
      }
    }
  }
  
  pathExists(start, goal, walls) {
    const queue = [start];
    const visited = new Set([`${start[0]},${start[1]}`]);
    
    while (queue.length > 0) {
      const [row, col] = queue.shift();
      
      if (row === goal[0] && col === goal[1]) {
        return true;
      }
      
      for (const [dr, dc] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
        const newRow = row + dr;
        const newCol = col + dc;
        
        if (newRow >= 0 && newRow < this.rows && 
            newCol >= 0 && newCol < this.cols && 
            !walls[newRow][newCol]) {
          
          const key = `${newRow},${newCol}`;
          if (!visited.has(key)) {
            visited.add(key);
            queue.push([newRow, newCol]);
          }
        }
      }
    }
    
    return false;
  }
  
  createDirectPath(start, goal, walls) {
    let [x0, y0] = start;
    const [x1, y1] = goal;
    
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;
    
    while (x0 !== x1 || y0 !== y1) {
      walls[x0][y0] = false;
      
      for (const [dr, dc] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
        const r = x0 + dr;
        const c = y0 + dc;
        if (r >= 0 && r < this.rows && c >= 0 && c < this.cols) {
          walls[r][c] = false;
        }
      }
      
      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        x0 += sx;
      }
      if (e2 < dx) {
        err += dx;
        y0 += sy;
      }
    }
    
    walls[goal[0]][goal[1]] = false;
  }
  
  widenCorridors(walls) {
    const corridors = [];
    
    for (let row = 1; row < this.rows - 1; row++) {
      for (let col = 1; col < this.cols - 1; col++) {
        if (!walls[row][col]) {
          if (!walls[row][col-1] && !walls[row][col+1] && 
              walls[row-1][col] && walls[row+1][col]) {
            corridors.push([row, col, "horizontal"]);
          }
          else if (!walls[row-1][col] && !walls[row+1][col] && 
                  walls[row][col-1] && walls[row][col+1]) {
            corridors.push([row, col, "vertical"]);
          }
        }
      }
    }
    
    for (const [row, col, type] of corridors) {
      if (Math.random() < 0.5) {
        if (type === "horizontal") {
          if (Math.random() < 0.5 && row > 1) {
            walls[row-1][col] = false; 
          } else if (row < this.rows - 2) {
            walls[row+1][col] = false;
          }
        } else {
          if (Math.random() < 0.5 && col > 1) {
            walls[row][col-1] = false; 
          } else if (col < this.cols - 2) {
            walls[row][col+1] = false;
          }
        }
      }
    }
  }
  
  clearAgentPositionAreas(walls, agentStarts, agentGoals) {
    const positions = [...agentStarts, ...agentGoals];
    
    for (const [row, col] of positions) {
      this.clearArea(walls, row, col, 2);
    }
  }
  
  generateOpenGrid(isMultiAgent = false, agentStarts = [], agentGoals = []) {
    const walls = Array(this.rows).fill().map(() => Array(this.cols).fill(false));
    
    for (let i = 0; i < this.rows; i++) {
      walls[i][0] = true;
      walls[i][this.cols - 1] = true;
    }
    for (let i = 0; i < this.cols; i++) {
      walls[0][i] = true;
      walls[this.rows - 1][i] = true;
    }
    
    const obstacleDensity = isMultiAgent ? 0.05 : 0.1; 
    const obstacleCount = Math.floor(this.rows * this.cols * obstacleDensity);
    
    const agentPositions = new Set();
    // Use agent positions in both multi-agent and single-agent modes if provided
    if (agentStarts?.length > 0 && agentGoals?.length > 0) {
      for (const pos of [...agentStarts, ...agentGoals]) {
        if (pos && pos.length === 2) {
          agentPositions.add(`${pos[0]},${pos[1]}`);
        }
      }
    }
    
    for (let i = 0; i < obstacleCount; i++) {
      const row = 1 + Math.floor(Math.random() * (this.rows - 2));
      const col = 1 + Math.floor(Math.random() * (this.cols - 2));
      
      if (!agentPositions.has(`${row},${col}`)) {
        walls[row][col] = true;
      }
    }
    
    // Clear agent positions for both multi-agent and single-agent modes if positions are provided
    if (agentStarts?.length > 0 && agentGoals?.length > 0) {
      this.ensureAgentPositionsAreClear(walls, agentStarts, agentGoals);
    }
    
    return walls;
  }
  
  validateAgentPlacement(agentStarts, agentGoals) {
    const allPositions = [...agentStarts, ...agentGoals];
    
    for (let i = 0; i < allPositions.length; i++) {
      for (let j = i + 1; j < allPositions.length; j++) {
        const [r1, c1] = allPositions[i];
        const [r2, c2] = allPositions[j];
        
        if (Math.abs(r1 - r2) <= 1 && Math.abs(c1 - c2) <= 1) {
          return false;
        }
      }
    }
    
    return true;
  }
}

const MazeGenerationControls = ({ 
  gridSize, 
  onWallsChange, 
  isMultiAgent = false, 
  agentStarts = [], 
  agentGoals = []   
}) => {
  const generateMaze = useCallback(() => {
    try {
      const generator = new MazeGenerator(gridSize.rows, gridSize.cols);
      const walls = generator.generateMaze(0.1, isMultiAgent, agentStarts, agentGoals);
      onWallsChange(walls);
    } catch (error) {
      console.error("Error generating maze:", error);
    }
  }, [gridSize, isMultiAgent, agentStarts, agentGoals, onWallsChange]);

  const generateRandomMaze = useCallback(() => {
    try {
      const generator = new MazeGenerator(gridSize.rows, gridSize.cols);
      const density = isMultiAgent ? 0.2 : 0.3;
      const walls = generator.generateRandomMaze(density, isMultiAgent, agentStarts, agentGoals);
      onWallsChange(walls);
    } catch (error) {
      console.error("Error generating random maze:", error);
    }
  }, [gridSize, isMultiAgent, agentStarts, agentGoals, onWallsChange]);

  const generateOpenGrid = useCallback(() => {
    try {
      const generator = new MazeGenerator(gridSize.rows, gridSize.cols);
      const walls = generator.generateOpenGrid(isMultiAgent, agentStarts, agentGoals);
      onWallsChange(walls);
    } catch (error) {
      console.error("Error generating open grid:", error);
    }
  }, [gridSize, isMultiAgent, agentStarts, agentGoals, onWallsChange]);

  return (
    <div className="maze-management">
      <div className="maze-controls">
        <button 
          className="button-color maze-button" 
          onClick={generateMaze}
        >
          Generate Maze
        </button>
        
        {isMultiAgent ? (
          <button 
            className="button-color maze-button"
            onClick={generateOpenGrid}
          >
            Open Grid
          </button>
        ) : (
          <button 
            className="button-color maze-button" 
            onClick={generateRandomMaze}
          >
            Random Maze
          </button>
        )}
      </div>
      
      {isMultiAgent && agentStarts.length > 0 && (
        <div className="agent-count">
          {agentStarts.length} agents configured
        </div>
      )}
    </div>
  );
};

export { MazeGenerator, MazeGenerationControls };
export default MazeGenerator;