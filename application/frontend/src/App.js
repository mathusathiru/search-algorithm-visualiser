import React, { useState, useEffect, useRef, useCallback } from "react";
import "./css/App.css";
import "./css/font.css";
import ToggleButton from "./ToggleButton";
import KeyLegend from "./KeyLegend";
import AlgorithmInfoBox from "./AlgorithmInfo";
import PseudocodeDisplay from "./Pseudocode";
import MultiAgentGrid from "./MultiGrid";
import SingleAgentGrid from "./SingleAgentGrid";
import { MazeGenerator } from "./MazeGenerator";

// Constants
const GRID_SIZES = {
  small: { rows: 20, cols: 50, height: "400px" },
  medium: { rows: 15, cols: 40, height: "400px" },
  large: { rows: 10, cols: 30, height: "400px" },
};

const SINGLE_AGENT_ALGORITHMS = [
  { value: "bfs", label: "Breadth First Search" },
  { value: "dfs", label: "Depth First Search" },
  { value: "dijkstra", label: "Dijkstra's"},
  { value: "astar", label: "A* Search" },
  { value: "jps", label: "Jump Point Search" },
  { value: "gbfs", label: "Greedy Best First Search" },
  { value: "randomwalk", label: "Random Walk" },
  { value: "biasedrandomwalk", label: "Biased Random Walk" },
];

const MULTI_AGENT_ALGORITHMS = [
  { value: "cbs", label: "Conflict-Based Search" },
  { value: "icts", label: "Increasing Cost Tree Search" },
  { value: "mstar", label: "M* Search" },
  { value: "pushandrotate", label: "Push and Rotate" }
];

const GRID_SIZE_OPTIONS = [
  { value: "small", label: "Small" },
  { value: "medium", label: "Medium" },
  { value: "large", label: "Large" },
];

const ALGORITHM_ENDPOINTS = {
  randomwalk: "/api/randomwalk",
  biasedrandomwalk: "/api/biasedrandomwalk",
  bfs: "/api/bfs",
  dfs: "/api/dfs",
  dijkstra: "/api/dijkstra",
  astar: "/api/astar",
  gbfs: "/api/gbfs",
  jps: "/api/jps",
  cbs: "/api/cbs",
  icts: "/api/icts",
  mstar: "/api/mstar",
  pushandrotate: "/api/pushandrotate"
};

// Helpers
function initialiseWalls(size) {
  const { rows, cols } = GRID_SIZES[size];
  return Array(rows).fill().map(() => Array(cols).fill(false));
}

function formatTime(time) {
  const minutes = Math.floor(time / 60000);
  const seconds = Math.floor((time % 60000) / 1000);
  const milliseconds = Math.floor((time % 1000) / 10);
  return `${minutes}:${seconds.toString().padStart(2, "0")}:${milliseconds.toString().padStart(2, "0")}`;
}

// Dropdown arrow
const CustomArrow = () => (
  <svg width="10" height="10" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M2 4L6 8L10 4" stroke="black" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const App = () => {
  // Refs
  const singleAgentGridRef = useRef(null);
  const multiAgentGridRef = useRef(null);
  
  // General states
  const [speed, setSpeed] = useState(50);
  const [isRunning, setIsRunning] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [time, setTime] = useState(0);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState("");
  const [gridSize, setGridSize] = useState("medium");
  const [walkSteps, setWalkSteps] = useState(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [isFinished, setIsFinished] = useState(false);
  const [walls, setWalls] = useState(() => initialiseWalls(gridSize));
  const [agentMode, setAgentMode] = useState("left");
  
  // Single agent states
  const [startPosition, setStartPosition] = useState(null);
  const [endPosition, setEndPosition] = useState(null);
  const [displayAltitudeColorScheme, setDisplayAltitudeColorScheme] = useState(false);
  const [initialAltitudes, setInitialAltitudes] = useState(null);
  const [singleAgentHistory, setSingleAgentHistory] = useState([]);
  const [singleAgentHistoryIndex, setSingleAgentHistoryIndex] = useState(-1);
  
  // Multi agent states
  const [agentStarts, setAgentStarts] = useState([]);
  const [agentGoals, setAgentGoals] = useState([]);
  const [agentList, setAgentList] = useState([]);
  const [multiAgentHistory, setMultiAgentHistory] = useState([]);
  const [multiAgentHistoryIndex, setMultiAgentHistoryIndex] = useState(-1);

  // Initialise multi-agent grid when switching to multi-agent mode
  useEffect(() => {
    if (agentMode === "right" && agentList.length === 0) {
      if (multiAgentGridRef.current) {
        const initialAgents = multiAgentGridRef.current.initialiseAgents();
        setAgentList(initialAgents);
        setAgentStarts(initialAgents.map(a => a.start));
        setAgentGoals(initialAgents.map(a => a.goal));
        
        const initialPositions = {
          starts: initialAgents.map(a => [...a.start]),
          goals: initialAgents.map(a => [...a.goal])
        };
        setMultiAgentHistory([initialPositions]);
        setMultiAgentHistoryIndex(0);
      }
    }
  }, [agentMode, agentList.length]);

  // Reset selected algorithm when changing agent mode
  useEffect(() => {
    setSelectedAlgorithm("");
    
    if (agentMode === "left") {
      setAgentList([]);
      setAgentStarts([]);
      setAgentGoals([]);
    }
  }, [agentMode]);

  // Initialise walls when grid size changes
  useEffect((i) => {
    const { rows, cols } = GRID_SIZES[gridSize];
    setWalls(Array(rows).fill().map(() => Array(cols).fill(false)));
  }, [gridSize]);

  // Timer effect
  useEffect(() => {
    let interval;
    if (isRunning && !isPaused) {
      const startTime = Date.now() - time;
      interval = setInterval(() => {
        setTime(Date.now() - startTime);
      }, 10);
    }
    return () => clearInterval(interval);
  }, [isRunning, isPaused, time]);

  // Step animation effect
  useEffect(() => {
    let stepInterval;
    if (isRunning && !isPaused && walkSteps) {
      const intervalDelay = Math.floor(2000 * Math.pow(0.005, speed / 100));
      
      stepInterval = setInterval(() => {
        setCurrentStepIndex((prevStep) => {
          if (prevStep >= walkSteps.length - 1) {
            setIsRunning(false);
            setIsFinished(true);
            return prevStep;
          }
          return prevStep + 1;
        });
      }, intervalDelay);
    }
    return () => clearInterval(stepInterval);
  }, [isRunning, isPaused, walkSteps, speed]);

  // Reset when algorithm changes and visualisation is finished
  useEffect(() => {
    if (isFinished) {
      handleReset();
    }
  }, [selectedAlgorithm]);
  
  // initialise history for single agent
  useEffect(() => {
    if (startPosition && endPosition && singleAgentHistory.length === 0) {
      setSingleAgentHistory([{ start: [...startPosition], end: [...endPosition] }]);
      setSingleAgentHistoryIndex(0);
    }
  }, [startPosition, endPosition, singleAgentHistory.length]);

  // Handler functions
  const handleAgentModeChange = (mode) => {
    setAgentMode(mode);
  
    // Reset visualisation data regardless of which mode we"re switching to
    setWalkSteps(null);
    setCurrentStepIndex(0);
    setIsFinished(false);
    setTime(0);
      
    if (mode === "right") {
      // Multi-agent specific resets
      setWalls(initialiseWalls(gridSize));
    } else {
      // Single-agent specific resets
      setAgentList([]);
      setAgentStarts([]);
      setAgentGoals([]);
      // Reset walls for single agent mode too
      setWalls(initialiseWalls(gridSize));
    }
  };

  const handleNodeDragEnd = async (start, end, newWalls = walls) => {
    if (walkSteps?.length && !isRunning) {
      const currentGridSize = GRID_SIZES[gridSize];
      const requestData = {
        start: start,
        end: end,
        gridSize: currentGridSize,
        walls: newWalls,
        initialAltitudes: initialAltitudes,
      };

      try {
        const endpoint = ALGORITHM_ENDPOINTS[selectedAlgorithm];
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestData),
        });

        const data = await response.json();
        if (data.steps && Array.isArray(data.steps)) {
          setCurrentStepIndex(data.steps.length - 1);
          setWalkSteps(data.steps);
        }
      } catch (error) {
      }
    }
  };

  const handleMultiNodeDragEnd = async (starts, goals, newWalls = walls) => {
    if (walkSteps?.length && !isRunning) {
      const currentGridSize = GRID_SIZES[gridSize];
      const requestData = {
        starts: starts,
        goals: goals,
        gridSize: currentGridSize,
        walls: newWalls,
      };

      try {
        const endpoint = ALGORITHM_ENDPOINTS[selectedAlgorithm];
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestData),
        });

        const data = await response.json();
        if (data.steps && Array.isArray(data.steps)) {
          setCurrentStepIndex(data.steps.length - 1);
          setWalkSteps(data.steps);
        }
      } catch (error) {
      }
    }
  };

  const getCurrentAlgorithmOptions = () => {
    return agentMode === "left" ? SINGLE_AGENT_ALGORITHMS : MULTI_AGENT_ALGORITHMS;
  };

  const handleClear = () => {
    setWalls(initialiseWalls(gridSize));
    setWalkSteps(null);
    setCurrentStepIndex(0);
    setTime(0);
    setIsFinished(false);
  };

  const handleNodePositionsChange = (start, end) => {
    if (JSON.stringify(start) === JSON.stringify(startPosition) && 
        JSON.stringify(end) === JSON.stringify(endPosition)) {
      return;
    }
    
    if (singleAgentHistoryIndex < singleAgentHistory.length - 1) {
      const newHistory = singleAgentHistory.slice(0, singleAgentHistoryIndex + 1);
      newHistory.push({ start: [...start], end: [...end] });
      setSingleAgentHistory(newHistory);
      setSingleAgentHistoryIndex(singleAgentHistoryIndex + 1);
    } else {
      setSingleAgentHistory([...singleAgentHistory, { start: [...start], end: [...end] }]);
      setSingleAgentHistoryIndex(singleAgentHistory.length);
    }
    
    setStartPosition(start);
    setEndPosition(end);
  };
  
  const handleSingleAgentUndo = useCallback(() => {
    if (isRunning || singleAgentHistoryIndex <= 0) return;
    
    const newIndex = singleAgentHistoryIndex - 1;
    setSingleAgentHistoryIndex(newIndex);
    
    const { start, end } = singleAgentHistory[newIndex];
    setStartPosition(start);
    setEndPosition(end);
    
    if (singleAgentGridRef.current) {
      singleAgentGridRef.current.updateNodePositions(start, end);
    }
    
    if (walkSteps?.length && !isRunning) {
      handleNodeDragEnd(start, end);
    }
  }, [singleAgentHistoryIndex, singleAgentHistory, isRunning, walkSteps]);
  
  const handleSingleAgentRedo = useCallback(() => {
    if (isRunning || singleAgentHistoryIndex >= singleAgentHistory.length - 1) return;
    
    const newIndex = singleAgentHistoryIndex + 1;
    setSingleAgentHistoryIndex(newIndex);
    
    const { start, end } = singleAgentHistory[newIndex];
    setStartPosition(start);
    setEndPosition(end);
    
    if (singleAgentGridRef.current) {
      singleAgentGridRef.current.updateNodePositions(start, end);
    }
    
    if (walkSteps?.length && !isRunning) {
      handleNodeDragEnd(start, end);
    }
  }, [singleAgentHistoryIndex, singleAgentHistory, isRunning, walkSteps]);
  
  const handleMultiAgentUndo = useCallback(() => {
    if (isRunning || multiAgentHistoryIndex <= 0) return;
    
    const newIndex = multiAgentHistoryIndex - 1;
    setMultiAgentHistoryIndex(newIndex);
    
    const { starts, goals } = multiAgentHistory[newIndex];
    
    const startsCopy = starts.map(start => [...start]);
    const goalsCopy = goals.map(goal => [...goal]);
    
    setAgentStarts(startsCopy);
    setAgentGoals(goalsCopy);
    
    const newAgentList = startsCopy.map((start, index) => ({
      id: index,
      start: [...start],
      goal: goalsCopy[index] ? [...goalsCopy[index]] : [...start]
    }));
    setAgentList(newAgentList);
    
    if (multiAgentGridRef.current) {
      multiAgentGridRef.current.updateAgentPositions(startsCopy, goalsCopy);
    }
    
    if (walkSteps?.length && !isRunning) {
      handleMultiNodeDragEnd(startsCopy, goalsCopy);
    }
  }, [multiAgentHistoryIndex, multiAgentHistory, isRunning, walkSteps]);
  
  const handleMultiAgentRedo = useCallback(() => {
    if (isRunning || multiAgentHistoryIndex >= multiAgentHistory.length - 1) return;
    
    const newIndex = multiAgentHistoryIndex + 1;
    setMultiAgentHistoryIndex(newIndex);
    
    const { starts, goals } = multiAgentHistory[newIndex];
    
    const startsCopy = starts.map(start => [...start]);
    const goalsCopy = goals.map(goal => [...goal]);
    
    setAgentStarts(startsCopy);
    setAgentGoals(goalsCopy);
    
    const newAgentList = startsCopy.map((start, index) => ({
      id: index,
      start: [...start],
      goal: goalsCopy[index] ? [...goalsCopy[index]] : [...start]
    }));
    setAgentList(newAgentList);
    
    if (multiAgentGridRef.current) {
      multiAgentGridRef.current.updateAgentPositions(startsCopy, goalsCopy);
    }
    
    if (walkSteps?.length && !isRunning) {
      handleMultiNodeDragEnd(startsCopy, goalsCopy);
    }
  }, [multiAgentHistoryIndex, multiAgentHistory, isRunning, walkSteps]);

    // Keyboard shortcut effect for undo/redo
    useEffect(() => {
      const handleKeyDown = (e) => {
        if (isRunning) return;
        
        if (e.ctrlKey && e.key === "z") {
          e.preventDefault();
          agentMode === "left" ? handleSingleAgentUndo() : handleMultiAgentUndo();
        }
        else if (e.ctrlKey && e.key === "y") {
          e.preventDefault();
          agentMode === "left" ? handleSingleAgentRedo() : handleMultiAgentRedo();
        }
      };
      
      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }, [
      isRunning, 
      agentMode,
      handleSingleAgentUndo,
      handleSingleAgentRedo,
      handleMultiAgentUndo,
      handleMultiAgentRedo
    ]);

  const onAgentPositionsChange = useCallback((starts, goals) => {
    if (JSON.stringify(starts) === JSON.stringify(agentStarts) && 
        JSON.stringify(goals) === JSON.stringify(agentGoals)) {
      return;
    }
    
    const startsDeepCopy = starts.map(start => [...start]);
    const goalsDeepCopy = goals.map(goal => [...goal]);
    
    const positionsForHistory = { 
      starts: startsDeepCopy, 
      goals: goalsDeepCopy 
    };
    
    if (multiAgentHistoryIndex < multiAgentHistory.length - 1) {
      const newHistory = multiAgentHistory.slice(0, multiAgentHistoryIndex + 1);
      newHistory.push(positionsForHistory);
      setMultiAgentHistory(newHistory);
      setMultiAgentHistoryIndex(multiAgentHistoryIndex + 1);
    } else {
      setMultiAgentHistory([...multiAgentHistory, positionsForHistory]);
      setMultiAgentHistoryIndex(multiAgentHistory.length);
    }
    
    setAgentStarts(startsDeepCopy);
    setAgentGoals(goalsDeepCopy);
    
    const newAgentList = startsDeepCopy.map((start, index) => ({
      id: index,
      start: start,
      goal: goalsDeepCopy[index] || start
    }));
    
    setAgentList(newAgentList);
  }, [agentStarts, agentGoals, multiAgentHistory, multiAgentHistoryIndex]);

  const handleSpeedChange = (e) => {
    setSpeed(e.target.value);
  };

  const handlePauseResume = () => {
    setIsPaused(!isPaused);
  };

  const handleStop = () => {
    setIsRunning(false);
    setIsPaused(false);
    setTime(0);
    setCurrentStepIndex(0);
    setIsFinished(false);
    setWalkSteps(null);
  };

  const handleReset = () => {
    setIsRunning(false);
    setIsPaused(false);
    setTime(0);
    setCurrentStepIndex(0);
    setIsFinished(false);
    setWalkSteps(null);
  };

  const handleAlgorithmChange = (e) => {
    const algorithm = e.target.value;
    setSelectedAlgorithm(algorithm);

    const { rows, cols } = GRID_SIZES[gridSize];

    if (["dijkstra", "astar"].includes(algorithm)) {
      setInitialAltitudes(
        Array(rows).fill().map(() => Array(cols).fill().map(() => Math.floor(Math.random() * 9) + 1))
      );
      setDisplayAltitudeColorScheme(true);
    } else {
      setInitialAltitudes(null);
      setDisplayAltitudeColorScheme(false);
    }
  };

  const handleGridSizeChange = (e) => {
    const newSize = e.target.value;
    const oldSize = gridSize;

    // Always clear any existing path data when grid size changes
    setWalkSteps(null);
    setCurrentStepIndex(0);
    setIsFinished(false);
        
    // If in multi-agent mode and we have agents, delegate positioning to MultiGrid
    if (agentMode === "right" && agentStarts.length > 0 && multiAgentGridRef.current) {
      const { newStarts, newGoals } = multiAgentGridRef.current.handleGridSizeChange(oldSize, newSize);
      
      setAgentStarts(newStarts);
      setAgentGoals(newGoals);
      
      // Update agent list with new positions
      const newAgentList = newStarts.map((start, index) => ({
        id: index,
        start: [...start],
        goal: [...newGoals[index]]
      }));
      
      setAgentList(newAgentList);
      
      // Add to history
      const positionsForHistory = {
        starts: newStarts.map(start => [...start]),
        goals: newGoals.map(goal => [...goal])
      };
      
      setMultiAgentHistory([...multiAgentHistory, positionsForHistory]);
      setMultiAgentHistoryIndex(multiAgentHistory.length);
    }
    
    setGridSize(newSize);
  };

  const handleStart = async () => {
    setIsRunning(true);
    setIsPaused(false);
    setIsFinished(false);
    setTime(0);
    setCurrentStepIndex(0);

    try {
      const currentGridSize = GRID_SIZES[gridSize];
      let requestData;
      
      if (agentMode === "left") {
        requestData = {
          start: startPosition,
          end: endPosition,
          gridSize: currentGridSize,
          walls: walls,
          initialAltitudes: initialAltitudes,
        };
      } else {
        requestData = {
          starts: agentStarts,
          goals: agentGoals,
          gridSize: currentGridSize,
          walls: walls,
        };
      }

      const endpoint = ALGORITHM_ENDPOINTS[selectedAlgorithm];
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestData),
      });

      const data = await response.json();
      if (data.steps && Array.isArray(data.steps)) {
        setWalkSteps(data.steps);
      }
    } catch (error) {
      handleStop();
    }
  };

  // Map generation functions

  const handleNodeRandomisation = () => {
    if (isRunning) return;
    
    // Clear grid for random walk algorithms, no recalculation
    const isRandomWalkAlgorithm = ['randomwalk', 'biasedrandomwalk'].includes(selectedAlgorithm);
    if (isRandomWalkAlgorithm) {
      setWalkSteps(null);
      setCurrentStepIndex(0);
      setIsFinished(false);
    }
    
    if (agentMode === "left") {
      if (singleAgentGridRef.current) {
        singleAgentGridRef.current.randomizeNodes();
      }
    } else {
      if (multiAgentGridRef.current) {
        multiAgentGridRef.current.randomizeNodes();
      }
    }
  };

  const handleGenerateMaze = () => {
    if (isRunning) return;
    
    try {
      const { rows, cols } = GRID_SIZES[gridSize];
      const mazeGenerator = new MazeGenerator(rows, cols);
      
      let newWalls;
      const isMultiAgent = agentMode === "right";
      
      // Get current node positions for single agent mode
      if (isMultiAgent && agentStarts.length > 0) {
        newWalls = mazeGenerator.generateMaze(0.1, isMultiAgent, agentStarts, agentGoals);
      } else if (agentMode === "left" && startPosition && endPosition) {
        // Pass current node positions to preserve them
        newWalls = mazeGenerator.generateMaze(0.1, false, [startPosition], [endPosition]);
      } else {
        newWalls = mazeGenerator.generateMaze(0.1);
      }
      
      setWalls(newWalls);
      setWalkSteps(null);
      setCurrentStepIndex(0);
      setIsFinished(false);
    } catch (error) {
      console.error('Error generating maze:', error);
    }
  };
  
  // Updated handleGenerateRandomMaze without node repositioning
  const handleGenerateRandomMaze = () => {
    if (isRunning) return;
    
    try {
      const { rows, cols } = GRID_SIZES[gridSize];
      const mazeGenerator = new MazeGenerator(rows, cols);
      
      const isMultiAgent = agentMode === "right";
      const density = isMultiAgent ? 0.2 : 0.3;
      
      let newWalls;
      if (isMultiAgent && agentStarts.length > 0) {
        newWalls = mazeGenerator.generateRandomMaze(density, isMultiAgent, agentStarts, agentGoals);
      } else if (agentMode === "left" && startPosition && endPosition) {
        // Pass current node positions to preserve them
        newWalls = mazeGenerator.generateRandomMaze(density, false, [startPosition], [endPosition]);
      } else {
        newWalls = mazeGenerator.generateRandomMaze(density, isMultiAgent);
      }
      
      setWalls(newWalls);
      setWalkSteps(null);
      setCurrentStepIndex(0);
      setIsFinished(false);
    } catch (error) {
      console.error('Error generating random maze:', error);
    }
  };
  
  // Updated handleGenerateOpenGrid without node repositioning
  const handleGenerateOpenGrid = () => {
    if (isRunning) return;
    
    try {
      const { rows, cols } = GRID_SIZES[gridSize];
      const mazeGenerator = new MazeGenerator(rows, cols);
      
      let newWalls;
      const isMultiAgent = agentMode === "right";
      
      if (isMultiAgent && agentStarts.length > 0) {
        newWalls = mazeGenerator.generateOpenGrid(isMultiAgent, agentStarts, agentGoals);
      } else if (agentMode === "left" && startPosition && endPosition) {
        // Pass current node positions to preserve them
        newWalls = mazeGenerator.generateOpenGrid(false, [startPosition], [endPosition]);
      } else {
        newWalls = mazeGenerator.generateOpenGrid(isMultiAgent);
      }
      
      setWalls(newWalls);
      setWalkSteps(null);
      setCurrentStepIndex(0);
      setIsFinished(false);
    } catch (error) {
      console.error('Error generating open grid:', error);
    }
  };

  // Agent management functions
  const handleAddAgent = () => {
    const currentAgentList = [...agentList];
    const currentAgentStarts = [...agentStarts];
    const currentAgentGoals = [...agentGoals];
    
    if (currentAgentList.length >= 4) return;
    
    const { rows, cols } = GRID_SIZES[gridSize];
    
    const newAgentId = currentAgentList.length;
    
    let startRow, startCol, goalRow, goalCol;
    let attempts = 0;
    const MAX_ATTEMPTS = 100;
    
    // Find valid start position
    do {
      startRow = Math.floor(Math.random() * rows);
      startCol = Math.floor(Math.random() * (cols / 3));
      attempts++;
      if (attempts > MAX_ATTEMPTS) break;
    } while (
      walls[startRow]?.[startCol] === true || 
      currentAgentStarts.some(([r, c]) => r === startRow && c === startCol) ||
      currentAgentGoals.some(([r, c]) => r === startRow && c === startCol)
    );
    
    attempts = 0;
    
    // Find valid goal position
    do {
      goalRow = Math.floor(Math.random() * rows);
      goalCol = Math.floor(Math.random() * (cols / 3) + (2 * cols / 3));
      attempts++;
      if (attempts > MAX_ATTEMPTS) break;
    } while (
      walls[goalRow]?.[goalCol] === true || 
      currentAgentStarts.some(([r, c]) => r === goalRow && c === goalCol) ||
      currentAgentGoals.some(([r, c]) => r === goalRow && c === goalCol)
    );
    
    const newAgent = {
      id: newAgentId,
      start: [startRow, startCol],
      goal: [goalRow, goalCol]
    };
    
    const updatedAgents = [...currentAgentList, newAgent];
    const newStarts = [...currentAgentStarts, [startRow, startCol]];
    const newGoals = [...currentAgentGoals, [goalRow, goalCol]];
    
    setAgentList(updatedAgents);
    setAgentStarts(newStarts);
    setAgentGoals(newGoals);
    
    if (multiAgentGridRef.current) {
      multiAgentGridRef.current.updateAgentPositions(newStarts, newGoals);
    }
    
    const positionsForHistory = {
      starts: newStarts.map(start => [...start]),
      goals: newGoals.map(goal => [...goal])
    };
    
    setMultiAgentHistory([...multiAgentHistory, positionsForHistory]);
    setMultiAgentHistoryIndex(multiAgentHistory.length);
  };

  const handleRemoveAgent = () => {
    const currentAgentList = [...agentList];
    const currentAgentStarts = [...agentStarts];
    const currentAgentGoals = [...agentGoals];
    
    if (currentAgentList.length <= 2) return;
    
    const updatedAgents = currentAgentList.slice(0, -1);
    const newStarts = currentAgentStarts.slice(0, -1);
    const newGoals = currentAgentGoals.slice(0, -1);
    
    setAgentList(updatedAgents);
    setAgentStarts(newStarts);
    setAgentGoals(newGoals);
    
    if (multiAgentGridRef.current) {
      multiAgentGridRef.current.updateAgentPositions(newStarts, newGoals);
    }
    
    const positionsForHistory = {
      starts: newStarts.map(start => [...start]),
      goals: newGoals.map(goal => [...goal])
    };
    
    setMultiAgentHistory([...multiAgentHistory, positionsForHistory]);
    setMultiAgentHistoryIndex(multiAgentHistory.length);
  };

  // UI helper functions
  const renderControlButtons = () => {
    if (isFinished) {
      return (
        <button
          className="button-color"
          onClick={handleReset}
          style={{ width: "100%" }}
        >
          Reset
        </button>
      );
    }

    return (
      <>
        <button
          className="button-color"
          onClick={handlePauseResume}
          disabled={!isRunning}
        >
          {isPaused ? "Resume" : "Pause"}
        </button>
        <button
          className="button-color"
          onClick={handleStop}
          disabled={!isRunning}
        >
          Stop
        </button>
      </>
    );
  };

  // Render function
  return (
    <div id="app">
      <header id="app-header">
        <h1 id="app-title">Search Algorithm Visualiser</h1>
      </header>
      <main id="app-main">
        <div id="column-container">
          {/* Left column - Controls */}
          <div id="left-column" className="column">
            <div id="toggle-group">
              <ToggleButton
                onChange={handleAgentModeChange}
                disabled={isRunning}
              />
            </div>
            <div id="dropdown-container">
              {/* Algorithm dropdown */}
              <div className="dropdown-item">
                <label htmlFor="algorithm">Algorithm</label>
                <div className="select-wrapper">
                  <select 
                    id="algorithm" 
                    name="algorithm" 
                    value={selectedAlgorithm}
                    onChange={handleAlgorithmChange}
                    disabled={isRunning}
                  >
                    <option value="" disabled hidden>Select</option>
                    {getCurrentAlgorithmOptions().map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <div className="arrow-background">
                    <CustomArrow />
                  </div>
                </div>
              </div>
              
              {/* Grid Size dropdown */}
              <div className="dropdown-item">
                <label htmlFor="gridSize">Grid Size</label>
                <div className="select-wrapper">
                  <select 
                    id="gridSize" 
                    name="gridSize" 
                    value={gridSize}
                    onChange={handleGridSizeChange}
                    disabled={isRunning}
                  >
                    {GRID_SIZE_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <div className="arrow-background">
                    <CustomArrow />
                  </div>
                </div>
              </div>
              
              {/* Start button */}
              <div className="dropdown-item">
                <div className="label-spacer"></div>
                <button
                  id="start-traversal-btn"
                  className="button-color"
                  onClick={handleStart}
                  disabled={
                    isRunning ||
                    isFinished ||
                    !selectedAlgorithm ||
                    (agentMode === "left" 
                      ? (!startPosition || !endPosition)
                      : (agentStarts.length < 2 || agentStarts.some(start => !start) || agentGoals.some(goal => !goal))
                    )
                  }
                >
                  {isFinished
                    ? `Finished: ${formatTime(time)}`
                    : isRunning
                    ? formatTime(time)
                    : "Start Traversal"}
                </button>
              </div>
              
              {/* Speed slider */}
              <div className="dropdown-item">
                <label htmlFor="speed">Speed</label>
                <div id="slider-container">
                  <input
                    type="range"
                    id="speed"
                    name="speed"
                    min="1"
                    max="100"
                    value={speed}
                    onChange={handleSpeedChange}
                    className="speed-slider"
                  />
                </div>
              </div>
              
              {/* Control buttons */}
              <div id="button-row">{renderControlButtons()}</div>
              
              {/* Clear button */}
              <div id="button-row">
                <button
                  className="button-color"
                  disabled={isRunning}
                  onClick={handleClear}
                >
                  Clear
                </button>
              </div>

              {/* Agent and node management */}
              <div style={{ 
                  marginBottom: "8px", 
                  display: "flex", 
                  gap: "7px", 
                  width: "calc(100px + 8px + 180px)",
                  alignItems: "center",
                  justifyContent: "space-between"
                }}>
                {agentMode === "right" && (
                  <>
                    <button
                      className="button-color"
                      onClick={handleAddAgent}
                      disabled={isRunning || agentList.length >= 4}
                      style={{ 
                        width: "31%", 
                        padding: "4px 2px", 
                        fontSize: "0.85rem", 
                        height: "32px",
                        lineHeight: "1",
                        margin: "0",
                      }}
                    >
                      Add Agent
                    </button>
                    <button
                      className="button-color"
                      onClick={handleRemoveAgent}
                      disabled={isRunning || agentList.length <= 2}
                      style={{ 
                        width: "31%", 
                        padding: "4px 2px", 
                        fontSize: "0.85rem", 
                        height: "32px",
                        lineHeight: "1",
                        margin: "0"
                      }}
                    >
                      Remove Agent
                    </button>
                  </>
                )}
                
                <button
                  id="randomise-nodes-btn"
                  className="button-color"
                  onClick={handleNodeRandomisation}
                  disabled={isRunning}
                  style={{ 
                    width: agentMode === "right" ? "31%" : "100%", 
                    padding: "4px 2px", 
                    fontSize: "0.85rem", 
                    height: "32px",
                    lineHeight: "1",
                    margin: "0"
                  }}
                >
                  {agentMode === "right" ? "Randomise Agents" : "Randomise Nodes"}
                </button>
              </div>

              {/* Maze generation */}
              <div className="maze-management">
                <div className="maze-controls">
                  <button
                    className="button-color maze-button"
                    onClick={handleGenerateMaze}
                    disabled={isRunning}
                  >
                    Generate Maze
                  </button>
                  
                  {agentMode === "left" ? (
                    <button
                      className="button-color maze-button"
                      onClick={handleGenerateRandomMaze}
                      disabled={isRunning}
                    >
                      Random Maze
                    </button>
                  ) : (
                    <button
                      className="button-color maze-button"
                      onClick={handleGenerateOpenGrid}
                      disabled={isRunning}
                    >
                      Open Grid
                    </button>
                  )}
                </div>
              </div>
              
              {/* Legend */}
              {agentMode === "left" ? 
                <KeyLegend mode="single" selectedAlgorithm={selectedAlgorithm} /> : 
                <KeyLegend mode="multi" />
              }
            </div>
          </div>
          
          <div id="column-divider"></div>
          
          {/* Right column - visualisation */}
          <div id="right-column" className="column">
            <div className="grid-layout">
              <div className="grid-col-3">
                <AlgorithmInfoBox
                  selectedAlgorithm={selectedAlgorithm}
                  isRunning={isRunning}
                  isFinished={isFinished}
                  walkData={walkSteps}
                  agentMode={agentMode}
                />
              </div>
              <div className="grid-col-2">
                <PseudocodeDisplay selectedAlgorithm={selectedAlgorithm} />
              </div>
            </div>
            <div className="grid-container">
              {agentMode === "left" ? (
                <SingleAgentGrid
                  ref={singleAgentGridRef}
                  size={gridSize}
                  walkData={walkSteps}
                  currentStep={currentStepIndex}
                  onNodePositionsChange={handleNodePositionsChange}
                  isRunning={isRunning}
                  isFinished={isFinished}
                  walls={walls}
                  onWallsChange={setWalls}
                  initialAltitudes={initialAltitudes}
                  selectedAlgorithm={selectedAlgorithm}
                  onNodeDragEnd={handleNodeDragEnd}
                />
              ) : (
                <MultiAgentGrid
                  ref={multiAgentGridRef}
                  size={gridSize}
                  walkData={walkSteps}
                  currentStep={currentStepIndex}
                  onAgentPositionsChange={onAgentPositionsChange}
                  isRunning={isRunning}
                  isFinished={isFinished}
                  walls={walls}
                  onWallsChange={setWalls}
                  onNodeDragEnd={handleMultiNodeDragEnd}
                  agentStarts={agentStarts}
                  agentGoals={agentGoals}
                />
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;