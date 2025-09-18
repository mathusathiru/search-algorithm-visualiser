import React from "react";

const AGENT_COLORS = ["#3366FF", "#FF9933", "#66CC66", "#FF66B2"];

const KeyLegend = ({ mode = "single", selectedAlgorithm = "" }) => {
  const startSymbol = { id: "start-symbol", label: "Start Node", isCircle: true };
  const endSymbol = { id: "end-symbol", label: "Goal Node", isCircle: true };
  const unvisitedSymbol = { id: "unvisited-symbol", label: "Unvisited Node" };
  const visitedSymbol = { id: "visited-symbol", label: "Visited Node" };
  const wallSymbol = { id: "wall-symbol", label: "Wall" };
  const pathSymbol = { id: "path-symbol", label: "Path" };
  const currentSymbol = { id: "current-symbol", label: "Current Node" };
  const lowWeightSymbol = { id: "low-weight-symbol", label: "Low Cost" };
  const highWeightSymbol = { id: "high-weight-symbol", label: "High Cost" };
  
  const getLegendRows = () => {
    if (mode === "multi") {
      return null;
    }

    if (["dijkstra", "astar"].includes(selectedAlgorithm)) {
      return [
        [startSymbol, endSymbol],
        [lowWeightSymbol, highWeightSymbol],
        [wallSymbol, currentSymbol],
        [pathSymbol, { id: "parent-symbol", label: "Parent Link" }]
      ];
    }
    
    if (selectedAlgorithm === "dfs") {
              return [
        [startSymbol, endSymbol],
        [unvisitedSymbol, currentSymbol],
        [{ id: "considered-symbol", label: "Considered Node" }, pathSymbol],
        [visitedSymbol, wallSymbol]
      ];
    }
    
    return [
      [startSymbol, endSymbol],
      [unvisitedSymbol, pathSymbol],
      [currentSymbol, wallSymbol],
      [visitedSymbol, null]
    ];
  };

  const legendRows = getLegendRows();

  const renderSymbol = (item) => (
    <div 
      className={`symbol ${item.isCircle ? "circle" : ""}`} 
      id={item.id} 
    />
  );

  const renderKeyItem = (item, isLeft) => {
    if (!item) return <div className={`key-item ${isLeft ? "left-column" : "right-column"}`}></div>;
    return (
      <div className={`key-item ${isLeft ? "left-column" : "right-column"}`}>
        {renderSymbol(item)}
        <span>{item.label}</span>
      </div>
    );
  };

  if (mode === "multi") {
    const multiAgentItems = [
      [
        { id: "agent1-start", label: "Start Node", isCircle: true, customStyle: { backgroundColor: AGENT_COLORS[0] } },
        { id: "agent1-considered", label: "Considered Node", customStyle: { backgroundColor: `${AGENT_COLORS[0]}35` } },
        { id: "agent1-visit", label: "Visited Node", customStyle: { backgroundColor: `${AGENT_COLORS[0]}60` } },
        { id: "agent-travel", label: "Travelling Node", customStyle: { backgroundColor: AGENT_COLORS[0] } }
      ],
      [
        { id: "agent1-goal", label: "Goal Node", isCircle: true, customStyle: { backgroundColor: "white", border: `2px solid ${AGENT_COLORS[0]}` } },
        { id: "multi-path", label: "Path", customStyle: { backgroundColor: AGENT_COLORS[0] } },
        { id: "multi-wall", label: "Wall", customStyle: { backgroundColor: "black" } }, null
      ]
    ];

    const renderMultiAgentSymbol = (item) => (
      <div 
        className={`symbol ${item.isCircle ? "circle" : ""}`}
        style={item.customStyle || {}}
      />
    );

    const renderMultiAgentItem = (item, columnIndex) => (
      <div className={`key-item ${columnIndex === 0 ? "left-column" : "right-column"}`}>
        {renderMultiAgentSymbol(item)}
        <span>{item.label}</span>
      </div>
    );

    return (
      <div id="key-box">
        {multiAgentItems[0].map((item, rowIndex) => (
          <div key={`row-${rowIndex}`} className="key-row">
            {renderMultiAgentItem(item, 0)}
            {multiAgentItems[1][rowIndex] && renderMultiAgentItem(multiAgentItems[1][rowIndex], 1)}
          </div>
        ))}
      </div>
    );
  }

  return (
    <div id="key-box">
      {legendRows.map((row, index) => (
        <div key={`row-${index}`} className="key-row">
          {renderKeyItem(row[0], true)}
          {renderKeyItem(row[1], false)}
        </div>
      ))}
    </div>
  );
};

export default KeyLegend;