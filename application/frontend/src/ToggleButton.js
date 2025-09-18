import React, { useState } from "react";
import { User, Users } from "lucide-react";
import "./css/ToggleButton.css";

const ToggleButton = ({ disabled = false, onChange }) => {
  const [selected, setSelected] = useState("left");

  const handleClick = (side) => {
    if (!disabled && side !== selected) {
      setSelected(side);
      onChange?.(side);
    }
  };

  return (
    <div className="toggle-container" role="group" aria-label="Agent Mode Selection">
      <div className="toggle-buttons-wrapper">
        <div className={`toggle-slider ${selected}`} />
        <button
          type="button"
          onClick={() => handleClick("left")}
          disabled={disabled}
          className={`toggle-button ${selected === "left" ? "active" : ""}`}
          aria-pressed={selected === "left"}
        >
          Single Agent
          <User 
            className="toggle-icon"
            size={14} 
            color={selected === "left" ? "black" : "white"}
            opacity={disabled ? 0.5 : 1} 
          />
        </button>
        <button
          type="button"
          onClick={() => handleClick("right")}
          disabled={disabled}
          className={`toggle-button ${selected === "right" ? "active" : ""}`}
          aria-pressed={selected === "right"}
        >
          Multi Agent
          <Users 
            className="toggle-icon"
            size={14} 
            color={selected === "right" ? "black" : "white"}
            opacity={disabled ? 0.5 : 1} 
          />
        </button>
      </div>
    </div>
  );
};

export default ToggleButton;