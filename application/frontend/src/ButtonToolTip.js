import React, { useState, useRef, useEffect } from "react";

const tooltipStyles = `
  .tooltip-flex-container {
    flex: 1;
    margin: 0 4px;
  }
  
  .tooltip-flex-container:first-child {
    margin-left: 0;
  }
  
  .tooltip-flex-container:last-child {
    margin-right: 0;
  }
  
  .tooltip-flex-container button {
    width: 100%;
  }
`;

const ButtonTooltip = ({ children, tooltip, position = "top", width = "180px" }) => {
  useEffect(() => {
    if (!document.getElementById("tooltip-styles")) {
      const styleElement = document.createElement("style");
      styleElement.id = "tooltip-styles";
      styleElement.innerHTML = tooltipStyles;
      document.head.appendChild(styleElement);
      
      return () => {
        document.head.removeChild(styleElement);
      };
    }
  }, []);

  const [isVisible, setIsVisible] = useState(false);
  const [tooltipPosition, setTooltipPosition] = useState({
    top: "auto",
    bottom: "125%",
    left: "50%",
    transform: "translateX(-50%)"
  });
  
  const tooltipRef = useRef(null);
  const containerRef = useRef(null);
  
  useEffect(() => {
    if (isVisible && containerRef.current && tooltipRef.current) {
      const containerRect = containerRef.current.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      
      // Get position of viewport
      const viewportWidth = window.innerWidth;
      
      // Calculating positions
      let newPosition = { ...tooltipPosition };
      
      switch (position) {
        case "top":
          newPosition.bottom = "125%";
          newPosition.top = "auto";
          break;
        case "bottom":
          newPosition.top = "125%";
          newPosition.bottom = "auto";
          break;
        case "left":
          newPosition.right = "125%";
          newPosition.left = "auto";
          newPosition.top = "50%";
          newPosition.transform = "translateY(-50%)";
          break;
        case "right":
          newPosition.left = "125%";
          newPosition.right = "auto";
          newPosition.top = "50%";
          newPosition.transform = "translateY(-50%)";
          break;
        default:
          break;
      }
      
      // Viewpoint adjustments
      const tooltipLeft = containerRect.left + (containerRect.width / 2) - (tooltipRect.width / 2);
      const tooltipRight = tooltipLeft + tooltipRect.width;
      
      if (position === "top" || position === "bottom") {
        if (tooltipLeft < 0) {
          newPosition.left = "0";
          newPosition.transform = "translateX(0)";
        } else if (tooltipRight > viewportWidth) {
          newPosition.left = "auto";
          newPosition.right = "0";
          newPosition.transform = "translateX(0)";
        }
      }
      
      setTooltipPosition(newPosition);
    }
  }, [isVisible, position, tooltipPosition]);
  
  // Button row checks
  useEffect(() => {
    if (containerRef.current) {
      const isInButtonRow = containerRef.current.parentElement?.id === "button-row";
      if (isInButtonRow) {
        containerRef.current.className = "tooltip-flex-container";
      }
    }
  }, []);
  
  const tooltipStyle = {
    position: "absolute",
    backgroundColor: "rgba(51, 51, 51, 0.95)",
    color: "white",
    textAlign: "center",
    padding: "8px 12px",
    borderRadius: "6px",
    width: width,
    fontSize: "11px",
    lineHeight: "1.4",
    zIndex: 100,
    opacity: isVisible ? 1 : 0,
    visibility: isVisible ? "visible" : "hidden",
    transition: "opacity 0.3s, visibility 0.3s",
    boxShadow: "0 2px 8px rgba(0, 0, 0, 0.3)",
    pointerEvents: "none",
    ...tooltipPosition
  };
  
  return (
    <div 
      ref={containerRef}
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
    >
      {children}
      <div 
        ref={tooltipRef}
        style={tooltipStyle}
      >
        {tooltip}
      </div>
    </div>
  );
};

export default ButtonTooltip;