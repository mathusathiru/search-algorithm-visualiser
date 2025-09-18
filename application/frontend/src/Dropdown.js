import React from "react";

const CustomArrow = () => (
  <svg width="10" height="10" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M2 4L6 8L10 4" stroke="black" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const Dropdown = ({ 
  label, options,  id,  name,  onChange,  defaultValue,  showSelect = true, disabled = false 
}) => {
  return (
    <div className="dropdown-item">
      <label htmlFor={id}>{label}</label>
      <div className="select-wrapper">
        <select 
          id={id} 
          name={name} 
          defaultValue={defaultValue || (showSelect ? "" : options[0].value)}
          onChange={onChange}
          disabled={disabled}
        >
          {showSelect && <option value="" disabled hidden>Select</option>}
          {options.map((option) => (
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
  );
};

export default Dropdown;