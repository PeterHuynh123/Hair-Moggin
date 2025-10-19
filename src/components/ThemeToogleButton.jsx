import React, { useEffect, useState } from "react";
import "@theme-toggles/react/css/Within.css";
import { Within } from "@theme-toggles/react";
import "./ThemeToogleButton.css";

const DARK_PALETTE = ["#e3e3e3", "#0c0c0c", "#6200ff"];
const LIGHT_PALETTE = ["#0c0c0c", "#e3e3e3", "#6200ff"];

function setCSSVariables(palette) {
  document.documentElement.style.setProperty("--primary", palette[0]);
  document.documentElement.style.setProperty("--background", palette[1]);
  document.documentElement.style.setProperty("--accent", palette[2]);
}

const ThemeToogleButton = () => {
  const [isDark, setIsDark] = useState(() => {
    const stored = localStorage.getItem("theme");
    return stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  useEffect(() => {
    setCSSVariables(isDark ? DARK_PALETTE : LIGHT_PALETTE);
    document.body.style.background = isDark ? DARK_PALETTE[1] : LIGHT_PALETTE[1];
    localStorage.setItem("theme", isDark ? "dark" : "light");
  }, [isDark]);

  return (
    <div className="theme-toggle-btn">
      <Within
        toggled={isDark}
        duration={750}
        onToggle={() => setIsDark((d) => !d)}
        className="theme-toggle-icon"
        aria-label="Toggle theme"
      />
    </div>
  );
};

export default ThemeToogleButton; 