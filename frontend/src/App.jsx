import React from "react";
import Dashboard from "./components/Dashboard";

const App = () => {
  return (
    <div className="app-root">
      <header className="app-header">
        <div>
          <h1>Live Facial Expression Dashboard</h1>
          <p>Real-time expression estimates for faces in the camera view</p>
        </div>
        <div className="app-header-badge">
          <span className="dot live-dot"></span>
          <span>Local demo</span>
        </div>
      </header>
      <Dashboard />
    </div>
  );
};

export default App;
