import React from "react";
import VideoFeed from "./VideoFeed";
import EmotionChart from "./EmotionChart";
import EmotionDistribution from "./EmotionDistribution";
import AttendanceTable from "./AttendanceTable";
import SummaryCards from "./SummaryCards";
import useEventsData from "./useEventsData";

const Dashboard = () => {
  const { events, loading, error } = useEventsData();

  return (
    <div className="dashboard-layout">
      <section className="top-row">
        <div className="card video-card">
          <h2>Live Camera Feed</h2>
          <VideoFeed />
        </div>
        <div className="card summary-card">
          <SummaryCards events={events} loading={loading} />
        </div>
      </section>

      <section className="middle-row">
        <div className="card">
          <h2>Expression Timeline</h2>
          <EmotionChart events={events} />
        </div>
        <div className="card">
          <h2>Expression Distribution</h2>
          <EmotionDistribution events={events} />
        </div>
      </section>

      <section className="bottom-row">
        <div className="card">
          <h2>Recent Expression Predictions</h2>
          {error && (
            <p className="error-text">
              Failed to load events. Check backend at http://localhost:8000
            </p>
          )}
          <AttendanceTable events={events} loading={loading} />
        </div>
      </section>
    </div>
  );
};

export default Dashboard;
