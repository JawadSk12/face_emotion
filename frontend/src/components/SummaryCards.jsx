import React, { useMemo } from "react";

const SummaryCards = ({ events, loading }) => {
  const stats = useMemo(() => {
    const records = events || [];
    const confident = records.filter((event) => event.emotion !== "Uncertain");
    const counts = confident.reduce((result, event) => {
      result[event.emotion] = (result[event.emotion] || 0) + 1;
      return result;
    }, {});
    const mostDetected = Object.entries(counts).sort(
      ([, countA], [, countB]) => countB - countA
    )[0]?.[0];

    return {
      total: records.length,
      confidentPercentage: records.length
        ? (confident.length / records.length) * 100
        : 0,
      mostDetected: mostDetected || "—",
    };
  }, [events]);

  return (
    <div className="summary-grid">
      <div className="summary-item">
        <span className="summary-label">Expression samples</span>
        <span className="summary-value">{loading ? "…" : stats.total}</span>
      </div>
      <div className="summary-item">
        <span className="summary-label">Clear predictions</span>
        <span className="summary-value">
          {loading ? "…" : `${stats.confidentPercentage.toFixed(0)}%`}
        </span>
      </div>
      <div className="summary-item">
        <span className="summary-label">Most detected expression</span>
        <span className="summary-value">
          {loading ? "…" : stats.mostDetected}
        </span>
      </div>
    </div>
  );
};

export default SummaryCards;
