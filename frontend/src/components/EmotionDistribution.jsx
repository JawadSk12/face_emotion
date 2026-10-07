import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid
} from "recharts";

const EmotionDistribution = ({ events }) => {
  const data = useMemo(() => {
    const counts = {};
    (events || []).forEach((e) => {
      counts[e.emotion] = (counts[e.emotion] || 0) + 1;
    });

    return Object.entries(counts).map(([emotion, count]) => ({
      emotion,
      count
    }));
  }, [events]);

  return (
    <div className="chart-container">
      {data.length === 0 ? (
        <p className="muted-text">No events yet…</p>
      ) : (
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
            <XAxis dataKey="emotion" tick={{ fontSize: 10 }} />
            <YAxis allowDecimals={false} tick={{ fontSize: 10 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default EmotionDistribution;
