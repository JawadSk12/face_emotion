import React, { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const EMOTIONS = [
  "Angry",
  "Disgust",
  "Fear",
  "Happy",
  "Sad",
  "Surprise",
  "Neutral",
  "Uncertain",
];

const EmotionChart = ({ events }) => {
  const data = useMemo(
    () =>
      (events || [])
        .slice()
        .reverse()
        .map((event, index) => ({
          index: index + 1,
          emotion: event.emotion,
          emotionIndex: EMOTIONS.indexOf(event.emotion),
          timestamp: event.timestamp,
        }))
        .filter((event) => event.emotionIndex >= 0),
    [events]
  );

  return (
    <div className="chart-container">
      {data.length === 0 ? (
        <p className="muted-text">No expression predictions yet.</p>
      ) : (
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
            <XAxis
              dataKey="index"
              tick={{ fontSize: 10 }}
              label={{ value: "Recent samples", position: "insideBottom", offset: -4 }}
            />
            <YAxis
              dataKey="emotionIndex"
              type="number"
              domain={[0, 7]}
              ticks={[0, 1, 2, 3, 4, 5, 6, 7]}
              tickFormatter={(value) => EMOTIONS[value] || ""}
              tick={{ fontSize: 10 }}
              width={76}
            />
            <Tooltip
              formatter={(_value, _name, item) => [item.payload.emotion, "Expression"]}
              labelFormatter={(_label, payload) =>
                payload?.[0]?.payload?.timestamp || "Sample"
              }
            />
            <Line
              type="stepAfter"
              dataKey="emotionIndex"
              stroke="#818cf8"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default EmotionChart;
