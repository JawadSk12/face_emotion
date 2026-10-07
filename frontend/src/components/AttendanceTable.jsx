import React from "react";

const AttendanceTable = ({ events, loading }) => {
  if (loading) {
    return <p className="muted-text">Loading events…</p>;
  }

  if (!events || events.length === 0) {
    return <p className="muted-text">No events recorded yet.</p>;
  }

  const rows = events.slice().reverse(); // newest at bottom or top depending on taste

  return (
    <div className="table-wrapper">
      <table className="events-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Facial expression</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((e, idx) => (
            <tr key={idx}>
              <td>{e.timestamp}</td>
              <td>{e.emotion}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AttendanceTable;
