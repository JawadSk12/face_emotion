import React from "react";

const RecentEventsTable = ({ events, loading }) => {
  if (loading) {
    return <p className="muted-text">Loading events…</p>;
  }

  if (!events || events.length === 0) {
    return <p className="muted-text">No events recorded yet.</p>;
  }

  const rows = events.slice().reverse();

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
          {rows.map((event, index) => (
            <tr key={`${event.timestamp}-${index}`}>
              <td>{event.timestamp}</td>
              <td>{event.emotion}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default RecentEventsTable;
