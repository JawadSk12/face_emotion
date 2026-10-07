import { useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function VideoFeed() {
  const [streamKey, setStreamKey] = useState(0);
  const [status, setStatus] = useState("connecting");

  return (
    <div className="video-feed">
      <img
        key={streamKey}
        className="video-feed-image"
        src={`${API_URL}/video_feed`}
        alt="Live camera feed with face and expression detections"
        onLoad={() => setStatus("live")}
        onError={() => setStatus("offline")}
      />
      {status !== "live" && (
        <div className="video-feed-overlay" role="status">
          <p>
            {status === "connecting"
              ? "Connecting to the local camera…"
              : "Camera stream unavailable. Start the backend and check camera access."}
          </p>
          {status === "offline" && (
            <button
              type="button"
              className="retry-button"
              onClick={() => {
                setStatus("connecting");
                setStreamKey((key) => key + 1);
              }}
            >
              Retry camera
            </button>
          )}
        </div>
      )}
      <span className={`video-status video-status-${status}`}>
        <span className="dot" />
        {status === "live" ? "Live" : status === "connecting" ? "Connecting" : "Offline"}
      </span>
    </div>
  );
}
