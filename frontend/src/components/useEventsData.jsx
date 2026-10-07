import { useEffect, useState } from "react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function useEventsData(pollIntervalMs = 5000) {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let active = true;

    const fetchEvents = async () => {
      try {
        const response = await fetch(`${API_URL}/events?limit=100`);
        if (!response.ok) {
          throw new Error(`Events API returned HTTP ${response.status}.`);
        }
        const data = await response.json();
        if (active) {
          setEvents(data.events);
          setError(null);
        }
      } catch (fetchError) {
        if (active) {
          console.error("Error fetching events:", fetchError);
          setError(fetchError);
        }
      } finally {
        if (active) setLoading(false);
      }
    };

    fetchEvents();
    const intervalId = window.setInterval(fetchEvents, pollIntervalMs);
    return () => {
      active = false;
      window.clearInterval(intervalId);
    };
  }, [pollIntervalMs]);

  return { events, loading, error };
}
