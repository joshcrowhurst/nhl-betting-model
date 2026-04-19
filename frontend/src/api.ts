import { Prediction, Performance, ModelRun } from "./types";

const BASE = process.env.REACT_APP_API_URL || "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  getToday: () => get<Prediction[]>("/api/predictions/today"),
  getRecent: (days = 14) => get<Prediction[]>(`/api/predictions/recent?days=${days}`),
  getPerformance: () => get<Performance | null>("/api/performance"),
  getRuns: () => get<ModelRun[]>("/api/performance/runs"),
};
