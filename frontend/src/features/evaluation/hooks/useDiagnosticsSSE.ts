"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { DiagnoseRequest, DiagnoseResponse } from "../types";
import { diagnoseEngines, mapReportsToDiagnoseResponse } from "../api";

export const useDiagnosticsSSE = () => {
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DiagnoseResponse | null>(null);
  
  const eventSourceRef = useRef<EventSource | null>(null);

  const cleanup = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);

  const startDiagnostic = useCallback(async (params: DiagnoseRequest) => {
    // Reset states
    setIsEvaluating(true);
    setProgress(0);
    setStatusMessage("Queueing task...");
    setError(null);
    setResult(null);
    cleanup();

    try {
      // 1. Trigger the asynchronous task to get task_id
      const { task_id } = await diagnoseEngines(params);
      
      // 2. Open SSE connection
      const es = new EventSource(`/api/v1/evaluation/status/${task_id}`);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          if (data.event === "error" || data.status === "Error") {
            setError(data.message || "An unexpected error occurred during diagnostics.");
            cleanup();
            setIsEvaluating(false);
            return;
          }

          if (data.progress !== undefined) {
            setProgress(data.progress);
          }
          
          if (data.status) {
            setStatusMessage(data.status);
          }

          if ((data.event === "done" || data.status === "100% Complete") && data.reports) {
            const mappedResult = mapReportsToDiagnoseResponse(data.reports);
            setResult(mappedResult);
            cleanup();
            setIsEvaluating(false);
          }

        } catch (parseError) {
          console.error("Failed to parse SSE message:", parseError);
        }
      };

      es.onerror = (err) => {
        console.error("SSE Connection Error:", err);
        setError("Connection lost. The task might still be running; please check back in a moment.");
        cleanup();
        setIsEvaluating(false);
      };

    } catch (err: any) {
      console.error("Failed to initiate diagnostics:", err);
      setError(err.message || "Failed to start diagnostics.");
      setIsEvaluating(false);
    }
  }, [cleanup]);

  return {
    isEvaluating,
    progress,
    statusMessage,
    error,
    result,
    startDiagnostic
  };
};
