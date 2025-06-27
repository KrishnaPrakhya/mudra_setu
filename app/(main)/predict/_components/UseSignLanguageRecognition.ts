"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { HolisticLandmarkerResult } from "@mediapipe/tasks-vision";
import { LandmarkData, mediaPipeClient } from "./MediaPipeClient";

// Define the structure of the prediction received from the backend
interface PredictionResponse {
  status: "prediction" | "buffering" | "low_confidence" | "error";
  prediction?: string;
  confidence?: number;
  progress?: number;
  message?: string;
}

export function useSignLanguageRecognition() {
  // Refs for core objects
  const videoRef = useRef<HTMLVideoElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const animationFrameId = useRef<number | null>(null);

  // State for UI and data
  const [isCapturing, setIsCapturing] = useState(false);
  const [landmarks, setLandmarks] = useState<HolisticLandmarkerResult | null>(null);
  const [predictions, setPredictions] = useState<string[]>([]);
  const [isBuffering, setIsBuffering] = useState(false);
  const [progress, setProgress] = useState(0);

  // WebSocket connection management
  const connectSocket = useCallback(() => {
    // Replace with your actual backend WebSocket URL
    const socket = new WebSocket("ws://localhost:8000/ws/predict");

    socket.onopen = () => {
      console.log("WebSocket connection established.");
    };

    socket.onmessage = (event) => {
      try {
        const data: PredictionResponse = JSON.parse(event.data);
        
        if (data.status === "prediction" && data.prediction && data.confidence) {
          setIsBuffering(false);
          const newPrediction = `${data.prediction} (${data.confidence.toFixed(2)}%)`;
          setPredictions((prev) => [newPrediction, ...prev].slice(0, 10));
        } else if (data.status === "buffering" && data.progress) {
          setIsBuffering(true);
          setProgress(data.progress);
        } else if (data.status === "low_confidence") {
          setIsBuffering(false); 
        } else if (data.status === "error") {
          console.error("Backend error:", data.message);
          setIsBuffering(false);
        }
      } catch (error) {
        console.error("Failed to parse prediction message:", error);
      }
    };

    socket.onclose = () => {
      console.log("WebSocket connection closed.");
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    socketRef.current = socket;
  }, []);

  // Main detection loop
  const runDetection = useCallback(async () => {
    if (videoRef.current) {
      const landmarkData: LandmarkData | null = mediaPipeClient.detect(videoRef.current);

      if (landmarkData) {
        // Update landmarks for canvas drawing
        setLandmarks(landmarkData.results);
        
        // Send keypoints to backend if socket is ready
        if (socketRef.current?.readyState === WebSocket.OPEN) {
          const keypoints = landmarkData.keypoints;
          socketRef.current.send(JSON.stringify(keypoints));
        }
      }
    }
    // Continue the loop
    animationFrameId.current = requestAnimationFrame(runDetection);
  }, []);

  // Function to start/stop the capture
  const toggleCapture = useCallback(async () => {
    if (isCapturing) {
      // --- STOP CAPTURING ---
      setIsCapturing(false);
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
      setLandmarks(null);
    } else {
      // --- START CAPTURING ---
      setIsCapturing(true);
      setPredictions([]);
      setIsBuffering(false);
      setProgress(0);
      await mediaPipeClient.initialize();
      connectSocket();
      runDetection();
    }
  }, [isCapturing, runDetection, connectSocket]);
  
  // Cleanup effect
  useEffect(() => {
    // This function is returned and will be called when the component unmounts
    return () => {
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  return {
    videoRef,
    landmarks,
    predictions,
    isCapturing,
    toggleCapture,
    isBuffering,
    progress,
  };
}