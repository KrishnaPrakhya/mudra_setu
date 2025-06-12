"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import PredictionPanel from "./_components/PredictionPanel";
import VideoDisplay from "./_components/VideoDisplay";

// Define hand connections for landmark visualization
const HAND_CONNECTIONS: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4], // Thumb
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8], // Index finger
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12], // Middle finger
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16], // Ring finger
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20], // Pinky
  [5, 9],
  [9, 13],
  [13, 17], // Palm connections
];
const blobToBase64 = (blob: Blob): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
};

export default function PredictPage() {
  const videoRef = useRef<HTMLVideoElement>(
    null
  ) as React.RefObject<HTMLVideoElement>;
  const [predictions, setPredictions] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [landmarks, setLandmarks] = useState<any>(null);
  const captureLoopRef = useRef<number | null>(null);
  const [fps, setFps] = useState(0);
  const frameTimesRef = useRef<number[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(
    null
  ) as React.RefObject<HTMLCanvasElement>;
  const landmarkCanvasRef = useRef<HTMLCanvasElement>(
    null
  ) as React.RefObject<HTMLCanvasElement>;
  const offscreenCanvas = useRef<HTMLCanvasElement | null>(null);

  // This function is now stable and doesn't depend on changing state
  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const frameTimes = frameTimesRef.current;
    const now = performance.now();
    while (frameTimes.length > 0 && frameTimes[0] <= now - 1000) {
      frameTimes.shift();
    }
    frameTimes.push(now);
    setFps(frameTimes.length);

    const ctx = canvasRef.current.getContext("2d", { alpha: false });
    if (!ctx) return;

    const video = videoRef.current;
    if (canvasRef.current.width !== video.videoWidth) {
      canvasRef.current.width = video.videoWidth;
      canvasRef.current.height = video.videoHeight;
    }

    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    const imageBlob = await new Promise<Blob | null>((resolve) => {
      canvasRef.current?.toBlob((blob) => resolve(blob), "image/jpeg", 0.6);
    });

    if (!imageBlob) return;

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: await blobToBase64(imageBlob) }),
      });

      if (!response.ok) throw new Error(`Error: ${response.status}`);

      const data = await response.json();
      setLandmarks(data.landmarks);

      if (data.prediction && data.prediction !== "...") {
        const confidenceDisplay = data.confidence
          ? `(${Math.round(data.confidence)}%)`
          : "";
        const newPrediction = `${data.prediction} ${confidenceDisplay}`;
        setPredictions((prev) =>
          prev[0] === newPrediction
            ? prev
            : [newPrediction, ...prev.slice(0, 4)]
        );
      }
    } catch (error) {
      console.error("Prediction error:", error);
    }
  }, []); // Empty dependency array makes this function stable

  const toggleCapture = useCallback(async () => {
    if (!videoLoaded) return;

    // If we are about to stop capturing, call the reset API
    if (isCapturing) {
      try {
        const response = await fetch("/api/predict?reset=true", {
          method: "POST",
        }); //
        if (!response.ok)
          throw new Error(`Failed to reset sequence: ${response.status}`);
        setPredictions([]);
        setLandmarks(null);
      } catch (error) {
        console.error("Error resetting sequence:", error);
      }
    }
    // Toggle the state. The useEffect hook below will handle the rest.
    setIsCapturing((prev) => !prev);
  }, [isCapturing, videoLoaded]);

  // Main logic for starting/stopping the capture loop is now in useEffect
  useEffect(() => {
    if (!isCapturing) {
      if (captureLoopRef.current) {
        cancelAnimationFrame(captureLoopRef.current);
      }
      return;
    }

    if (!videoLoaded) return;

    // Reset FPS counter on start
    frameTimesRef.current = [];

    const captureLoop = async () => {
      await captureFrame();
      captureLoopRef.current = requestAnimationFrame(captureLoop);
    };

    captureLoopRef.current = requestAnimationFrame(captureLoop);

    return () => {
      if (captureLoopRef.current) {
        cancelAnimationFrame(captureLoopRef.current);
      }
    };
  }, [isCapturing, videoLoaded, captureFrame]);

  // Initialize camera
  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30, max: 30 },
            facingMode: "user",
          },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => setVideoLoaded(true);
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
      }
    };

    initCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((track) => track.stop());
      }
    };
  }, []);

  // Landmark drawing logic
  useEffect(() => {
    if (!landmarks) return;

    const landmarkCanvas = landmarkCanvasRef.current;
    if (!landmarkCanvas) return;

    if (!offscreenCanvas.current) {
      offscreenCanvas.current = document.createElement("canvas");
    }

    const offscreenCtx = offscreenCanvas.current.getContext("2d", {
      alpha: true,
    });
    const ctx = landmarkCanvas.getContext("2d", { alpha: true });

    if (!ctx || !offscreenCtx) return;

    if (
      videoRef.current &&
      landmarkCanvas.width !== videoRef.current.videoWidth //
    ) {
      const { videoWidth, videoHeight } = videoRef.current;
      landmarkCanvas.width = videoWidth;
      landmarkCanvas.height = videoHeight;
      offscreenCanvas.current.width = videoWidth;
      offscreenCanvas.current.height = videoHeight;
    }

    const drawFrame = () => {
      if (!offscreenCtx || !ctx) return;

      offscreenCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

      // --- NEW: Function to draw connectors ---
      const drawConnectors = (
        points: Array<{ x: number; y: number }>,
        connections: Array<[number, number]>,
        color: string
      ) => {
        offscreenCtx.strokeStyle = color;
        offscreenCtx.lineWidth = 3;
        for (const connection of connections) {
          const start = points[connection[0]];
          const end = points[connection[1]];
          if (start && end) {
            offscreenCtx.beginPath();
            offscreenCtx.moveTo(start.x, start.y);
            offscreenCtx.lineTo(end.x, end.y);
            offscreenCtx.stroke();
          }
        }
      };

      const drawHand = (
        hand: Array<{ x: number; y: number }>,
        color: string
      ) => {
        if (!hand?.length || !offscreenCtx) return;
        offscreenCtx.fillStyle = color; //

        const coords = hand.map((point) => ({
          x: point.x * landmarkCanvas.width,
          y: point.y * landmarkCanvas.height,
        }));

        // --- NEW: Draw connections first ---
        if (HAND_CONNECTIONS) {
          drawConnectors(coords, HAND_CONNECTIONS, color);
        }

        // Draw landmark points on top of lines
        coords.forEach((point) => {
          //
          offscreenCtx.beginPath();
          offscreenCtx.arc(point.x, point.y, 5, 0, 2 * Math.PI); // Slightly larger dots
          offscreenCtx.fill();
        });
      };

      if (landmarks?.leftHand) {
        drawHand(landmarks.leftHand, "#ff00ff"); //
      }
      if (landmarks?.rightHand) {
        drawHand(landmarks.rightHand, "#00ff00"); //
      }

      ctx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
      if (offscreenCanvas.current) {
        ctx.drawImage(offscreenCanvas.current, 0, 0); //
      }
    };

    const animationFrameId = requestAnimationFrame(drawFrame); //
    return () => cancelAnimationFrame(animationFrameId);
  }, [landmarks]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-gray-900 p-4 sm:p-8 text-white ">
      <div className="max-w-6xl mx-auto mt-[100px]">
        {/* --- Enhanced Header --- */}
        <div className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-extrabold bg-gradient-to-r from-cyan-400 to-sky-500 bg-clip-text text-transparent pb-2">
            Real-Time Sign Recognition
          </h1>
          <p className="text-slate-400 text-lg">
            Point your camera and start signing
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
          {/* --- Video Display Column --- */}
          <div className="lg:col-span-3 relative group">
            <div className="relative aspect-video bg-slate-800 rounded-2xl overflow-hidden border-2 border-slate-700 shadow-2xl shadow-cyan-500/10 transition-all duration-300 group-hover:border-cyan-400/50">
              <video
                ref={videoRef} //
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              <canvas
                ref={landmarkCanvasRef} //
                className="absolute top-0 left-0 w-full h-full"
                style={{ pointerEvents: "none" }}
              />
              <canvas ref={canvasRef} style={{ display: "none" }} />

              {/* --- Enhanced Capture Button --- */}
              <button
                onClick={toggleCapture} //
                className={`absolute bottom-4 right-4 px-5 py-2.5 rounded-lg font-semibold
                  transition-all duration-300 flex items-center gap-2 transform hover:scale-105
                  ${
                    isCapturing
                      ? "bg-red-600 hover:bg-red-500 shadow-lg shadow-red-500/30" //
                      : "bg-cyan-600 hover:bg-cyan-500 shadow-lg shadow-cyan-500/30" //
                  } text-white`}
                disabled={!videoLoaded} //
              >
                {isLoading ? ( //
                  <>
                    <svg
                      className="animate-spin h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Processing...
                  </>
                ) : isCapturing ? (
                  "Stop Capture" //
                ) : (
                  "Start Capture" //
                )}
              </button>
            </div>
          </div>

          {/* --- Predictions Column --- */}
          <div className="lg:col-span-2 bg-slate-800/50 p-6 rounded-2xl border border-slate-700 shadow-xl shadow-cyan-500/5">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-2xl font-bold text-cyan-400">
                Live Predictions
              </h2>
              {isCapturing && (
                <div className="flex items-center space-x-2 px-3 py-1 bg-slate-700/50 rounded-full">
                  <div
                    className={`w-2.5 h-2.5 rounded-full ${
                      fps >= 20 ? "bg-green-400" : "bg-yellow-400" //
                    } animate-pulse`}
                  />
                  <span className="text-sm font-mono text-cyan-300">
                    {fps} FPS
                  </span>
                </div>
              )}
            </div>

            <div className="space-y-3 h-96 overflow-y-auto pr-2">
              {predictions.length > 0 ? (
                predictions.map((pred, i) => (
                  <div
                    key={i}
                    // --- Added animation for new predictions ---
                    className="p-4 bg-slate-700/50 rounded-lg text-cyan-200 border-l-4 border-cyan-400/60 transition-all duration-300 animate-fade-in"
                  >
                    {pred}
                  </div>
                ))
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center p-4 bg-slate-700/30 rounded-lg text-slate-400 border border-dashed border-slate-600">
                    {isCapturing
                      ? "Awaiting prediction..."
                      : "Start capture to begin."}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
