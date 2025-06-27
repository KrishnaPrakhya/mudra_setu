"use client";
import { useEffect, useRef } from "react";
import { DrawingUtils, HolisticLandmarker } from "@mediapipe/tasks-vision";
import PredictionPanel from "./_components/PredictionPanel";
import { useSignLanguageRecognition } from "./_components/UseSignLanguageRecognition";

export default function PredictPage() {
  const {
    videoRef,
    landmarks,
    predictions,
    isCapturing,
    toggleCapture,
    isBuffering,
    progress,
  } = useSignLanguageRecognition();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const initCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            facingMode: "user",
          },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing camera:", error);
      }
    };
    initCamera();
  }, [videoRef]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video || !landmarks) {
      if (canvas) {
        const canvasCtx = canvas.getContext("2d");
        if (canvasCtx) canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const canvasCtx = canvas.getContext("2d");
    if (!canvasCtx) return;

    if (
      canvas.width !== video.clientWidth ||
      canvas.height !== video.clientHeight
    ) {
      canvas.width = video.clientWidth;
      canvas.height = video.clientHeight;
    }

    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    // Flip the canvas context horizontally
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvas.width, 0);

    const drawingUtils = new DrawingUtils(canvasCtx);

    if (landmarks.poseLandmarks) {
      drawingUtils.drawConnectors(
        landmarks.poseLandmarks[0],
        HolisticLandmarker.POSE_CONNECTIONS,
        { color: "#FF0000" }
      );
      drawingUtils.drawLandmarks(landmarks.poseLandmarks[0], {
        color: "#FF0000",
        lineWidth: 2,
      });
    }
    if (landmarks.leftHandLandmarks) {
      drawingUtils.drawConnectors(
        landmarks.leftHandLandmarks[0],
        HolisticLandmarker.HAND_CONNECTIONS,
        { color: "#00FF00" }
      );
      drawingUtils.drawLandmarks(landmarks.leftHandLandmarks[0], {
        color: "#00FF00",
        lineWidth: 2,
      });
    }
    if (landmarks.rightHandLandmarks) {
      drawingUtils.drawConnectors(
        landmarks.rightHandLandmarks[0],
        HolisticLandmarker.HAND_CONNECTIONS,
        { color: "#0000FF" }
      );
      drawingUtils.drawLandmarks(landmarks.rightHandLandmarks[0], {
        color: "#0000FF",
        lineWidth: 2,
      });
    }
    canvasCtx.restore();
  }, [landmarks, videoRef]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-gray-900 p-4 sm:p-8 text-white">
      <div className="max-w-6xl mx-auto mt-[100px]">
        <div className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-extrabold text-white">
            Real-Time Sign Recognition
          </h1>
          <p className="text-slate-400 text-lg">
            Backend-Annotated Video Stream
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
          <div className="lg:col-span-3 relative group">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover transform scale-x-[-1]"
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full"
            />
            <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-10">
              <button
                onClick={toggleCapture}
                className="px-6 py-3 bg-cyan-500 text-white font-semibold rounded-lg shadow-lg hover:bg-cyan-600 active:scale-95 transition-all"
              >
                {isCapturing ? "Stop Capture" : "Start Capture"}
              </button>
            </div>
          </div>

          <div className="lg:col-span-2">
            <PredictionPanel
              predictions={predictions}
              isCapturing={isCapturing}
              isBuffering={isBuffering}
              progress={progress}
              fps={20} // This could be made dynamic in the hook
            />
          </div>
        </div>
      </div>
    </div>
  );
}
