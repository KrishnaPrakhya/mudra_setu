"use client";
import { useEffect, useRef } from "react";
import { DrawingUtils, HolisticLandmarker } from "@mediapipe/tasks-vision";
import PredictionPanel from "./_components/PredictionPanel";
import { useSignLanguageRecognition } from "./_components/UseSignLanguageRecognition";
import VideoControls from "./_components/VideoControls";
import StatsPanel from "./_components/StatsPanel";
import SettingsPanel from "./_components/SettingsPanel";

export default function PredictPage() {
  const {
    videoRef,
    landmarks,
    predictions,
    aggregatedPredictions,
    isCapturing,
    toggleCapture,
    isBuffering,
    progress,
    speakPredictions,
    clearPredictions,
    exportPredictions,
    stats,
    settings,
    updateSettings,
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
    canvasCtx.scale(-1, 1);
    canvasCtx.translate(-canvas.width, 0);

    const drawingUtils = new DrawingUtils(canvasCtx);

    if (landmarks.poseLandmarks) {
      drawingUtils.drawConnectors(
        landmarks.poseLandmarks[0],
        HolisticLandmarker.POSE_CONNECTIONS,
        {
          color: settings.poseColor,
        }
      );
      drawingUtils.drawLandmarks(landmarks.poseLandmarks[0], {
        color: settings.poseColor,
        lineWidth: 2,
      });
    }
    if (landmarks.leftHandLandmarks) {
      drawingUtils.drawConnectors(
        landmarks.leftHandLandmarks[0],
        HolisticLandmarker.HAND_CONNECTIONS,
        {
          color: settings.leftHandColor,
        }
      );
      drawingUtils.drawLandmarks(landmarks.leftHandLandmarks[0], {
        color: settings.leftHandColor,
        lineWidth: 2,
      });
    }
    if (landmarks.rightHandLandmarks) {
      drawingUtils.drawConnectors(
        landmarks.rightHandLandmarks[0],
        HolisticLandmarker.HAND_CONNECTIONS,
        {
          color: settings.rightHandColor,
        }
      );
      drawingUtils.drawLandmarks(landmarks.rightHandLandmarks[0], {
        color: settings.rightHandColor,
        lineWidth: 2,
      });
    }
    canvasCtx.restore();
  }, [landmarks, videoRef, settings]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      {/* Header */}
      <div className="top-24 pt-[100px]">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                Sign Language Recognition Studio
              </h1>
              <p className="text-slate-400 text-sm">
                Real-time AI-powered sign detection
              </p>
            </div>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div
                  className={`w-3 h-3 rounded-full ${
                    isCapturing ? "bg-green-500 animate-pulse" : "bg-red-500"
                  }`}
                />
                <span className="text-sm font-medium">
                  {isCapturing ? "LIVE" : "OFFLINE"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-4 space-y-6">
        {/* Main Content Grid */}
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          {/* Video Section */}
          <div className="xl:col-span-8 space-y-4">
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-cyan-600/20 rounded-2xl blur-xl" />
              <div className="relative bg-slate-800/50 backdrop-blur-sm rounded-2xl border border-purple-500/20 overflow-hidden">
                <div className="aspect-video relative">
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

                  {/* Video Overlay */}
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-transparent" />

                  {/* Status Indicators */}
                  <div className="absolute top-4 left-4 flex gap-2">
                    {isCapturing && (
                      <div className="bg-red-500/90 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-medium flex items-center gap-2">
                        <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                        RECORDING
                      </div>
                    )}
                    {isBuffering && (
                      <div className="bg-yellow-500/90 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-medium">
                        BUFFERING {Math.round(progress * 100)}%
                      </div>
                    )}
                  </div>
                </div>

                {/* Video Controls */}
                <VideoControls
                  isCapturing={isCapturing}
                  toggleCapture={toggleCapture}
                  speakPredictions={speakPredictions}
                  clearPredictions={clearPredictions}
                  exportPredictions={exportPredictions}
                  aggregatedCount={aggregatedPredictions.length}
                />
              </div>
            </div>

            {/* Stats Panel */}
            <StatsPanel stats={stats} />
          </div>

          {/* Side Panel */}
          <div className="xl:col-span-4 space-y-6">
            {/* Predictions Panel */}
            <PredictionPanel
              predictions={predictions}
              aggregatedPredictions={aggregatedPredictions}
              isCapturing={isCapturing}
              isBuffering={isBuffering}
              progress={progress}
              fps={20}
            />

            {/* Settings Panel */}
            <SettingsPanel
              settings={settings}
              updateSettings={updateSettings}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
