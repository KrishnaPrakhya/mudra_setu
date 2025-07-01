"use client";

import type React from "react";
import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Brain,
  Camera,
  FileText,
  Zap,
  Video,
  ImageIcon,
  Play,
  Square,
  Download,
  Settings,
  BarChart3,
  Clock,
  Target,
  TrendingUp,
  CheckCircle2,
  Upload,
  Network,
  Cpu,
  Activity,
  Eye,
  ArrowRight,
} from "lucide-react";

interface PredictionResult {
  gesture: string;
  confidence: number;
  timestamp: Date;
  processingTime: number;
  neuralActivation: number[];
}

// Neural network visualization component
function NeuralNetworkViz({ isActive }: { isActive: boolean }) {
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {Array.from({ length: 15 }).map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1 h-1 bg-cyan-400/30 rounded-full"
          animate={
            isActive
              ? {
                  x: [0, Math.random() * 200 - 100],
                  y: [0, Math.random() * 200 - 100],
                  scale: [1, 1.5, 1],
                  opacity: [0.3, 0.8, 0.3],
                }
              : {}
          }
          transition={{
            duration: 2,
            repeat: Number.POSITIVE_INFINITY,
            delay: i * 0.1,
          }}
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
        />
      ))}
    </div>
  );
}

// Holographic display component
function HolographicDisplay({
  children,
  isActive,
}: {
  children: React.ReactNode;
  isActive: boolean;
}) {
  return (
    <div className="relative">
      <motion.div
        className="absolute inset-0 bg-gradient-to-r from-cyan-400/10 to-purple-400/10 rounded-lg"
        animate={
          isActive
            ? {
                opacity: [0.1, 0.3, 0.1],
                scale: [1, 1.02, 1],
              }
            : {}
        }
        transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY }}
      />
      <div className="relative z-10 bg-slate-900/50 backdrop-blur-xl border border-cyan-400/30 rounded-lg">
        {children}
      </div>
    </div>
  );
}

export default function PredictPage() {
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [realTimePrediction, setRealTimePrediction] = useState("");
  const [realTimeConfidence, setRealTimeConfidence] = useState(0);
  const [activeTab, setActiveTab] = useState("neural-cam");
  const [neuralActivity, setNeuralActivity] = useState<number[]>([]);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Holds the interval ID for the fake real-time prediction loop so we can
  // stop it later without storing it on the global `window` object.
  const predictionIntervalRef = useRef<ReturnType<typeof setInterval> | null>(
    null
  );

  // Start webcam
  const startWebcam = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: "user" },
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (error) {
      console.error("Error accessing webcam:", error);
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  // Start real-time prediction
  const startRealTimePrediction = () => {
    setIsRecording(true);

    // Simulate real-time predictions with neural activity
    const interval = setInterval(() => {
      const gestures = [
        "Hello",
        "Thank You",
        "Please",
        "Sorry",
        "Yes",
        "No",
        "Help",
        "Love",
        "Peace",
        "Stop",
      ];
      const randomGesture =
        gestures[Math.floor(Math.random() * gestures.length)];
      const randomConfidence = Math.floor(Math.random() * 25) + 75;
      const neuralActivation = Array.from({ length: 10 }, () => Math.random());

      setRealTimePrediction(randomGesture);
      setRealTimeConfidence(randomConfidence);
      setNeuralActivity(neuralActivation);

      // Add to predictions history
      const newPrediction: PredictionResult = {
        gesture: randomGesture,
        confidence: randomConfidence,
        timestamp: new Date(),
        processingTime: Math.random() * 50 + 25,
        neuralActivation,
      };

      setPredictions((prev) => [newPrediction, ...prev.slice(0, 9)]);
    }, 800);

    // Store interval ID for cleanup
    predictionIntervalRef.current = interval;
  };

  const stopRealTimePrediction = () => {
    setIsRecording(false);
    setNeuralActivity([]);
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
    }
  };

  useEffect(() => {
    if (activeTab === "neural-cam") {
      startWebcam();
    } else {
      stopWebcam();
      stopRealTimePrediction();
    }

    return () => {
      stopWebcam();
      stopRealTimePrediction();
    };
  }, [activeTab]);

  const handleModelUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setModelFile(e.target.files[0]);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0]);
    }
  };

  const handlePredict = async () => {
    if (!modelFile || !imageFile) return;

    setIsLoading(true);

    // Simulate prediction process with neural activity
    setTimeout(() => {
      const gestures = [
        "Hello",
        "Thank You",
        "Please",
        "Sorry",
        "Yes",
        "No",
        "Help",
      ];
      const randomGesture =
        gestures[Math.floor(Math.random() * gestures.length)];
      const randomConfidence = Math.floor(Math.random() * 25) + 75;

      setPrediction(randomGesture);
      setConfidence(randomConfidence);
      setIsLoading(false);
    }, 3500);
  };

  const averageConfidence =
    predictions.length > 0
      ? predictions.reduce((sum, p) => sum + p.confidence, 0) /
        predictions.length
      : 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-cyan-950 relative overflow-hidden">
      {/* Animated background */}
      <div className="fixed inset-0 pointer-events-none">
        {Array.from({ length: 30 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400/20 rounded-full"
            animate={{
              x: [0, Math.random() * window.innerWidth],
              y: [0, Math.random() * window.innerHeight],
            }}
            transition={{
              duration: Math.random() * 20 + 10,
              repeat: Number.POSITIVE_INFINITY,
              ease: "linear",
            }}
            style={{
              left: Math.random() * window.innerWidth,
              top: Math.random() * window.innerHeight,
            }}
          />
        ))}
      </div>

      <div className="pt-24 pb-12 px-6 relative z-10">
        <div className="container mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-12"
          >
            <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-cyan-300 via-purple-300 to-pink-300 bg-clip-text text-transparent">
              Neural Recognition Engine
            </h1>
            <p className="text-cyan-100/70 text-xl max-w-4xl mx-auto">
              Advanced AI-powered gesture classification with real-time neural
              network visualization and deep learning analysis
            </p>
          </motion.div>

          {/* Model Upload Section */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="mb-8"
          >
            <HolographicDisplay isActive={!!modelFile}>
              <Card className="bg-transparent border-0">
                <CardHeader>
                  <CardTitle className="flex items-center text-cyan-100">
                    <Brain className="mr-3 h-7 w-7 text-cyan-400" />
                    Neural Model Configuration
                    <motion.div
                      className="ml-auto"
                      animate={{ rotate: 360 }}
                      transition={{
                        duration: 4,
                        repeat: Number.POSITIVE_INFINITY,
                        ease: "linear",
                      }}
                    >
                      <Cpu className="h-6 w-6 text-purple-400" />
                    </motion.div>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid md:grid-cols-3 gap-6">
                    <div className="md:col-span-2">
                      <Label
                        htmlFor="model-upload"
                        className="text-cyan-100/90 mb-3 block text-lg"
                      >
                        Upload Deep Learning Model
                      </Label>
                      <div className="relative">
                        <Input
                          id="model-upload"
                          type="file"
                          accept=".h5,.pkl,.pt,.onnx,.tflite,.pb"
                          onChange={handleModelUpload}
                          className="bg-slate-800/50 border-cyan-400/30 text-cyan-100 file:bg-gradient-to-r file:from-cyan-600 file:to-purple-600 file:text-white file:border-0 file:rounded-md file:px-4 file:py-2"
                        />
                        <NeuralNetworkViz isActive={!!modelFile} />
                      </div>
                      {modelFile && (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="mt-4 p-4 bg-gradient-to-r from-green-500/10 to-cyan-500/10 rounded-lg border border-green-400/30"
                        >
                          <div className="flex items-center text-green-400 mb-2">
                            <CheckCircle2 className="mr-2 h-5 w-5" />
                            <span className="font-semibold">
                              Neural Model Loaded
                            </span>
                          </div>
                          <div className="text-cyan-100/80 text-sm">
                            {modelFile.name}
                          </div>
                          <div className="text-cyan-100/60 text-xs mt-1">
                            Size: {(modelFile.size / (1024 * 1024)).toFixed(2)}{" "}
                            MB
                          </div>
                        </motion.div>
                      )}
                    </div>

                    <div className="space-y-4">
                      <div className="grid grid-cols-1 gap-4">
                        <div className="bg-gradient-to-br from-cyan-500/10 to-purple-500/10 p-4 rounded-lg border border-cyan-400/20">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-cyan-100/70 text-sm">
                              Neural Status
                            </span>
                            <Activity className="h-4 w-4 text-cyan-400" />
                          </div>
                          <p className="text-cyan-100 font-semibold flex items-center">
                            {modelFile ? (
                              <>
                                <motion.div
                                  className="w-2 h-2 bg-green-400 rounded-full mr-2"
                                  animate={{
                                    scale: [1, 1.5, 1],
                                    opacity: [1, 0.5, 1],
                                  }}
                                  transition={{
                                    duration: 1,
                                    repeat: Number.POSITIVE_INFINITY,
                                  }}
                                />
                                Active
                              </>
                            ) : (
                              <>
                                <div className="w-2 h-2 bg-yellow-400 rounded-full mr-2" />
                                Standby
                              </>
                            )}
                          </p>
                        </div>
                        <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 p-4 rounded-lg border border-purple-400/20">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-cyan-100/70 text-sm">
                              Accuracy
                            </span>
                            <Target className="h-4 w-4 text-purple-400" />
                          </div>
                          <p className="text-cyan-100 font-semibold">96.8%</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </HolographicDisplay>
          </motion.div>

          {/* Main Prediction Interface */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Tabs
              value={activeTab}
              onValueChange={setActiveTab}
              className="space-y-8"
            >
              <TabsList className="grid w-full grid-cols-2 bg-slate-800/50 border border-cyan-400/20 backdrop-blur-xl">
                <TabsTrigger
                  value="neural-cam"
                  className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-cyan-600 data-[state=active]:to-purple-600 text-cyan-100"
                >
                  <Video className="mr-2 h-5 w-5" />
                  Neural Webcam
                </TabsTrigger>
                <TabsTrigger
                  value="deep-analysis"
                  className="data-[state=active]:bg-gradient-to-r data-[state=active]:from-purple-600 data-[state=active]:to-pink-600 text-cyan-100"
                >
                  <ImageIcon className="mr-2 h-5 w-5" />
                  Deep Analysis
                </TabsTrigger>
              </TabsList>

              <TabsContent value="neural-cam" className="space-y-8">
                <div className="grid lg:grid-cols-3 gap-8">
                  {/* Webcam Feed */}
                  <div className="lg:col-span-2">
                    <HolographicDisplay isActive={isRecording}>
                      <Card className="bg-transparent border-0">
                        <CardHeader>
                          <CardTitle className="flex items-center justify-between text-cyan-100">
                            <span className="flex items-center">
                              <Camera className="mr-3 h-6 w-6 text-cyan-400" />
                              Neural Vision Feed
                            </span>
                            <div className="flex items-center space-x-3">
                              {isRecording && (
                                <motion.div
                                  animate={{
                                    scale: [1, 1.2, 1],
                                    opacity: [1, 0.5, 1],
                                  }}
                                  transition={{
                                    duration: 1,
                                    repeat: Number.POSITIVE_INFINITY,
                                  }}
                                >
                                  <Badge className="bg-gradient-to-r from-red-500 to-pink-500 text-white">
                                    <div className="w-2 h-2 bg-white rounded-full mr-2"></div>
                                    NEURAL ACTIVE
                                  </Badge>
                                </motion.div>
                              )}
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{
                                  duration: 3,
                                  repeat: Number.POSITIVE_INFINITY,
                                  ease: "linear",
                                }}
                              >
                                <Network className="h-5 w-5 text-purple-400" />
                              </motion.div>
                            </div>
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="p-0">
                          <div className="relative">
                            <video
                              ref={videoRef}
                              autoPlay
                              playsInline
                              muted
                              className="w-full h-[500px] object-cover rounded-lg bg-slate-900"
                            />
                            <canvas
                              ref={canvasRef}
                              className="absolute top-0 left-0 w-full h-full pointer-events-none"
                            />

                            {/* Neural overlay */}
                            <div className="absolute inset-0 pointer-events-none">
                              {isRecording && (
                                <>
                                  {/* Neural grid */}
                                  <div className="absolute inset-0 opacity-20">
                                    <div className="grid grid-cols-8 grid-rows-6 h-full w-full">
                                      {Array.from({ length: 48 }).map(
                                        (_, i) => (
                                          <motion.div
                                            key={i}
                                            className="border border-cyan-400/30"
                                            animate={{
                                              opacity: [0.1, 0.5, 0.1],
                                            }}
                                            transition={{
                                              duration: 2,
                                              repeat: Number.POSITIVE_INFINITY,
                                              delay: i * 0.05,
                                            }}
                                          />
                                        )
                                      )}
                                    </div>
                                  </div>

                                  {/* Neural activity indicators */}
                                  {neuralActivity.map((activity, i) => (
                                    <motion.div
                                      key={i}
                                      className="absolute w-3 h-3 bg-cyan-400 rounded-full"
                                      style={{
                                        left: `${10 + (i % 8) * 10}%`,
                                        top: `${10 + Math.floor(i / 8) * 15}%`,
                                      }}
                                      animate={{
                                        scale: [1, 1 + activity, 1],
                                        opacity: [0.3, activity, 0.3],
                                      }}
                                      transition={{ duration: 0.5 }}
                                    />
                                  ))}
                                </>
                              )}
                            </div>

                            {/* Real-time prediction overlay */}
                            <AnimatePresence>
                              {isRecording && realTimePrediction && (
                                <motion.div
                                  initial={{ opacity: 0, scale: 0.8, y: 20 }}
                                  animate={{ opacity: 1, scale: 1, y: 0 }}
                                  exit={{ opacity: 0, scale: 0.8, y: -20 }}
                                  className="absolute top-6 left-6 bg-gradient-to-r from-slate-900/90 to-purple-900/90 backdrop-blur-xl rounded-xl p-6 border border-cyan-400/30"
                                >
                                  <div className="text-cyan-100">
                                    <p className="text-sm text-cyan-300/70 mb-1">
                                      Neural Detection
                                    </p>
                                    <p className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                                      {realTimePrediction}
                                    </p>
                                    <div className="flex items-center mt-3">
                                      <div className="w-24 bg-slate-700/50 rounded-full h-2 mr-3">
                                        <motion.div
                                          className="bg-gradient-to-r from-cyan-400 to-purple-400 h-2 rounded-full"
                                          initial={{ width: 0 }}
                                          animate={{
                                            width: `${realTimeConfidence}%`,
                                          }}
                                          transition={{ duration: 0.5 }}
                                        />
                                      </div>
                                      <span className="text-sm text-cyan-300/70 font-mono">
                                        {realTimeConfidence}%
                                      </span>
                                    </div>
                                  </div>
                                </motion.div>
                              )}
                            </AnimatePresence>
                          </div>

                          <div className="p-6 flex justify-center space-x-4">
                            {!isRecording ? (
                              <Button
                                onClick={startRealTimePrediction}
                                disabled={!stream || !modelFile}
                                className="bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-700 hover:to-purple-700 px-8 py-3 text-lg"
                              >
                                <Play className="mr-3 h-5 w-5" />
                                Activate Neural Recognition
                              </Button>
                            ) : (
                              <Button
                                onClick={stopRealTimePrediction}
                                className="bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 px-8 py-3 text-lg"
                              >
                                <Square className="mr-3 h-5 w-5" />
                                Deactivate Neural System
                              </Button>
                            )}

                            <Button
                              variant="outline"
                              className="border-cyan-400/30 text-cyan-300 hover:bg-cyan-400/10 px-6 py-3"
                            >
                              <Settings className="mr-2 h-5 w-5" />
                              Neural Config
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    </HolographicDisplay>
                  </div>

                  {/* Real-time Stats */}
                  <div className="space-y-6">
                    <HolographicDisplay isActive={isRecording}>
                      <Card className="bg-transparent border-0">
                        <CardHeader>
                          <CardTitle className="flex items-center text-cyan-100">
                            <BarChart3 className="mr-2 h-6 w-6 text-cyan-400" />
                            Neural Analytics
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-6">
                          <div className="grid grid-cols-2 gap-4">
                            <div className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 p-4 rounded-lg border border-cyan-400/20">
                              <p className="text-cyan-100/70 text-sm mb-1">
                                Predictions
                              </p>
                              <p className="text-2xl font-bold text-cyan-100">
                                {predictions.length}
                              </p>
                            </div>
                            <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 p-4 rounded-lg border border-purple-400/20">
                              <p className="text-cyan-100/70 text-sm mb-1">
                                Avg Confidence
                              </p>
                              <p className="text-2xl font-bold text-cyan-100">
                                {averageConfidence.toFixed(1)}%
                              </p>
                            </div>
                          </div>

                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <span className="text-cyan-100/70 text-sm">
                                Neural Processing
                              </span>
                              <Clock className="h-4 w-4 text-cyan-400" />
                            </div>
                            <div className="flex items-center">
                              <motion.div
                                className="w-2 h-2 bg-green-400 rounded-full mr-2"
                                animate={{ scale: [1, 1.5, 1] }}
                                transition={{
                                  duration: 1,
                                  repeat: Number.POSITIVE_INFINITY,
                                }}
                              />
                              <span className="text-cyan-100 font-mono">
                                ~45ms
                              </span>
                            </div>
                          </div>

                          {/* Neural activity visualization */}
                          <div className="space-y-3">
                            <span className="text-cyan-100/70 text-sm">
                              Neural Layer Activity
                            </span>
                            <div className="space-y-2">
                              {neuralActivity.slice(0, 5).map((activity, i) => (
                                <div
                                  key={i}
                                  className="flex items-center space-x-2"
                                >
                                  <span className="text-xs text-cyan-300/70 w-12">
                                    L{i + 1}
                                  </span>
                                  <div className="flex-1 bg-slate-700/50 rounded-full h-2">
                                    <motion.div
                                      className="bg-gradient-to-r from-cyan-400 to-purple-400 h-2 rounded-full"
                                      initial={{ width: 0 }}
                                      animate={{ width: `${activity * 100}%` }}
                                      transition={{ duration: 0.3 }}
                                    />
                                  </div>
                                  <span className="text-xs text-cyan-300/70 w-8">
                                    {(activity * 100).toFixed(0)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    </HolographicDisplay>

                    <HolographicDisplay isActive={predictions.length > 0}>
                      <Card className="bg-transparent border-0">
                        <CardHeader>
                          <CardTitle className="flex items-center text-cyan-100">
                            <TrendingUp className="mr-2 h-6 w-6 text-purple-400" />
                            Recent Detections
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="space-y-3 max-h-80 overflow-y-auto">
                            {predictions.map((pred, index) => (
                              <motion.div
                                key={index}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                className="flex items-center justify-between p-3 bg-gradient-to-r from-slate-800/50 to-purple-800/30 rounded-lg border border-cyan-400/20"
                              >
                                <div>
                                  <p className="text-cyan-100 font-medium">
                                    {pred.gesture}
                                  </p>
                                  <p className="text-cyan-300/50 text-xs font-mono">
                                    {pred.timestamp.toLocaleTimeString()}
                                  </p>
                                </div>
                                <div className="flex items-center space-x-2">
                                  <Badge
                                    className={`${
                                      pred.confidence > 85
                                        ? "bg-gradient-to-r from-green-500 to-emerald-500"
                                        : pred.confidence > 70
                                          ? "bg-gradient-to-r from-yellow-500 to-orange-500"
                                          : "bg-gradient-to-r from-red-500 to-pink-500"
                                    } text-white`}
                                  >
                                    {pred.confidence}%
                                  </Badge>
                                </div>
                              </motion.div>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    </HolographicDisplay>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="deep-analysis" className="space-y-8">
                <div className="grid lg:grid-cols-2 gap-8">
                  {/* Image Upload Section */}
                  <HolographicDisplay isActive={!!imageFile}>
                    <Card className="bg-transparent border-0">
                      <CardHeader>
                        <CardTitle className="flex items-center text-cyan-100">
                          <Upload className="mr-3 h-6 w-6 text-cyan-400" />
                          Deep Learning Analysis
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-6">
                        <div>
                          <Label
                            htmlFor="image-upload"
                            className="text-cyan-100/90 mb-3 block text-lg"
                          >
                            Upload Image for Analysis
                          </Label>
                          <div className="relative">
                            <Input
                              id="image-upload"
                              type="file"
                              accept="image/*"
                              onChange={handleImageUpload}
                              className="bg-slate-800/50 border-cyan-400/30 text-cyan-100 file:bg-gradient-to-r file:from-purple-600 file:to-pink-600 file:text-white file:border-0 file:rounded-md file:px-4 file:py-2"
                            />
                            <NeuralNetworkViz isActive={!!imageFile} />
                          </div>
                          {imageFile && (
                            <motion.div
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              className="mt-4"
                            >
                              <div className="bg-gradient-to-r from-green-500/10 to-cyan-500/10 p-4 rounded-lg border border-green-400/30">
                                <div className="flex items-center text-green-400 text-sm mb-2">
                                  <CheckCircle2 className="mr-2 h-4 w-4" />
                                  <span className="font-semibold">
                                    Image Loaded Successfully
                                  </span>
                                </div>
                                <div className="text-cyan-100/80 text-sm">
                                  {imageFile.name}
                                </div>
                                <div className="text-cyan-100/60 text-xs mt-1">
                                  Size: {(imageFile.size / 1024).toFixed(1)} KB
                                </div>
                              </div>
                            </motion.div>
                          )}
                        </div>

                        <Button
                          onClick={handlePredict}
                          disabled={!modelFile || !imageFile || isLoading}
                          className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white py-4 text-lg"
                        >
                          {isLoading ? (
                            <>
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{
                                  duration: 1,
                                  repeat: Number.POSITIVE_INFINITY,
                                  ease: "linear",
                                }}
                                className="mr-3"
                              >
                                <Zap className="h-6 w-6" />
                              </motion.div>
                              Neural Processing...
                            </>
                          ) : (
                            <>
                              <Brain className="mr-3 h-6 w-6" />
                              Analyze with Deep Learning
                            </>
                          )}
                        </Button>

                        <div className="grid grid-cols-3 gap-3 text-xs">
                          <div className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 p-3 rounded text-center border border-cyan-400/20">
                            <p className="text-cyan-100/70 mb-1">Supported</p>
                            <p className="text-cyan-100 font-semibold">
                              JPG, PNG, WebP
                            </p>
                          </div>
                          <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 p-3 rounded text-center border border-purple-400/20">
                            <p className="text-cyan-100/70 mb-1">Max Size</p>
                            <p className="text-cyan-100 font-semibold">25MB</p>
                          </div>
                          <div className="bg-gradient-to-br from-pink-500/10 to-cyan-500/10 p-3 rounded text-center border border-pink-400/20">
                            <p className="text-cyan-100/70 mb-1">Resolution</p>
                            <p className="text-cyan-100 font-semibold">Any</p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </HolographicDisplay>

                  {/* Results Section */}
                  <HolographicDisplay isActive={!!prediction}>
                    <Card className="bg-transparent border-0">
                      <CardHeader>
                        <CardTitle className="flex items-center text-cyan-100">
                          <Target className="mr-3 h-6 w-6 text-green-400" />
                          Deep Analysis Results
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-6">
                        {prediction ? (
                          <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="space-y-6"
                          >
                            <div className="text-center p-8 bg-gradient-to-r from-green-500/10 to-cyan-500/10 rounded-xl border border-green-400/30">
                              <motion.h3
                                className="text-xl font-bold text-cyan-100 mb-3"
                                animate={{ scale: [1, 1.05, 1] }}
                                transition={{
                                  duration: 2,
                                  repeat: Number.POSITIVE_INFINITY,
                                }}
                              >
                                Neural Classification
                              </motion.h3>
                              <p className="text-5xl font-bold bg-gradient-to-r from-green-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
                                {prediction}
                              </p>
                            </div>

                            <div className="space-y-3">
                              <div className="flex justify-between text-cyan-100/90 mb-2">
                                <span>Neural Confidence</span>
                                <span className="font-mono">{confidence}%</span>
                              </div>
                              <div className="relative">
                                <Progress
                                  value={confidence}
                                  className="h-4 bg-slate-700/50"
                                />
                                <motion.div
                                  className="absolute inset-0 bg-gradient-to-r from-cyan-400/20 to-purple-400/20 rounded-full"
                                  animate={{ opacity: [0.2, 0.5, 0.2] }}
                                  transition={{
                                    duration: 2,
                                    repeat: Number.POSITIVE_INFINITY,
                                  }}
                                />
                              </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                              <div className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 p-4 rounded-lg border border-cyan-400/20">
                                <p className="text-cyan-100/70 text-sm mb-1">
                                  Processing Time
                                </p>
                                <p className="text-cyan-100 font-semibold text-lg font-mono">
                                  2.8s
                                </p>
                              </div>
                              <div className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 p-4 rounded-lg border border-purple-400/20">
                                <p className="text-cyan-100/70 text-sm mb-1">
                                  Model Accuracy
                                </p>
                                <p className="text-cyan-100 font-semibold text-lg">
                                  96.8%
                                </p>
                              </div>
                            </div>

                            <Separator className="bg-cyan-400/20" />

                            <div className="space-y-4">
                              <h4 className="text-cyan-100 font-medium flex items-center">
                                <Network className="mr-2 h-4 w-4 text-purple-400" />
                                Alternative Neural Predictions
                              </h4>
                              {["Peace", "Thank You", "Hello"].map(
                                (alt, index) => (
                                  <motion.div
                                    key={index}
                                    className="flex items-center justify-between p-3 bg-gradient-to-r from-slate-800/50 to-purple-800/30 rounded-lg border border-cyan-400/20"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: index * 0.1 }}
                                  >
                                    <span className="text-cyan-100/80">
                                      {alt}
                                    </span>
                                    <Badge className="bg-gradient-to-r from-slate-600 to-purple-600 text-cyan-100">
                                      {Math.floor(Math.random() * 30 + 15)}%
                                    </Badge>
                                  </motion.div>
                                )
                              )}
                            </div>

                            <div className="flex space-x-3">
                              <Button
                                size="sm"
                                className="bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-700 hover:to-purple-700"
                              >
                                <Download className="mr-2 h-4 w-4" />
                                Export Results
                              </Button>
                              <Button
                                variant="outline"
                                size="sm"
                                className="border-cyan-400/30 text-cyan-300 hover:bg-cyan-400/10"
                              >
                                <FileText className="mr-2 h-4 w-4" />
                                Detailed Report
                              </Button>
                            </div>
                          </motion.div>
                        ) : (
                          <div className="text-center py-16">
                            <motion.div
                              className="w-32 h-32 mx-auto mb-6 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-full flex items-center justify-center border border-cyan-400/20"
                              animate={{
                                scale: [1, 1.05, 1],
                                rotate: [0, 5, -5, 0],
                              }}
                              transition={{
                                duration: 4,
                                repeat: Number.POSITIVE_INFINITY,
                              }}
                            >
                              <Brain className="h-16 w-16 text-cyan-400" />
                            </motion.div>
                            <p className="text-cyan-100/70 text-lg mb-2">
                              Upload an image to begin neural analysis
                            </p>
                            <p className="text-cyan-100/50 text-sm">
                              Advanced deep learning models ready for processing
                            </p>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </HolographicDisplay>
                </div>
              </TabsContent>
            </Tabs>
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="mt-12 flex justify-center space-x-6"
          >
            <Link href="/visualize">
              <Button
                variant="outline"
                className="border-cyan-400/30 text-cyan-300 hover:bg-cyan-400/10 px-8 py-3"
              >
                <Eye className="mr-2 h-5 w-5" />
                Visualize in 3D Space
                <ArrowRight className="ml-2 h-5 w-5 rotate-45" />
              </Button>
            </Link>
            <Link href="/dashboard">
              <Button
                variant="outline"
                className="border-purple-400/30 text-purple-300 hover:bg-purple-400/10 px-8 py-3"
              >
                <BarChart3 className="mr-2 h-5 w-5" />
                Neural Analytics
              </Button>
            </Link>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
