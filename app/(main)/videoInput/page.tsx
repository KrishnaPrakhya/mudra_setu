"use client";
import { useState, useRef, useCallback } from "react";
import type React from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Upload,
  Play,
  Download,
  FileVideo,
  Clock,
  Target,
  BarChart3,
  Trash2,
  RefreshCw,
  AlertCircle,
  CheckCircle,
} from "lucide-react";

interface Prediction {
  frame: number;
  prediction: string;
  confidence: number;
  timestamp?: number;
}

interface UploadStats {
  totalFrames: number;
  processingTime: number;
  averageConfidence: number;
  uniquePredictions: number;
}

export default function VideoPredictPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [annotatedUrl, setAnnotatedUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<UploadStats | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type.startsWith("video/")) {
        setFile(droppedFile);
        setError(null);
      } else {
        setError("Please upload a valid video file");
      }
    }
  }, []);

  const calculateStats = (predictions: Prediction[]): UploadStats => {
    const totalFrames = predictions.length;
    const averageConfidence =
      predictions.reduce((sum, p) => sum + p.confidence, 0) / totalFrames;
    const uniquePredictions = new Set(predictions.map((p) => p.prediction))
      .size;

    return {
      totalFrames,
      processingTime: 0, // This would come from backend
      averageConfidence,
      uniquePredictions,
    };
  };

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setProgress(0);
    setError(null);
    setPredictions([]);
    setAnnotatedUrl(null);
    setStats(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setProgress((prev) => Math.min(prev + Math.random() * 15, 90));
      }, 500);

      const res = await fetch("http://localhost:8000/api/video-predict", {
        method: "POST",
        body: formData,
      });

      clearInterval(progressInterval);

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const data = await res.json();
      setPredictions(data.predictions || []);
      setAnnotatedUrl(data.annotated_video_url);
      setStats(calculateStats(data.predictions || []));
      setProgress(100);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed");
      setProgress(0);
    } finally {
      setUploading(false);
    }
  };

  const clearResults = () => {
    setPredictions([]);
    setAnnotatedUrl(null);
    setStats(null);
    setProgress(0);
    setError(null);
  };

  const resetAll = () => {
    setFile(null);
    clearResults();
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (
      Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
    );
  };

  const getTopPredictions = () => {
    const predictionCounts: {
      [key: string]: { count: number; avgConfidence: number };
    } = {};

    predictions.forEach((p) => {
      if (!predictionCounts[p.prediction]) {
        predictionCounts[p.prediction] = { count: 0, avgConfidence: 0 };
      }
      predictionCounts[p.prediction].count++;
      predictionCounts[p.prediction].avgConfidence =
        (predictionCounts[p.prediction].avgConfidence *
          (predictionCounts[p.prediction].count - 1) +
          p.confidence) /
        predictionCounts[p.prediction].count;
    });

    return Object.entries(predictionCounts)
      .sort(([, a], [, b]) => b.count - a.count)
      .slice(0, 5);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="pt-[100px]">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                Video Sign Language Analysis
              </h1>
              <p className="text-slate-400 text-sm">
                Upload and analyze sign language videos
              </p>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={clearResults}
                variant="outline"
                size="sm"
                disabled={predictions.length === 0}
              >
                <Trash2 className="w-4 h-4 mr-2" />
                Clear Results
              </Button>
              <Button onClick={resetAll} variant="outline" size="sm">
                <RefreshCw className="w-4 h-4 mr-2" />
                Reset All
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-4 space-y-6">
        {/* Upload Section */}
        <Card className="bg-slate-800/50 backdrop-blur-sm border-purple-500/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-xl">
              <Upload className="w-6 h-6 text-purple-400" />
              Upload Video
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Drag & Drop Area */}
            <div
              className={`relative border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive
                  ? "border-purple-400 bg-purple-400/10"
                  : "border-slate-600 hover:border-purple-400/50 hover:bg-slate-700/30"
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <Input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileChange}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              <div className="space-y-4">
                <FileVideo className="w-12 h-12 mx-auto text-slate-400" />
                <div>
                  <p className="text-lg font-medium">
                    {dragActive
                      ? "Drop your video here"
                      : "Click to upload or drag and drop"}
                  </p>
                  <p className="text-sm text-slate-400 mt-1">
                    Supports MP4, AVI, MOV, and other video formats
                  </p>
                </div>
              </div>
            </div>

            {/* File Info */}
            {file && (
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <FileVideo className="w-5 h-5 text-purple-400" />
                    <div>
                      <p className="font-medium">{file.name}</p>
                      <p className="text-sm text-slate-400">
                        {formatFileSize(file.size)}
                      </p>
                    </div>
                  </div>
                  <Button
                    onClick={() => setFile(null)}
                    variant="ghost"
                    size="sm"
                    className="text-red-400 hover:text-red-300"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            )}

            {/* Upload Button */}
            <Button
              onClick={handleUpload}
              disabled={!file || uploading}
              size="lg"
              className="w-full bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600"
            >
              {uploading ? (
                <>
                  <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                  Processing Video...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5 mr-2" />
                  Analyze Video
                </>
              )}
            </Button>

            {/* Progress */}
            {uploading && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span>Processing...</span>
                  <span>{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} className="h-2" />
              </div>
            )}

            {/* Error */}
            {error && (
              <Alert className="border-red-500/50 bg-red-500/10">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription className="text-red-400">
                  {error}
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        {(predictions.length > 0 || stats) && (
          <div className="space-y-6">
            {/* Stats Cards */}
            {stats && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="bg-slate-800/50 backdrop-blur-sm border-purple-500/20">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="w-5 h-5 text-purple-400" />
                      <div>
                        <p className="text-sm text-slate-400">Total Frames</p>
                        <p className="text-2xl font-bold">
                          {stats.totalFrames}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-slate-800/50 backdrop-blur-sm border-cyan-500/20">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2">
                      <Target className="w-5 h-5 text-cyan-400" />
                      <div>
                        <p className="text-sm text-slate-400">Avg Confidence</p>
                        <p className="text-2xl font-bold">
                          {(stats.averageConfidence * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-slate-800/50 backdrop-blur-sm border-green-500/20">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="w-5 h-5 text-green-400" />
                      <div>
                        <p className="text-sm text-slate-400">Unique Signs</p>
                        <p className="text-2xl font-bold">
                          {stats.uniquePredictions}
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="bg-slate-800/50 backdrop-blur-sm border-yellow-500/20">
                  <CardContent className="p-4">
                    <div className="flex items-center gap-2">
                      <Clock className="w-5 h-5 text-yellow-400" />
                      <div>
                        <p className="text-sm text-slate-400">Processing</p>
                        <p className="text-2xl font-bold">Complete</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Results Tabs */}
            <Card className="bg-slate-800/50 backdrop-blur-sm border-slate-500/20">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Analysis Results</span>
                  {annotatedUrl && (
                    <Button asChild variant="outline" size="sm">
                      <a
                        href={annotatedUrl}
                        download
                        className="flex items-center gap-2"
                      >
                        <Download className="w-4 h-4" />
                        Download Video
                      </a>
                    </Button>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="timeline" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="timeline">Frame Timeline</TabsTrigger>
                    <TabsTrigger value="summary">Summary</TabsTrigger>
                    <TabsTrigger value="export">Export Data</TabsTrigger>
                  </TabsList>

                  <TabsContent value="timeline" className="space-y-4">
                    <ScrollArea className="h-96">
                      <div className="space-y-2">
                        {predictions.map((p, i) => (
                          <div
                            key={i}
                            className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg border border-slate-600/50"
                          >
                            <div className="flex items-center gap-3">
                              <Badge variant="outline" className="text-xs">
                                Frame {p.frame}
                              </Badge>
                              <span className="font-medium">
                                {p.prediction}
                              </span>
                            </div>
                            <div className="flex items-center gap-2">
                              <div className="text-sm text-slate-400">
                                {(p.confidence * 100).toFixed(1)}%
                              </div>
                              <div
                                className={`w-2 h-2 rounded-full ${
                                  p.confidence > 0.8
                                    ? "bg-green-500"
                                    : p.confidence > 0.6
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                                }`}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </TabsContent>

                  <TabsContent value="summary" className="space-y-4">
                    <div>
                      <h3 className="text-lg font-semibold mb-3">
                        Top Predictions
                      </h3>
                      <div className="space-y-2">
                        {getTopPredictions().map(([prediction, data], i) => (
                          <div
                            key={prediction}
                            className="flex items-center justify-between p-3 bg-slate-700/50 rounded-lg"
                          >
                            <div className="flex items-center gap-3">
                              <Badge variant="secondary">#{i + 1}</Badge>
                              <span className="font-medium">{prediction}</span>
                            </div>
                            <div className="text-right">
                              <div className="text-sm font-medium">
                                {data.count} occurrences
                              </div>
                              <div className="text-xs text-slate-400">
                                Avg: {(data.avgConfidence * 100).toFixed(1)}%
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="export" className="space-y-4">
                    <div className="text-center space-y-4">
                      <p className="text-slate-400">
                        Export your analysis results in different formats
                      </p>
                      <div className="flex gap-4 justify-center">
                        <Button
                          onClick={() => {
                            const data = JSON.stringify(
                              { predictions, stats },
                              null,
                              2
                            );
                            const blob = new Blob([data], {
                              type: "application/json",
                            });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `video-analysis-${Date.now()}.json`;
                            a.click();
                            URL.revokeObjectURL(url);
                          }}
                          variant="outline"
                        >
                          <Download className="w-4 h-4 mr-2" />
                          Export JSON
                        </Button>
                        <Button
                          onClick={() => {
                            const csv = [
                              "Frame,Prediction,Confidence",
                              ...predictions.map(
                                (p) =>
                                  `${p.frame},${p.prediction},${p.confidence}`
                              ),
                            ].join("\n");
                            const blob = new Blob([csv], { type: "text/csv" });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `video-analysis-${Date.now()}.csv`;
                            a.click();
                            URL.revokeObjectURL(url);
                          }}
                          variant="outline"
                        >
                          <Download className="w-4 h-4 mr-2" />
                          Export CSV
                        </Button>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
