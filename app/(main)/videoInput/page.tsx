"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

export default function VideoPredictPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [annotatedUrl, setAnnotatedUrl] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setProgress(0);
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/api/video-predict", {
      method: "POST",
      body: formData,
    });
    if (res.ok) {
      const data = await res.json();
      setPredictions(data.predictions);
      setAnnotatedUrl(data.annotated_video_url);
    }
    setUploading(false);
    setProgress(100);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle className="text-3xl font-bold text-center">
            Video Sign Prediction
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <Input type="file" accept="video/*" onChange={handleFileChange} />
            <Button onClick={handleUpload} disabled={!file || uploading}>
              {uploading ? "Uploading..." : "Upload & Predict"}
            </Button>
            {uploading && <Progress value={progress} />}
            {predictions.length > 0 && (
              <div className="mt-4">
                <h3 className="font-semibold mb-2">Frame-wise Predictions:</h3>
                <div className="flex flex-wrap gap-2">
                  {predictions.map((p, i) => (
                    <Badge key={i} variant="secondary">
                      Frame {p.frame}: {p.prediction} (
                      {(p.confidence * 100).toFixed(1)}%)
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            {annotatedUrl && (
              <div className="mt-4">
                <a
                  href={annotatedUrl}
                  className="underline text-blue-400"
                  download
                >
                  Download Annotated Video
                </a>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
