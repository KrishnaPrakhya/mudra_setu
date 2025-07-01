"use client";
import { memo } from "react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";

interface Prediction {
  id: string;
  prediction: string;
  confidence: number;
  timestamp: Date;
}

interface PredictionDisplayProps {
  predictions: Prediction[];
  title: string;
  variant: "local" | "remote";
}

function PredictionDisplay({
  predictions,
  title,
  variant,
}: PredictionDisplayProps) {
  const colorClass =
    variant === "local" ? "bg-purple-500/90" : "bg-cyan-500/90";

  return (
    <div className="bg-black/50 backdrop-blur-sm rounded-lg p-3 max-h-32">
      <div className="flex items-center justify-between mb-2">
        <h4 className="text-sm font-medium text-white">{title}</h4>
        <Badge className={`${colorClass} text-white text-xs`}>
          {predictions.length}
        </Badge>
      </div>

      <ScrollArea className="h-20">
        <div className="space-y-1">
          {predictions.length > 0 ? (
            predictions.slice(0, 3).map((prediction) => (
              <div
                key={prediction.id}
                className="flex items-center justify-between text-xs bg-white/10 rounded px-2 py-1"
              >
                <span className="text-white font-medium">
                  {prediction.prediction}
                </span>
                <span className="text-white/70">
                  {prediction.confidence.toFixed(1)}%
                </span>
              </div>
            ))
          ) : (
            <div className="text-xs text-white/50 text-center py-2">
              No predictions yet
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

export default memo(PredictionDisplay);
