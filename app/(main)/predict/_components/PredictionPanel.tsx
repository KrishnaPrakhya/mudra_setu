import { memo } from "react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Clock, TrendingUp } from "lucide-react";

interface AggregatedPrediction {
  id: string;
  prediction: string;
  confidence: number;
  timestamp: Date;
  count: number;
}

interface PredictionPanelProps {
  predictions: string[];
  aggregatedPredictions: AggregatedPrediction[];
  isCapturing: boolean;
  isBuffering: boolean;
  progress: number;
  fps: number;
}

function PredictionPanel({
  predictions,
  aggregatedPredictions,
  isCapturing,
  isBuffering,
  progress,
  fps = 20,
}: PredictionPanelProps) {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  return (
    <div className="space-y-4">
      {/* Recent Predictions */}
      <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl border border-purple-500/20">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            Live Predictions
          </h2>
          {isCapturing && (
            <div className="flex items-center space-x-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  fps >= 20 ? "bg-green-500" : "bg-yellow-500"
                } animate-pulse`}
              />
              <span className="text-sm text-cyan-300 font-mono">{fps} FPS</span>
            </div>
          )}
        </div>

        <ScrollArea className="h-48">
          <div className="space-y-2">
            {isCapturing && isBuffering ? (
              <div className="flex flex-col justify-center items-center h-full py-8">
                <p className="text-purple-300/80 mb-2">Buffering frames...</p>
                <Progress value={progress * 100} className="w-full" />
              </div>
            ) : predictions.length > 0 ? (
              predictions.map((pred, i) => (
                <div
                  key={i}
                  className="p-3 bg-slate-700/50 rounded-lg text-purple-300 border border-purple-400/20 animate-fade-in"
                  style={{ animationDelay: `${i * 0.1}s` }}
                >
                  {pred}
                </div>
              ))
            ) : (
              <div className="p-3 bg-slate-700/50 rounded-lg text-purple-300/50 border border-purple-400/20 text-center py-8">
                {isCapturing
                  ? "Waiting for confident prediction..."
                  : "Start capture to begin detection"}
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Aggregated Predictions */}
      <div className="bg-slate-800/50 backdrop-blur-sm p-6 rounded-xl border border-cyan-500/20">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp className="w-5 h-5 text-cyan-400" />
          <h2 className="text-xl font-semibold bg-gradient-to-r from-cyan-400 to-blue-400 bg-clip-text text-transparent">
            Session Summary
          </h2>
          <Badge variant="secondary" className="ml-auto">
            {aggregatedPredictions.length} unique
          </Badge>
        </div>

        <ScrollArea className="h-64">
          <div className="space-y-2">
            {aggregatedPredictions.length > 0 ? (
              aggregatedPredictions
                .sort((a, b) => b.count - a.count)
                .map((pred) => (
                  <div
                    key={pred.id}
                    className="p-3 bg-slate-700/50 rounded-lg border border-cyan-400/20 hover:border-cyan-400/40 transition-colors"
                  >
                    <div className="flex justify-between items-start mb-2">
                      <span className="font-medium text-cyan-300">
                        {pred.prediction}
                      </span>
                      <Badge variant="outline" className="text-xs">
                        {pred.count}x
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center text-xs text-slate-400">
                      <span>Confidence: {pred.confidence.toFixed(1)}%</span>
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatTime(pred.timestamp)}
                      </div>
                    </div>
                  </div>
                ))
            ) : (
              <div className="text-center py-8 text-slate-400">
                No predictions recorded yet
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}

export default memo(PredictionPanel);
