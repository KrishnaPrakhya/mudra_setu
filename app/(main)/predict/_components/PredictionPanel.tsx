import { memo } from "react";

interface PredictionPanelProps {
  predictions: string[];
  isCapturing: boolean;
  fps: number;
}

function PredictionPanel({
  predictions,
  isCapturing,
  fps,
}: PredictionPanelProps) {
  return (
    <div className="bg-slate-800 p-6 rounded-xl border border-cyan-400/20">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-cyan-400">
          Recent Predictions
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
      <div className="space-y-3">
        {predictions.length > 0 ? (
          predictions.map((pred, i) => (
            <div
              key={i}
              className="p-3 bg-slate-700/50 rounded-lg text-cyan-300 border border-cyan-400/20"
            >
              {pred}
            </div>
          ))
        ) : (
          <div className="p-3 bg-slate-700/50 rounded-lg text-cyan-300/50 border border-cyan-400/20">
            No predictions yet. Start capture to begin.
          </div>
        )}
      </div>
    </div>
  );
}

export default memo(PredictionPanel);
