"use client";

import { memo } from "react";
import { Button } from "@/components/ui/button";
import { Play, Square, Volume2, Trash2, Download } from "lucide-react";

interface VideoControlsProps {
  isCapturing: boolean;
  toggleCapture: () => void;
  speakPredictions: () => void;
  clearPredictions: () => void;
  exportPredictions: () => void;
  aggregatedCount: number;
}

function VideoControls({
  isCapturing,
  toggleCapture,
  speakPredictions,
  clearPredictions,
  exportPredictions,
  aggregatedCount,
}: VideoControlsProps) {
  return (
    <div className="p-4 bg-slate-900/50 backdrop-blur-sm border-t border-purple-500/20">
      <div className="flex flex-wrap gap-3 justify-center">
        {/* Main Control */}
        <Button
          onClick={toggleCapture}
          size="lg"
          className={`px-6 py-3 font-semibold transition-all duration-300 ${
            isCapturing
              ? "bg-red-500 hover:bg-red-600 text-white"
              : "bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600 text-white"
          }`}
        >
          {isCapturing ? (
            <>
              <Square className="w-5 h-5 mr-2" />
              Stop Detection
            </>
          ) : (
            <>
              <Play className="w-5 h-5 mr-2" />
              Start Detection
            </>
          )}
        </Button>

        {/* Secondary Controls */}
        <Button
          onClick={speakPredictions}
          disabled={aggregatedCount === 0}
          variant="outline"
          size="lg"
          className="border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10 bg-transparent"
        >
          <Volume2 className="w-5 h-5 mr-2" />
          Speak All ({aggregatedCount})
        </Button>

        <Button
          onClick={clearPredictions}
          disabled={aggregatedCount === 0}
          variant="outline"
          size="lg"
          className="border-red-500/50 text-red-400 hover:bg-red-500/10 bg-transparent"
        >
          <Trash2 className="w-5 h-5 mr-2" />
          Clear
        </Button>

        <Button
          onClick={exportPredictions}
          disabled={aggregatedCount === 0}
          variant="outline"
          size="lg"
          className="border-green-500/50 text-green-400 hover:bg-green-500/10 bg-transparent"
        >
          <Download className="w-5 h-5 mr-2" />
          Export
        </Button>
      </div>
    </div>
  );
}

export default memo(VideoControls);
