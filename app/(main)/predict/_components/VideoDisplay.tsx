import { memo } from "react";
import type { RefObject } from "react";

interface VideoDisplayProps {
  videoRef: RefObject<HTMLVideoElement>;
  landmarkCanvasRef: RefObject<HTMLCanvasElement>;
  canvasRef: RefObject<HTMLCanvasElement>;
}

function VideoDisplay({
  videoRef,
  landmarkCanvasRef,
  canvasRef,
}: VideoDisplayProps) {
  return (
    <div className="relative aspect-video bg-slate-800 rounded-xl overflow-hidden border-2 border-cyan-400/30">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full h-full object-cover"
      />
      <canvas
        ref={landmarkCanvasRef}
        id="landmarks-canvas"
        className="absolute top-0 left-0 w-full h-full"
        style={{ pointerEvents: "none" }}
      />
      <canvas ref={canvasRef} style={{ display: "none" }} />
    </div>
  );
}

export default memo(VideoDisplay);
