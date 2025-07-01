"use client";
import { memo } from "react";
import { Button } from "@/components/ui/button";
import {
  Video,
  VideoOff,
  Mic,
  MicOff,
  PhoneOff,
  RepeatIcon as Record,
  Square,
  Volume2,
  Trash2,
} from "lucide-react";

interface CallControlsProps {
  isVideoEnabled: boolean;
  isAudioEnabled: boolean;
  isRecording: boolean;
  onToggleVideo: () => void;
  onToggleAudio: () => void;
  onStartRecording: () => void;
  onStopRecording: () => void;
  onEndCall: () => void;
  onSpeakPredictions: () => void;
  onClearPredictions: () => void;
}

function CallControls({
  isVideoEnabled,
  isAudioEnabled,
  isRecording,
  onToggleVideo,
  onToggleAudio,
  onStartRecording,
  onStopRecording,
  onEndCall,
  onSpeakPredictions,
  onClearPredictions,
}: CallControlsProps) {
  return (
    <div className="flex items-center justify-center gap-4 p-4 bg-slate-800/50 backdrop-blur-sm rounded-xl border border-slate-700/50">
      {/* Media Controls */}
      <div className="flex items-center gap-2">
        <Button
          onClick={onToggleVideo}
          variant={isVideoEnabled ? "default" : "destructive"}
          size="lg"
          className="rounded-full w-12 h-12 p-0"
        >
          {isVideoEnabled ? (
            <Video className="w-5 h-5" />
          ) : (
            <VideoOff className="w-5 h-5" />
          )}
        </Button>

        <Button
          onClick={onToggleAudio}
          variant={isAudioEnabled ? "default" : "destructive"}
          size="lg"
          className="rounded-full w-12 h-12 p-0"
        >
          {isAudioEnabled ? (
            <Mic className="w-5 h-5" />
          ) : (
            <MicOff className="w-5 h-5" />
          )}
        </Button>
      </div>

      {/* Recording Controls */}
      <div className="flex items-center gap-2">
        <Button
          onClick={isRecording ? onStopRecording : onStartRecording}
          variant={isRecording ? "destructive" : "outline"}
          size="lg"
          className="rounded-full w-12 h-12 p-0"
        >
          {isRecording ? (
            <Square className="w-5 h-5" />
          ) : (
            <Record className="w-5 h-5" />
          )}
        </Button>
      </div>

      {/* Prediction Controls */}
      <div className="flex items-center gap-2">
        <Button
          onClick={onSpeakPredictions}
          variant="outline"
          size="lg"
          className="rounded-full w-12 h-12 p-0 bg-transparent"
        >
          <Volume2 className="w-5 h-5" />
        </Button>

        <Button
          onClick={onClearPredictions}
          variant="outline"
          size="lg"
          className="rounded-full w-12 h-12 p-0 bg-transparent"
        >
          <Trash2 className="w-5 h-5" />
        </Button>
      </div>

      {/* End Call */}
      <Button
        onClick={onEndCall}
        variant="destructive"
        size="lg"
        className="rounded-full w-12 h-12 p-0 bg-red-500 hover:bg-red-600"
      >
        <PhoneOff className="w-5 h-5" />
      </Button>
    </div>
  );
}

export default memo(CallControls);
