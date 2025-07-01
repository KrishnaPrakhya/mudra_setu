"use client";
import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Video,
  Users,
  MessageSquare,
  RepeatIcon as Record,
  Copy,
  UserPlus,
  Maximize2,
  Minimize2,
  AlertCircle,
  Loader2,
  VideoOff,
} from "lucide-react";
import { useVideoCall } from "./_components/useVideoCall";
import CallControls from "./_components/CallControls";
import PredictionDisplay from "./_components/PredictionDisplay";
import ChatPanel from "./_components/ChatPanel";
import ParticipantsList from "./_components/ParticipantsList";

export default function VideoCallPage() {
  const {
    // Call state
    isInCall,
    isConnecting,
    callId,
    participants,
    localStream,
    remoteStreams,
    connectionError,

    // Media controls
    isVideoEnabled,
    isAudioEnabled,
    toggleVideo,
    toggleAudio,

    // Call actions
    startCall,
    joinCall,
    endCall,

    // Predictions
    localPredictions,
    remotePredictions,
    speakPredictions,
    clearPredictions,

    // Recording
    isRecording,
    startRecording,
    stopRecording,
    startPredictionLoop,
    stopPredictionLoop,

    // Chat
    messages,
    sendMessage,

    // Settings
    settings,
    updateSettings,
  } = useVideoCall();

  const [roomInput, setRoomInput] = useState("");
  const [userName, setUserName] = useState("");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activeTab, setActiveTab] = useState("call");

  const localVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const remoteVideoRefs = useRef<{ [key: string]: HTMLVideoElement }>({});

  // Start/Stop prediction loop based on call state and video element readiness
  useEffect(() => {
    if (isInCall && localStream && localVideoRef.current && canvasRef.current) {
      startPredictionLoop(localVideoRef.current, canvasRef.current);
    }

    // Cleanup function to stop the loop when the component unmounts or dependencies change
    return () => {
      stopPredictionLoop();
    };
  }, [isInCall, localStream, startPredictionLoop, stopPredictionLoop]);

  // Set up local video stream
  useEffect(() => {
    if (localStream && localVideoRef.current) {
      const video = localVideoRef.current;
      video.srcObject = localStream;
      video.onloadedmetadata = () => {
        video.play().catch((e) => console.error("Local video play failed:", e));
      };
    }
  }, [localStream, isInCall]);

  // Set up remote video streams
  useEffect(() => {
    Object.entries(remoteStreams).forEach(([participantId, stream]) => {
      const videoElement = remoteVideoRefs.current[participantId];
      if (videoElement && stream) {
        videoElement.srcObject = stream;
        videoElement.onloadedmetadata = () => {
          videoElement
            .play()
            .catch((e) => console.error("Remote video play failed:", e));
        };
      }
    });
  }, [remoteStreams]);

  const handleStartCall = () => {
    if (!userName.trim()) {
      alert("Please enter your name");
      return;
    }
    startCall(userName);
  };

  const handleJoinCall = () => {
    if (!userName.trim() || !roomInput.trim()) {
      alert("Please enter your name and room ID");
      return;
    }
    joinCall(roomInput, userName);
  };

  const copyRoomId = () => {
    if (callId) {
      navigator.clipboard.writeText(callId);
      // You could add a toast notification here
    }
  };

  const toggleFullscreen = () => {
    setIsFullscreen(!isFullscreen);
  };

  if (!isInCall) {
    return (
      <div className="pt-[100px] min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
        <div className="max-w-4xl mx-auto p-8">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent mb-4">
              Sign Language Video Calls
            </h1>
            <p className="text-slate-400 text-lg">
              Connect with others and communicate through real-time sign
              language recognition
            </p>
          </div>

          {/* Connection Error */}
          {connectionError && (
            <Alert className="mb-6 border-red-500/50 bg-red-500/10">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription className="text-red-400">
                {connectionError}
              </AlertDescription>
            </Alert>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Start New Call */}
            <Card className="bg-slate-800/50 backdrop-blur-sm border-purple-500/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-xl">
                  <Video className="w-6 h-6 text-purple-400" />
                  Start New Call
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Your Name
                  </label>
                  <Input
                    value={userName}
                    onChange={(e) => setUserName(e.target.value)}
                    placeholder="Enter your name"
                    className="bg-slate-700/50 border-slate-600"
                    disabled={isConnecting}
                  />
                </div>
                <Button
                  onClick={handleStartCall}
                  disabled={isConnecting || !userName.trim()}
                  className="w-full bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600"
                  size="lg"
                >
                  {isConnecting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Starting...
                    </>
                  ) : (
                    "Start Call"
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Join Existing Call */}
            <Card className="bg-slate-800/50 backdrop-blur-sm border-cyan-500/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-xl">
                  <UserPlus className="w-6 h-6 text-cyan-400" />
                  Join Call
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Your Name
                  </label>
                  <Input
                    value={userName}
                    onChange={(e) => setUserName(e.target.value)}
                    placeholder="Enter your name"
                    className="bg-slate-700/50 border-slate-600"
                    disabled={isConnecting}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-2">
                    Room ID
                  </label>
                  <Input
                    value={roomInput}
                    onChange={(e) => setRoomInput(e.target.value)}
                    placeholder="Enter room ID"
                    className="bg-slate-700/50 border-slate-600"
                    disabled={isConnecting}
                  />
                </div>
                <Button
                  onClick={handleJoinCall}
                  disabled={
                    isConnecting || !userName.trim() || !roomInput.trim()
                  }
                  variant="outline"
                  className="w-full border-cyan-500/50 text-cyan-400 hover:bg-cyan-500/10 bg-transparent"
                  size="lg"
                >
                  {isConnecting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Joining...
                    </>
                  ) : (
                    "Join Call"
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Features Preview */}
          <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-6 bg-slate-800/30 rounded-xl border border-slate-700/50">
              <Video className="w-12 h-12 mx-auto mb-4 text-purple-400" />
              <h3 className="text-lg font-semibold mb-2">HD Video Calls</h3>
              <p className="text-slate-400 text-sm">
                Crystal clear video communication with adaptive quality
              </p>
            </div>
            <div className="text-center p-6 bg-slate-800/30 rounded-xl border border-slate-700/50">
              <MessageSquare className="w-12 h-12 mx-auto mb-4 text-cyan-400" />
              <h3 className="text-lg font-semibold mb-2">
                Real-time Recognition
              </h3>
              <p className="text-slate-400 text-sm">
                AI-powered sign language detection for both participants
              </p>
            </div>
            <div className="text-center p-6 bg-slate-800/30 rounded-xl border border-slate-700/50">
              <Record className="w-12 h-12 mx-auto mb-4 text-green-400" />
              <h3 className="text-lg font-semibold mb-2">Session Recording</h3>
              <p className="text-slate-400 text-sm">
                Record calls with synchronized predictions
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`${
        isFullscreen ? "fixed inset-0 z-50" : "min-h-screen"
      } bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white`}
    >
      {/* Error Display */}
      {connectionError && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 m-4 rounded-lg">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5" />
            <span>{connectionError}</span>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="pt-[80px] bg-black/20 backdrop-blur-sm border-b border-purple-500/20 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse" />
                <span className="font-medium">Live Call</span>
              </div>
              {callId && (
                <div className="flex items-center gap-2 bg-slate-800/50 px-3 py-1 rounded-full">
                  <span className="text-sm">Room: {callId}</span>
                  <Button
                    onClick={copyRoomId}
                    size="sm"
                    variant="ghost"
                    className="h-6 w-6 p-0"
                  >
                    <Copy className="w-3 h-3" />
                  </Button>
                </div>
              )}
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="flex items-center gap-1">
                <Users className="w-3 h-3" />
                {participants.length}
              </Badge>
              <Button onClick={toggleFullscreen} size="sm" variant="ghost">
                {isFullscreen ? (
                  <Minimize2 className="w-4 h-4" />
                ) : (
                  <Maximize2 className="w-4 h-4" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>

      <div className="flex h-[calc(100vh-64px)]">
        {/* Main Video Area */}
        <div className="flex-1 p-4">
          <div className="h-full grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Local Video */}
            <div className="relative bg-slate-800/50 rounded-xl overflow-hidden border border-purple-500/20 flex items-center justify-center">
              {isVideoEnabled && localStream ? (
                <>
                  <video
                    ref={localVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />
                  <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full"
                  />
                </>
              ) : (
                <div className="flex flex-col items-center justify-center text-slate-400">
                  <VideoOff className="w-12 h-12" />
                  <p className="mt-2">Your camera is off</p>
                </div>
              )}
              <div className="absolute top-4 left-4">
                <Badge className="bg-purple-500/90 text-white">You</Badge>
              </div>
              <div className="absolute bottom-4 left-4 right-4">
                <PredictionDisplay
                  predictions={localPredictions}
                  title="Your Signs"
                  variant="local"
                />
              </div>
            </div>

            {/* Remote Video(s) */}
            <div className="space-y-4">
              {participants
                .filter((p) => !p.isLocal)
                .map((participant) => {
                  const stream = remoteStreams[participant.id];
                  return (
                    <div
                      key={participant.id}
                      className="relative bg-slate-800/50 rounded-xl overflow-hidden border border-cyan-500/20 h-full"
                    >
                      {stream ? (
                        <video
                          ref={(el) => {
                            if (el) remoteVideoRefs.current[participant.id] = el;
                          }}
                          autoPlay
                          playsInline
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <div className="text-center">
                            <Loader2 className="w-8 h-8 mx-auto mb-4 text-slate-400 animate-spin" />
                            <p className="text-slate-400">
                              Connecting to {participant.name}...
                            </p>
                          </div>
                        </div>
                      )}
                      <div className="absolute top-4 left-4">
                        <Badge className="bg-cyan-500/90 text-white">
                          {participant?.name || "Participant"}
                        </Badge>
                      </div>
                      <div className="absolute bottom-4 left-4 right-4">
                        <PredictionDisplay
                          predictions={remotePredictions[participant.id] || []}
                          title={`${participant?.name || "Participant"}'s Signs`}
                          variant="remote"
                        />
                      </div>
                    </div>
                  );
                })}
              {participants.length <= 1 && (
                <div className="h-full flex items-center justify-center bg-slate-800/30 rounded-xl border border-slate-700/50">
                  <div className="text-center">
                    <Users className="w-12 h-12 mx-auto mb-4 text-slate-400" />
                    <p className="text-slate-400">
                      Waiting for participants to join...
                    </p>
                    {callId && (
                      <p className="text-slate-500 text-sm mt-2">
                        Share room ID:{" "}
                        <span className="font-mono">{callId}</span>
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Call Controls */}
          <div className="mt-4">
            <CallControls
              isVideoEnabled={isVideoEnabled}
              isAudioEnabled={isAudioEnabled}
              isRecording={isRecording}
              onToggleVideo={toggleVideo}
              onToggleAudio={toggleAudio}
              onStartRecording={startRecording}
              onStopRecording={stopRecording}
              onEndCall={endCall}
              onSpeakPredictions={speakPredictions}
              onClearPredictions={clearPredictions}
            />
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-80 bg-slate-800/30 border-l border-slate-700/50">
          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="h-full flex flex-col"
          >
            <TabsList className="grid w-full grid-cols-3 bg-slate-800/50 m-2">
              <TabsTrigger value="call">Call</TabsTrigger>
              <TabsTrigger value="chat">Chat</TabsTrigger>
              <TabsTrigger value="settings">Settings</TabsTrigger>
            </TabsList>

            <div className="flex-1 overflow-hidden">
              <TabsContent value="call" className="h-full m-0 p-4">
                <ParticipantsList participants={participants} />
              </TabsContent>

              <TabsContent value="chat" className="h-full m-0">
                <ChatPanel messages={messages} onSendMessage={sendMessage} />
              </TabsContent>

              <TabsContent value="settings" className="h-full m-0 p-4">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Call Settings</h3>
                  <div className="text-sm text-slate-400">
                    Settings panel - customize your call experience
                  </div>
                </div>
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
