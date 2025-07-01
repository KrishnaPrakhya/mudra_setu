"use client";
import { useState, useRef, useEffect, useCallback } from "react";
import { mediaPipeClient } from "../../predict/_components/MediaPipeClient";

interface Participant {
  id: string;
  name: string;
  isLocal: boolean;
}

interface Message {
  id: string;
  senderId: string;
  senderName: string;
  content: string;
  timestamp: Date;
  type: "text" | "system";
}

interface Prediction {
  id: string;
  prediction: string;
  confidence: number;
  timestamp: Date;
}

// Messages coming from the backend when we request/receive a prediction
interface ServerPrediction {
  status: "buffering" | "prediction";
  // Progress is present while buffering
  progress?: number;
  // These two are present when status === "prediction"
  prediction?: string;
  confidence?: number;
}

interface CallSettings {
  videoQuality: "low" | "medium" | "high";
  audioEnabled: boolean;
  videoEnabled: boolean;
  predictionEnabled: boolean;
  autoSpeak: boolean;
}

export function useVideoCall() {
  // Call state
  const [isInCall, setIsInCall] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [callId, setCallId] = useState<string | null>(null);
  const [participants, setParticipants] = useState<Participant[]>([]);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [localUserId, setLocalUserId] = useState<string | null>(null);

  // Media state
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStreams, setRemoteStreams] = useState<{
    [key: string]: MediaStream;
  }>({});
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);

  // Predictions
  const [localPredictions, setLocalPredictions] = useState<Prediction[]>([]);
  const [remotePredictions, setRemotePredictions] = useState<{
    [key: string]: Prediction[];
  }>({});

  // Recording
  const [isRecording, setIsRecording] = useState(false);

  // Chat
  const [messages, setMessages] = useState<Message[]>([]);

  // Settings
  const [settings, setSettings] = useState<CallSettings>({
    videoQuality: "medium",
    audioEnabled: true,
    videoEnabled: true,
    predictionEnabled: true,
    autoSpeak: false,
  });

  // Refs
  const socketRef = useRef<WebSocket | null>(null);
  const peerConnectionsRef = useRef<{ [key: string]: RTCPeerConnection }>({});
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const predictionIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isEndingCallRef = useRef(false);
  const connectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Local video reference
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const remoteVideoRefs = useRef<{ [key: string]: HTMLVideoElement }>({});

  // Initialize media stream
  /**
   * Try to obtain a local `MediaStream`.
   *
   * 1. First we request **video + audio**.
   * 2. If that fails with a `NotReadableError` (device already in use) or
   *    `OverconstrainedError`, we fall back to **video-only**.
   * 3. If that also fails, we finally try **audio-only**.
   * 4. If everything fails we simply return `null` so that the caller can still
   *    proceed without local media (e.g. for testing on a single machine).
   */
  const initializeMedia = useCallback(async (): Promise<MediaStream | null> => {
    const constraintsVariants: MediaStreamConstraints[] = [
      {
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: true,
      },
      { video: true, audio: false },
      { video: false, audio: true },
    ];

    for (const constraints of constraintsVariants) {
      try {
        console.log("Requesting media access with constraints", constraints);
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        console.log("Media access granted:", stream);
        setLocalStream(stream);
        // Update enabled flags so UI reflects the actual tracks we have
        setIsVideoEnabled(stream.getVideoTracks().length > 0);
        setIsAudioEnabled(stream.getAudioTracks().length > 0);
        return stream;
      } catch (err: unknown) {
        console.warn(
          `getUserMedia failed with constraints ${JSON.stringify(
            constraints
          )}:`,
          err
        );
        // Only keep trying if the error is related to device being busy or constraints issues
        if (
          !(err instanceof DOMException) ||
          (err.name !== "NotReadableError" &&
            err.name !== "TrackStartError" &&
            err.name !== "OverconstrainedError")
        ) {
          // For permission denied or other fatal errors, break early
          break;
        }
      }
    }

    console.error("Unable to obtain local media stream (camera/microphone)");
    // We do **not** throw here so that the calling flow can still continue
    return null;
  }, []);

  // WebRTC peer connection management
  const createPeerConnection = useCallback(
    async (participantId: string) => {
      console.log("Creating peer connection for:", participantId);

      // Close existing connection if any
      if (peerConnectionsRef.current[participantId]) {
        peerConnectionsRef.current[participantId].close();
        delete peerConnectionsRef.current[participantId];
      }

      const peerConnection = new RTCPeerConnection({
        iceServers: [
          { urls: "stun:stun.l.google.com:19302" },
          { urls: "stun:stun1.l.google.com:19302" },
        ],
      });

      // Add local stream tracks
      if (localStream) {
        localStream.getTracks().forEach((track) => {
          console.log("Adding track to peer connection:", track.kind);
          peerConnection.addTrack(track, localStream);
        });
      }

      // Handle remote stream
      peerConnection.ontrack = (event) => {
        console.log("Received remote track from", participantId, ":", event);
        const [remoteStream] = event.streams;
        setRemoteStreams((prev) => ({
          ...prev,
          [participantId]: remoteStream,
        }));
      };

      // Handle ICE candidates
      peerConnection.onicecandidate = (event) => {
        if (
          event.candidate &&
          socketRef.current?.readyState === WebSocket.OPEN
        ) {
          console.log(
            "Sending ICE candidate to",
            participantId,
            ":",
            event.candidate
          );
          socketRef.current.send(
            JSON.stringify({
              type: "ice-candidate",
              candidate: event.candidate,
              targetId: participantId,
            })
          );
        }
      };

      // Handle connection state changes
      peerConnection.onconnectionstatechange = () => {
        console.log(
          `Peer connection state for ${participantId}:`,
          peerConnection.connectionState
        );
        if (peerConnection.connectionState === "failed") {
          console.log(
            `Peer connection failed for ${participantId}, attempting to restart`
          );
        }
      };

      peerConnectionsRef.current[participantId] = peerConnection;
      return peerConnection;
    },
    [localStream]
  );

  const handleOffer = useCallback(
    async (offer: RTCSessionDescriptionInit, senderId: string) => {
      console.log("Handling offer from:", senderId);
      try {
        const peerConnection = await createPeerConnection(senderId);

        await peerConnection.setRemoteDescription(offer);
        const answer = await peerConnection.createAnswer();
        await peerConnection.setLocalDescription(answer);

        if (socketRef.current?.readyState === WebSocket.OPEN) {
          console.log("Sending answer to:", senderId);
          socketRef.current.send(
            JSON.stringify({
              type: "answer",
              answer,
              targetId: senderId,
            })
          );
        }
      } catch (error) {
        console.error("Error handling offer from", senderId, ":", error);
      }
    },
    [createPeerConnection]
  );

  const handleAnswer = useCallback(
    async (answer: RTCSessionDescriptionInit, senderId: string) => {
      console.log("Handling answer from:", senderId);
      try {
        const peerConnection = peerConnectionsRef.current[senderId];
        if (peerConnection) {
          await peerConnection.setRemoteDescription(answer);
          console.log("Answer processed successfully for:", senderId);
        } else {
          console.error("No peer connection found for:", senderId);
        }
      } catch (error) {
        console.error("Error handling answer from", senderId, ":", error);
      }
    },
    []
  );

  const handleIceCandidate = useCallback(
    async (candidate: RTCIceCandidateInit, senderId: string) => {
      console.log("Handling ICE candidate from:", senderId);
      try {
        const peerConnection = peerConnectionsRef.current[senderId];
        if (peerConnection && peerConnection.remoteDescription) {
          await peerConnection.addIceCandidate(candidate);
          console.log("ICE candidate added successfully for:", senderId);
        } else {
          console.log(
            "Peer connection not ready for ICE candidate from:",
            senderId
          );
        }
      } catch (error) {
        console.error(
          "Error handling ICE candidate from",
          senderId,
          ":",
          error
        );
      }
    },
    []
  );

  const handlePrediction = useCallback(
    (senderId: string, prediction: ServerPrediction) => {
      // Handle buffering status
      if (prediction.status === "buffering") {
        const progress = prediction.progress ?? 0;
        const bufferingPrediction: Prediction = {
          id: "buffering",
          prediction: `Buffering... (${Math.round(progress * 100)}%)`,
          confidence: progress * 100,
          timestamp: new Date(),
        };
        if (senderId === localUserId) {
          setLocalPredictions((prev) => {
            const other = prev.filter((p) => p.id !== "buffering");
            return [bufferingPrediction, ...other];
          });
        }
        return; // Stop here for buffering
      }

      // Handle actual predictions
      if (prediction.status === "prediction") {
        const newPrediction: Prediction = {
          id: Date.now().toString(),
          prediction: prediction.prediction ?? "",
          confidence: prediction.confidence ?? 0,
          timestamp: new Date(),
        };

        if (senderId === localUserId) {
          setLocalPredictions((prev) => {
            const other = prev.filter((p) => p.id !== "buffering");
            return [newPrediction, ...other].slice(0, 5);
          });

          if (settings.autoSpeak) {
            const utterance = new SpeechSynthesisUtterance(
              prediction.prediction ?? ""
            );
            speechSynthesis.speak(utterance);
          }
        } else {
          setRemotePredictions((prev) => ({
            ...prev,
            [senderId]: [newPrediction, ...(prev[senderId] || [])].slice(0, 5),
          }));
        }
      }
    },
    [localUserId, settings.autoSpeak]
  );

  const initiateCall = useCallback(
    async (participantId: string) => {
      console.log("Initiating call to:", participantId);
      try {
        const peerConnection = await createPeerConnection(participantId);

        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);

        if (socketRef.current?.readyState === WebSocket.OPEN) {
          console.log("Sending offer to:", participantId);
          socketRef.current.send(
            JSON.stringify({
              type: "offer",
              offer,
              targetId: participantId,
            })
          );
        }
      } catch (error) {
        console.error("Error initiating call to", participantId, ":", error);
      }
    },
    [createPeerConnection]
  );

  // WebSocket connection with better error handling
  const connectWebSocket = useCallback(
    (roomId: string, userName: string) => {
      console.log("Connecting to WebSocket:", roomId, userName);
      setConnectionError(null);
      setIsConnecting(true);
      isEndingCallRef.current = false;

      // Clear any existing timeouts
      if (connectionTimeoutRef.current) {
        clearTimeout(connectionTimeoutRef.current);
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }

      // Close existing connection
      if (socketRef.current) {
        console.log("Closing existing WebSocket connection");
        socketRef.current.close();
        socketRef.current = null;
      }

      // Build the WebSocket URL: always target the backend on port 8000
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const backendHost = "localhost:8000"; // Hardcode to localhost:8000 to match working fetch calls

      // Trim / encode the roomId to avoid accidental spaces or special chars
      const cleanRoomId = encodeURIComponent(roomId.trim());

      const wsUrl = `${protocol}//${backendHost}/ws/video-call/${cleanRoomId}`;
      console.log("Attempting to connect to WebSocket at:", wsUrl);

      try {
        const socket = new WebSocket(wsUrl);

        // Set connection timeout
        connectionTimeoutRef.current = setTimeout(() => {
          if (socket.readyState === WebSocket.CONNECTING) {
            console.log("WebSocket connection timeout");
            socket.close();
            setConnectionError("Connection timeout. Please try again.");
            setIsConnecting(false);
          }
        }, 10000); // 10 second timeout

        socket.onopen = () => {
          console.log("WebSocket connected successfully");
          if (connectionTimeoutRef.current) {
            clearTimeout(connectionTimeoutRef.current);
          }
          setConnectionError(null);

          // Send join message immediately after connection opens
          const joinMessage = {
            type: "join",
            userName,
            roomId,
          };
          console.log("Sending join message:", joinMessage);
          socket.send(JSON.stringify(joinMessage));
        };

        socket.onmessage = async (event) => {
          try {
            const data = JSON.parse(event.data);
            console.log("Received WebSocket message:", data);

            switch (data.type) {
              case "room-joined":
                console.log("Successfully joined room:", data);
                setIsConnecting(false);
                setIsInCall(true);
                setLocalUserId(data.userId);

                // Set participants including self, using the real ID from server
                const allParticipants = [
                  { id: data.userId, name: userName, isLocal: true },
                  ...data.participants.map((p: Participant) => ({
                    ...p,
                    isLocal: false,
                  })),
                ];
                setParticipants(allParticipants);

                // Initiate calls to existing participants
                if (data.participants.length > 0) {
                  console.log(
                    "Found existing participants, initiating calls...",
                    data.participants
                  );
                  data.participants.forEach((participant: Participant) => {
                    if (participant.id !== "local") {
                      initiateCall(participant.id);
                    }
                  });
                }
                break;

              case "participant-joined":
                console.log("Participant joined:", data.participant);
                setParticipants((prev) => {
                  // Avoid duplicates
                  if (prev.find((p) => p.id === data.participant.id)) {
                    return prev;
                  }
                  return [...prev, { ...data.participant, isLocal: false }];
                });

                // Initiate a call to the new participant
                if (data.participant.id !== "local") {
                  console.log(
                    "New participant detected, initiating call to:",
                    data.participant.id
                  );
                  initiateCall(data.participant.id);
                }
                break;

              case "participant-left":
                console.log("Participant left:", data.participantId);
                setParticipants((prev) =>
                  prev.filter((p) => p.id !== data.participantId)
                );
                closePeerConnection(data.participantId);
                break;

              case "offer":
                await handleOffer(data.offer, data.senderId);
                break;

              case "answer":
                await handleAnswer(data.answer, data.senderId);
                break;

              case "ice-candidate":
                await handleIceCandidate(data.candidate, data.senderId);
                break;

              case "prediction":
                handlePrediction(data.senderId, data.prediction);
                break;

              case "chat-message":
                setMessages((prev) => [
                  ...prev,
                  {
                    id: Date.now().toString(),
                    senderId: data.senderId,
                    senderName: data.senderName,
                    content: data.content,
                    timestamp: new Date(),
                    type: "text",
                  },
                ]);
                break;

              default:
                console.log("Unknown message type:", data.type);
            }
          } catch (error) {
            console.error("Error processing WebSocket message:", error);
          }
        };

        socket.onclose = (event) => {
          console.log("WebSocket disconnected:", event.code, event.reason);
          if (connectionTimeoutRef.current) {
            clearTimeout(connectionTimeoutRef.current);
          }

          // Only handle reconnection if we're not intentionally ending the call
          if (!isEndingCallRef.current) {
            setIsConnecting(false);

            if (event.code !== 1000) {
              const errorMessage = `Connection lost (code: ${event.code}${
                event.reason ? `, reason: ${event.reason}` : ""
              })`;
              console.log(errorMessage);
              setConnectionError(errorMessage);

              // Only attempt reconnection if we were previously in a call
              if (isInCall && event.code !== 1006) {
                if (reconnectTimeoutRef.current) {
                  clearTimeout(reconnectTimeoutRef.current);
                }
                reconnectTimeoutRef.current = setTimeout(() => {
                  if (!isEndingCallRef.current) {
                    console.log("Attempting to reconnect...");
                    connectWebSocket(roomId, userName);
                  }
                }, 3000);
              }
            }
          }
        };

        socket.onerror = (error: Event) => {
          console.error("WebSocket error:", error);
          if (connectionTimeoutRef.current) {
            clearTimeout(connectionTimeoutRef.current);
          }

          if (!isEndingCallRef.current) {
            setIsConnecting(false);
            setConnectionError(
              "Failed to connect to server. Please check if the server is running on port 8000."
            );
          }
        };

        socketRef.current = socket;
      } catch (error) {
        console.error("Error creating WebSocket:", error);
        setIsConnecting(false);
        setConnectionError("Failed to create WebSocket connection");
      }
    },
    [
      handleOffer,
      handleAnswer,
      handleIceCandidate,
      handlePrediction,
      initiateCall,
      isInCall,
    ]
  );

  const closePeerConnection = useCallback((participantId: string) => {
    console.log("Closing peer connection for:", participantId);
    const peerConnection = peerConnectionsRef.current[participantId];
    if (peerConnection) {
      peerConnection.close();
      delete peerConnectionsRef.current[participantId];
    }

    setRemoteStreams((prev) => {
      const newStreams = { ...prev };
      delete newStreams[participantId];
      return newStreams;
    });
  }, []);

  // Prediction handling
  const startPredictionLoop = useCallback(
    async (videoElement: HTMLVideoElement, canvasElement: HTMLCanvasElement) => {
      if (!settings.predictionEnabled) {
        console.log("Predictions disabled, skipping prediction loop");
        return;
      }

      try {
        console.log("Initializing MediaPipe...");
        await mediaPipeClient.initialize();
        console.log("MediaPipe initialized successfully");

        predictionIntervalRef.current = setInterval(async () => {
          if (
            videoElement &&
            videoElement.readyState >= 3 &&
            !videoElement.paused &&
            !isEndingCallRef.current
          ) {
            try {
              const landmarkData = mediaPipeClient.detect(
                videoElement
              );
              if (
                landmarkData &&
                landmarkData.keypoints.length > 0 &&
                socketRef.current?.readyState === WebSocket.OPEN
              ) {
                socketRef.current.send(
                  JSON.stringify({
                    type: "prediction-request",
                    landmarks: landmarkData.keypoints,
                  })
                );
              }
            } catch (error) {
              console.error("Error in prediction loop:", error);
            }
          }
        }, 200);
      } catch (error) {
        console.error("Error initializing prediction loop:", error);
        setConnectionError("Failed to initialize gesture recognition");
      }
    },
    [localStream, settings.predictionEnabled]
  );

  const stopPredictionLoop = useCallback(() => {
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
      predictionIntervalRef.current = null;
      console.log("Prediction loop stopped.");
    }
  }, []);

  // Call actions
  const startCall = useCallback(
    async (userName: string) => {
      console.log("Starting call for user:", userName);
      setIsConnecting(true);
      setConnectionError(null);
      isEndingCallRef.current = false;

      try {
        const stream = await initializeMedia();
        console.log("Media step done (stream present:", !!stream, "), creating room...");

        const response = await fetch(
          "http://localhost:8000/api/video-call/create-room",
          { method: "POST" }
        );

        if (!response.ok) {
          throw new Error("Failed to create room");
        }

        const data = await response.json();
        const roomId = data.roomId;

        setCallId(roomId);
        connectWebSocket(roomId, userName);

      } catch (error) {
        console.error("Error starting call:", error);
        setConnectionError(
          error instanceof Error ? error.message : "Failed to start call"
        );
        setIsInCall(false);
        setIsConnecting(false);
        isEndingCallRef.current = false;
      }
    },
    [initializeMedia, connectWebSocket]
  );

  const joinCall = useCallback(
    async (roomId: string, userName: string) => {
      console.log("Joining call:", roomId, userName);
      setIsConnecting(true);
      setConnectionError(null);
      isEndingCallRef.current = false;

      try {
        // Step 1: Initialize media, matching the order of the working startCall flow.
        const stream = await initializeMedia();

        // Step 2: Check if room exists on the server.
        console.log(`Verifying room with ID: [${roomId.trim()}]`);
        const checkResponse = await fetch(
          `http://localhost:8000/api/video-call/rooms/${roomId.trim()}`
        );

        if (!checkResponse.ok) {
          if (checkResponse.status === 404) {
            throw new Error("Room does not exist. Please check the ID.");
          }
          throw new Error(
            `Failed to verify room. Server responded with ${checkResponse.status}`
          );
        }
        console.log("Room verified successfully.");

        // Step 3: Connect WebSocket
        setCallId(roomId);
        connectWebSocket(roomId, userName);

      } catch (error) {
        console.error("Error joining call:", error);
        setConnectionError(
          error instanceof Error ? error.message : "Failed to join call"
        );
        setIsInCall(false);
        setIsConnecting(false);
        isEndingCallRef.current = false;
      }
    },
    [initializeMedia, connectWebSocket]
  );

  const endCall = useCallback(() => {
    console.log("Ending call...");
    isEndingCallRef.current = true;

    // Clear all timeouts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (connectionTimeoutRef.current) {
      clearTimeout(connectionTimeoutRef.current);
    }

    // Close all peer connections
    Object.values(peerConnectionsRef.current).forEach((pc) => pc.close());
    peerConnectionsRef.current = {};

    // Close WebSocket
    if (socketRef.current) {
      socketRef.current.close(1000, "Call ended");
      socketRef.current = null;
    } 

    // Stop local stream
    if (localStream) {
      localStream.getTracks().forEach((track) => track.stop());
    }

    // Stop prediction loop
    if (predictionIntervalRef.current) {
      clearInterval(predictionIntervalRef.current);
    }

    // Stop recording if active
    if (isRecording) {
      stopRecording();
    }

    // Reset state
    setIsInCall(false);
    setCallId(null);
    setParticipants([]);
    setLocalStream(null);
    setRemoteStreams({});
    setLocalPredictions([]);
    setRemotePredictions({});
    setMessages([]);
    setConnectionError(null);
    setIsConnecting(false);

    console.log("Call ended successfully");
  }, [localStream, isRecording]);

  // Media controls
  const toggleVideo = useCallback(() => {
    if (localStream) {
      const videoTrack = localStream.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setIsVideoEnabled(videoTrack.enabled);
      }
    }
  }, [localStream]);

  const toggleAudio = useCallback(() => {
    if (localStream) {
      const audioTrack = localStream.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsAudioEnabled(audioTrack.enabled);
      }
    }
  }, [localStream]);

  // Recording
  const startRecording = useCallback(() => {
    if (localStream) {
      const mediaRecorder = new MediaRecorder(localStream);
      const chunks: Blob[] = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: "video/webm" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `call-recording-${Date.now()}.webm`;
        a.click();
        URL.revokeObjectURL(url);
      };

      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
    }
  }, [localStream]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current = null;
      setIsRecording(false);
    }
  }, []);

  // Prediction actions
  const speakPredictions = useCallback(() => {
    const allPredictions = [
      ...localPredictions,
      ...Object.values(remotePredictions).flat(),
    ];

    if (allPredictions.length === 0) {
      const utterance = new SpeechSynthesisUtterance("No predictions to speak");
      speechSynthesis.speak(utterance);
      return;
    }

    const text = allPredictions
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 10)
      .map((p) => p.prediction)
      .join(", ");

    const utterance = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(utterance);
  }, [localPredictions, remotePredictions]);

  const clearPredictions = useCallback(() => {
    setLocalPredictions([]);
    setRemotePredictions({});
  }, []);

  // Chat
  const sendMessage = useCallback((content: string) => {
    if (socketRef.current && content.trim()) {
      socketRef.current.send(
        JSON.stringify({
          type: "chat-message",
          content: content.trim(),
        })
      );
    }
  }, []);

  // Settings
  const updateSettings = useCallback((newSettings: Partial<CallSettings>) => {
    setSettings((prev) => ({ ...prev, ...newSettings }));
  }, []);

  // Attach local tracks to any existing peer connections once we finally get a
  // MediaStream (this can happen asynchronously when the first attempt failed
  // and we retried with different constraints).
  useEffect(() => {
    if (!localStream) return;

    Object.values(peerConnectionsRef.current).forEach((pc) => {
      // Avoid adding duplicate tracks
      const sendersKinds = pc.getSenders().map((s) => s.track?.kind);
      localStream.getTracks().forEach((track) => {
        if (!sendersKinds.includes(track.kind)) {
          try {
            pc.addTrack(track, localStream);
            console.log(`Added ${track.kind} track to existing peer connection`);
          } catch (err) {
            console.warn("Failed to add track to peer connection", err);
          }
        }
      });
    });
  }, [localStream]);

  // Start/Stop prediction loop based on call state
  useEffect(() => {
    if (isInCall && localStream && localVideoRef.current && canvasRef.current) {
      console.log("Starting prediction loop from component.");
      startPredictionLoop(localVideoRef.current, canvasRef.current);
    }

    // Cleanup function to stop the loop when the component unmounts or dependencies change
    return () => {
      console.log("Stopping prediction loop from component cleanup.");
      stopPredictionLoop();
    };
  }, [isInCall, localStream, startPredictionLoop, stopPredictionLoop]);

  // Set up local video stream
  useEffect(() => {
    if (localStream) {
      const videoTrack = localStream.getVideoTracks()[0];
      if (videoTrack) {
        const videoElement = document.createElement("video");
        videoElement.srcObject = localStream;
        videoElement.autoplay = true;
        videoElement.muted = true;
        localVideoRef.current = videoElement;
      }
    }
  }, [localStream]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      isEndingCallRef.current = true;
      if (socketRef.current) {
        socketRef.current.close(1000, "Component unmounting");
      }
      if (localStream) {
        localStream.getTracks().forEach((track) => track.stop());
      }
      if (predictionIntervalRef.current) {
        clearInterval(predictionIntervalRef.current);
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (connectionTimeoutRef.current) {
        clearTimeout(connectionTimeoutRef.current);
      }
    };
  }, [localStream]);

  return {
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

    // Prediction actions
    startPredictionLoop,
    stopPredictionLoop,

    // Chat
    messages,
    sendMessage,

    // Settings
    settings,
    updateSettings,
  };
}
