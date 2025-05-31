"use client";

import { Suspense, useState, useRef, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  OrbitControls,
  Text3D,
  Environment,
  Html,
  PerspectiveCamera,
  ContactShadows,
} from "@react-three/drei";
import { motion } from "framer-motion";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Hand,
  ArrowLeft,
  Play,
  RotateCcw,
  Sparkles,
  Activity,
  Settings,
} from "lucide-react";
import * as THREE from "three";

// Enhanced anatomical hand model with realistic finger articulation
function RealisticHandModel({
  gesture,
  isAnimating,
}: {
  gesture: string;
  isAnimating: boolean;
}) {
  const handRef = useRef<THREE.Group>(null);
  const { clock } = useThree();

  // Finger joint references
  const thumbRef = useRef<THREE.Group>(null);
  const indexRef = useRef<THREE.Group>(null);
  const middleRef = useRef<THREE.Group>(null);
  const ringRef = useRef<THREE.Group>(null);
  const pinkyRef = useRef<THREE.Group>(null);

  // Finger joint references - second knuckles
  const thumbKnuckleRef = useRef<THREE.Group>(null);
  const indexKnuckleRef = useRef<THREE.Group>(null);
  const middleKnuckleRef = useRef<THREE.Group>(null);
  const ringKnuckleRef = useRef<THREE.Group>(null);
  const pinkyKnuckleRef = useRef<THREE.Group>(null);

  // Finger tip references
  const thumbTipRef = useRef<THREE.Group>(null);
  const indexTipRef = useRef<THREE.Group>(null);
  const middleTipRef = useRef<THREE.Group>(null);
  const ringTipRef = useRef<THREE.Group>(null);
  const pinkyTipRef = useRef<THREE.Group>(null);

  // Base glow settings for attractive visualization
  const [glowIntensity, setGlowIntensity] = useState(0.1);
  const [glowColor, setGlowColor] = useState(new THREE.Color("#00ffaa"));

  const baseRefs = [thumbRef, indexRef, middleRef, ringRef, pinkyRef];
  const knuckleRefs = [
    thumbKnuckleRef,
    indexKnuckleRef,
    middleKnuckleRef,
    ringKnuckleRef,
    pinkyKnuckleRef,
  ];
  const tipRefs = [
    thumbTipRef,
    indexTipRef,
    middleTipRef,
    ringTipRef,
    pinkyTipRef,
  ];

  // Apply gesture-specific finger positions
  useEffect(() => {
    if (!handRef.current) return;

    resetFingerPositions();

    switch (gesture) {
      case "Hello":
        applyHelloGesture();
        setGlowColor(new THREE.Color("#00ffaa"));
        break;
      case "Peace":
        applyPeaceGesture();
        setGlowColor(new THREE.Color("#00aaff"));
        break;
      case "Thumbs Up":
        applyThumbsUpGesture();
        setGlowColor(new THREE.Color("#ffaa00"));
        break;
      case "Thank You":
        applyThankYouGesture();
        setGlowColor(new THREE.Color("#ff00aa"));
        break;
      case "Please":
        applyPleaseGesture();
        setGlowColor(new THREE.Color("#aa00ff"));
        break;
      case "Sorry":
        applySorryGesture();
        setGlowColor(new THREE.Color("#ff5500"));
        break;
      case "Love": // ILY sign
        applyLoveGesture();
        setGlowColor(new THREE.Color("#ff0055"));
        break;
      case "Stop":
        applyStopGesture();
        setGlowColor(new THREE.Color("#ff0000"));
        break;
      default:
        applyDefaultGesture();
        setGlowColor(new THREE.Color("#00ffaa"));
    }
  }, [gesture]); // eslint-disable-line react-hooks/exhaustive-deps

  const checkAllBaseRefs = () => {
    return (
      thumbRef.current &&
      indexRef.current &&
      middleRef.current &&
      ringRef.current &&
      pinkyRef.current
    );
  };

  // Reset all finger positions to default
  const resetFingerPositions = () => {
    if (!checkAllBaseRefs()) return;

    baseRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0));
    knuckleRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0));
    tipRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0));
  };

  // Apply "Hello" gesture - all fingers extended and slight waving motion
  const applyHelloGesture = () => {
    if (!checkAllBaseRefs()) return;

    thumbRef.current!.rotation.set(0, 0, -0.3);
    indexRef.current!.rotation.set(0, 0, -0.1);
    middleRef.current!.rotation.set(0, 0, 0);
    ringRef.current!.rotation.set(0, 0, 0.1);
    pinkyRef.current!.rotation.set(0, 0, 0.2);

    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0, 0, -0.1);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(-0.1, 0, 0);
  };

  // Apply "Peace" gesture - index and middle fingers extended, others curled
  const applyPeaceGesture = () => {
    if (!checkAllBaseRefs()) return;

    // Curl thumb, ring, and pinky
    thumbRef.current!.rotation.set(Math.PI / 4, Math.PI / 6, Math.PI / 3);
    ringRef.current!.rotation.set(Math.PI / 2, 0, 0.1);
    pinkyRef.current!.rotation.set(Math.PI / 2, 0, 0.15);

    // Extend index and middle, slightly spread for V shape
    indexRef.current!.rotation.set(0, -0.1, -0.05);
    middleRef.current!.rotation.set(0, 0.1, 0);

    // Knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(Math.PI / 3, 0, 0);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (ringKnuckleRef.current)
      ringKnuckleRef.current.rotation.set(Math.PI / 2, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(Math.PI / 2, 0, 0);

    // Tips
    if (thumbTipRef.current)
      thumbTipRef.current.rotation.set(Math.PI / 4, 0, 0);
    // Index and Middle tips straight
    if (indexTipRef.current) indexTipRef.current.rotation.set(0, 0, 0);
    if (middleTipRef.current) middleTipRef.current.rotation.set(0, 0, 0);
    if (ringTipRef.current) ringTipRef.current.rotation.set(Math.PI / 3, 0, 0);
    if (pinkyTipRef.current)
      pinkyTipRef.current.rotation.set(Math.PI / 3, 0, 0);
  };

  // Apply "Thumbs Up" gesture
  const applyThumbsUpGesture = () => {
    if (!checkAllBaseRefs()) return;
    const curlAngle = (Math.PI / 2) * 0.9; // For fist
    const tipCurlAngle = Math.PI / 3;

    // Extend thumb upward
    thumbRef.current!.rotation.set(-0.5, 0, -Math.PI / 3);
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(-0.2, 0, 0);
    if (thumbTipRef.current) thumbTipRef.current.rotation.set(-0.2, 0, 0);

    // Curl other fingers into a fist
    [indexRef, middleRef, ringRef, pinkyRef].forEach((ref) =>
      ref.current!.rotation.set(curlAngle, 0, 0)
    );
    [
      indexKnuckleRef,
      middleKnuckleRef,
      ringKnuckleRef,
      pinkyKnuckleRef,
    ].forEach((ref) => ref.current?.rotation.set(curlAngle, 0, 0));
    [indexTipRef, middleTipRef, ringTipRef, pinkyTipRef].forEach(
      (ref) => ref.current?.rotation.set(tipCurlAngle, 0, 0)
    );
  };

  // Apply "Thank You" gesture - palm open, slight bow
  const applyThankYouGesture = () => {
    if (!checkAllBaseRefs()) return;
    // Open palm with slight finger spread
    thumbRef.current!.rotation.set(0, 0, -0.4);
    indexRef.current!.rotation.set(0, 0, -0.2);
    middleRef.current!.rotation.set(0, 0, 0);
    ringRef.current!.rotation.set(0, 0, 0.2);
    pinkyRef.current!.rotation.set(0, 0, 0.4);

    // Slight bend at knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(-0.2, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(-0.2, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(-0.2, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(-0.2, 0, 0);
  };

  // Apply "Please" gesture - flat hand (as if on chest or for prayer)
  const applyPleaseGesture = () => {
    if (!checkAllBaseRefs()) return;
    // Fingers straight and together
    thumbRef.current!.rotation.set(0.2, 0, -0.1);
    indexRef.current!.rotation.set(0, 0, 0);
    middleRef.current!.rotation.set(0, 0, 0);
    ringRef.current!.rotation.set(0, 0, 0);
    pinkyRef.current!.rotation.set(0, 0, 0);

    // Slight bend at knuckles for natural flat hand
    [
      thumbKnuckleRef,
      indexKnuckleRef,
      middleKnuckleRef,
      ringKnuckleRef,
      pinkyKnuckleRef,
    ].forEach((ref) => {
      if (ref.current) ref.current.rotation.set(-0.1, 0, 0);
    });
  };

  // Apply "Sorry" gesture - fist (as if to rub chest)
  const applySorryGesture = () => {
    if (!checkAllBaseRefs()) return;
    const curlAngle = (Math.PI / 2) * 0.9;
    const tipCurlAngle = Math.PI / 3;

    // Curl fingers into a fist
    [indexRef, middleRef, ringRef, pinkyRef].forEach((ref) =>
      ref.current!.rotation.set(curlAngle, 0, 0)
    );
    [
      indexKnuckleRef,
      middleKnuckleRef,
      ringKnuckleRef,
      pinkyKnuckleRef,
    ].forEach((ref) => ref.current?.rotation.set(curlAngle, 0, 0));
    [indexTipRef, middleTipRef, ringTipRef, pinkyTipRef].forEach(
      (ref) => ref.current?.rotation.set(tipCurlAngle, 0, 0)
    );
    // Thumb curls over fingers
    thumbRef.current!.rotation.set(0.6, 0.2, -0.2);
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0.5, 0, 0);
    if (thumbTipRef.current) thumbTipRef.current.rotation.set(0.5, 0, 0);
  };

  // Apply "Love" gesture - ILY sign
  const applyLoveGesture = () => {
    if (!checkAllBaseRefs()) return;
    const curlAngle = (Math.PI / 2) * 0.95; // For middle and ring fingers
    const tipCurlAngle = (Math.PI / 2) * 0.8;

    // Thumb extended and abducted
    thumbRef.current!.rotation.set(0.2, 0, -Math.PI / 3.5);
    if (thumbKnuckleRef.current) thumbKnuckleRef.current.rotation.set(0, 0, 0);
    if (thumbTipRef.current) thumbTipRef.current.rotation.set(0, 0, 0);

    // Index finger extended
    indexRef.current!.rotation.set(0, 0, 0);
    if (indexKnuckleRef.current) indexKnuckleRef.current.rotation.set(0, 0, 0);
    if (indexTipRef.current) indexTipRef.current.rotation.set(0, 0, 0);

    // Middle finger curled
    middleRef.current!.rotation.set(curlAngle, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(curlAngle, 0, 0);
    if (middleTipRef.current)
      middleTipRef.current.rotation.set(tipCurlAngle, 0, 0);

    // Ring finger curled
    ringRef.current!.rotation.set(curlAngle, 0, 0);
    if (ringKnuckleRef.current)
      ringKnuckleRef.current.rotation.set(curlAngle, 0, 0);
    if (ringTipRef.current) ringTipRef.current.rotation.set(tipCurlAngle, 0, 0);

    // Pinky finger extended
    pinkyRef.current!.rotation.set(0, 0, 0);
    if (pinkyKnuckleRef.current) pinkyKnuckleRef.current.rotation.set(0, 0, 0);
    if (pinkyTipRef.current) pinkyTipRef.current.rotation.set(0, 0, 0);
  };

  // Apply "Stop" gesture - palm forward, fingers extended
  const applyStopGesture = () => {
    if (!checkAllBaseRefs()) return;
    // All fingers extended straight
    thumbRef.current!.rotation.set(0, 0, -0.4);
    indexRef.current!.rotation.set(0, 0, -0.05);
    middleRef.current!.rotation.set(0, 0, 0);
    ringRef.current!.rotation.set(0, 0, 0.05);
    pinkyRef.current!.rotation.set(0, 0, 0.1);

    // No bend at knuckles - straight and firm
    knuckleRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0));
    tipRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0));
  };

  // Apply default gesture - relaxed open hand
  const applyDefaultGesture = () => {
    if (!checkAllBaseRefs()) return;
    // Relaxed finger positions
    thumbRef.current!.rotation.set(0.1, 0, -0.3);
    indexRef.current!.rotation.set(0.1, 0, -0.1);
    middleRef.current!.rotation.set(0.15, 0, 0);
    ringRef.current!.rotation.set(0.15, 0, 0.1);
    pinkyRef.current!.rotation.set(0.1, 0, 0.2);

    // Natural curve at knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0.1, 0, 0);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(0.2, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(0.25, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(0.25, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(0.2, 0, 0);

    // Slight bend at tips
    tipRefs.forEach((ref) => ref.current?.rotation.set(0.1, 0, 0));
  };

  useFrame(() => {
    if (handRef.current && isAnimating) {
      handRef.current.rotation.y = Math.sin(clock.elapsedTime * 0.5) * 0.2;
      handRef.current.position.y = Math.sin(clock.elapsedTime * 0.7) * 0.1;
      setGlowIntensity(0.1); // Keep glow stable for non-neural mode

      if (isAnimating && thumbRef.current) {
        thumbRef.current.rotation.z += Math.sin(clock.elapsedTime * 3) * 0.001;
        if (indexRef.current)
          indexRef.current.rotation.z +=
            Math.sin(clock.elapsedTime * 2.5) * 0.0005;
        if (middleRef.current)
          middleRef.current.rotation.z +=
            Math.sin(clock.elapsedTime * 2.7) * 0.0005;
        if (ringRef.current)
          ringRef.current.rotation.z +=
            Math.sin(clock.elapsedTime * 2.9) * 0.0005;
        if (pinkyRef.current)
          pinkyRef.current.rotation.z +=
            Math.sin(clock.elapsedTime * 3.1) * 0.0005;
      }
    } else if (handRef.current) {
      setGlowIntensity(0.1); // Ensure glow intensity is stable when not animating
    }
  });

  const FingerSegment = ({
    width,
    height,
    depth,
    position = [0, 0, 0],
    isJoint = false,
  }: {
    width: number;
    height: number;
    depth: number;
    position?: [number, number, number];
    isJoint?: boolean;
  }) => (
    <mesh position={position} castShadow receiveShadow>
      {isJoint ? (
        <sphereGeometry args={[width * 0.6, 12, 12]} />
      ) : (
        <capsuleGeometry args={[width * 0.5, height, 8, 16]} />
      )}
      <meshStandardMaterial
        color={glowColor}
        emissive={glowColor}
        emissiveIntensity={0.05} // Consistent low emissive for non-neural
        roughness={0.3}
        metalness={0.7}
        transparent
        opacity={1.0}
      />
    </mesh>
  );

  const currentEmissivePalm = 0.05;
  const currentEmissiveWrist = 0.05;
  const currentEmissiveTip = 0.1;
  const currentEmissiveKnuckleBump = 0.05;

  return (
    <group
      ref={handRef}
      position={[0, 0, 0]}
      rotation={[0.2, 0, 0]}
      scale={1.5}
    >
      <mesh castShadow receiveShadow>
        <boxGeometry args={[1.2, 1.8, 0.4]} />
        <meshStandardMaterial
          color={glowColor}
          emissive={glowColor}
          emissiveIntensity={currentEmissivePalm}
          roughness={0.4}
          metalness={0.6}
          transparent
          opacity={1.0}
        />
      </mesh>
      <mesh position={[0, -1.2, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.4, 0.5, 0.6, 16]} />
        <meshStandardMaterial
          color={glowColor}
          emissive={glowColor}
          emissiveIntensity={currentEmissiveWrist}
          roughness={0.4}
          metalness={0.6}
          transparent
          opacity={1.0}
        />
      </mesh>
      {/* Thumb */}
      <group
        ref={thumbRef}
        position={[-0.7, -0.2, 0.1]}
        rotation={[0, 0, -0.3]}
      >
        <FingerSegment width={0.25} height={0.6} depth={0.25} />
        <group
          ref={thumbKnuckleRef}
          position={[0, 0.6, 0]}
          rotation={[0, 0, -0.1]}
        >
          <FingerSegment width={0.22} height={0.4} depth={0.22} isJoint />
          <group ref={thumbTipRef} position={[0, 0.4, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.12, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={currentEmissiveTip}
                roughness={0.2}
                metalness={0.8}
                opacity={1.0}
              />
            </mesh>
          </group>
        </group>
      </group>
      {/* Index finger */}
      <group ref={indexRef} position={[-0.4, 0.9, 0.1]} rotation={[0, 0, -0.1]}>
        <FingerSegment width={0.22} height={0.7} depth={0.22} />
        <group
          ref={indexKnuckleRef}
          position={[0, 0.7, 0]}
          rotation={[-0.1, 0, 0]}
        >
          <FingerSegment width={0.2} height={0.5} depth={0.2} />
          <group ref={indexTipRef} position={[0, 0.5, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.11, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={currentEmissiveTip}
                roughness={0.2}
                metalness={0.8}
                opacity={1.0}
              />
            </mesh>
          </group>
        </group>
      </group>
      {/* Middle finger */}
      <group ref={middleRef} position={[0, 0.9, 0.1]} rotation={[0, 0, 0]}>
        <FingerSegment width={0.22} height={0.8} depth={0.22} />
        <group
          ref={middleKnuckleRef}
          position={[0, 0.8, 0]}
          rotation={[-0.1, 0, 0]}
        >
          <FingerSegment width={0.2} height={0.5} depth={0.2} />
          <group ref={middleTipRef} position={[0, 0.5, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.11, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={currentEmissiveTip}
                roughness={0.2}
                metalness={0.8}
                opacity={1.0}
              />
            </mesh>
          </group>
        </group>
      </group>
      {/* Ring finger */}
      <group ref={ringRef} position={[0.4, 0.9, 0.1]} rotation={[0, 0, 0.1]}>
        <FingerSegment width={0.22} height={0.7} depth={0.22} />
        <group
          ref={ringKnuckleRef}
          position={[0, 0.7, 0]}
          rotation={[-0.1, 0, 0]}
        >
          <FingerSegment width={0.2} height={0.5} depth={0.2} />
          <group ref={ringTipRef} position={[0, 0.5, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.11, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={currentEmissiveTip}
                roughness={0.2}
                metalness={0.8}
                opacity={1.0}
              />
            </mesh>
          </group>
        </group>
      </group>
      {/* Pinky finger */}
      <group ref={pinkyRef} position={[0.7, 0.8, 0.1]} rotation={[0, 0, 0.2]}>
        <FingerSegment width={0.2} height={0.6} depth={0.2} />
        <group
          ref={pinkyKnuckleRef}
          position={[0, 0.6, 0]}
          rotation={[-0.1, 0, 0]}
        >
          <FingerSegment width={0.18} height={0.4} depth={0.18} />
          <group ref={pinkyTipRef} position={[0, 0.4, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.1, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={currentEmissiveTip}
                roughness={0.2}
                metalness={0.8}
                opacity={1.0}
              />
            </mesh>
          </group>
        </group>
      </group>
      {/* Anatomical details - knuckle bumps */}
      {[
        [-0.4, 0.4, 0.15],
        [0, 0.4, 0.15],
        [0.4, 0.4, 0.15],
        [0.7, 0.3, 0.15],
      ].map((pos, i) => (
        <mesh key={i} position={pos as [number, number, number]} castShadow>
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshStandardMaterial
            color={glowColor}
            emissive={glowColor}
            emissiveIntensity={currentEmissiveKnuckleBump}
            transparent
            opacity={0.8}
            roughness={0.5}
            metalness={0.5}
          />
        </mesh>
      ))}
    </group>
  );
}

export default function VisualizePage() {
  const [inputGesture, setInputGesture] = useState("");
  const [currentGesture, setCurrentGesture] = useState("Hello");
  const [isAnimating, setIsAnimating] = useState(true);

  const predefinedGestures = [
    "Hello",
    "Peace",
    "Thumbs Up",
    "Thank You",
    "Please",
    "Sorry",
    "Love",
    "Stop",
  ];

  const handleGenerateGesture = () => {
    if (inputGesture.trim()) {
      setCurrentGesture(inputGesture.trim());
    }
  };

  const handleRandomGesture = () => {
    const randomGesture =
      predefinedGestures[Math.floor(Math.random() * predefinedGestures.length)];
    setCurrentGesture(randomGesture);
    setInputGesture(randomGesture);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-cyan-950 relative overflow-hidden">
      {/* Navigation */}
      <div className="pt-24 pb-12 px-6 relative z-10">
        <div className="container mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-cyan-300 via-purple-300 to-pink-300 bg-clip-text text-transparent">
              3D Gesture Visualization
            </h1>
            <p className="text-cyan-100/70 text-xl max-w-3xl mx-auto">
              Explore 3D hand gestures with detailed articulation and controls.
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-4 gap-8">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="lg:col-span-1"
            >
              <Card className="bg-gradient-to-br from-slate-900/50 to-purple-900/30 border border-cyan-400/20 backdrop-blur-xl">
                <CardHeader>
                  <CardTitle className="flex items-center text-cyan-100">
                    <Sparkles className="mr-2 h-6 w-6 text-cyan-400" />
                    Gesture Controls
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <Label
                      htmlFor="gesture-input"
                      className="text-cyan-100/90 mb-3 block"
                    >
                      Gesture Input
                    </Label>
                    <Input
                      id="gesture-input"
                      value={inputGesture}
                      onChange={(e) => setInputGesture(e.target.value)}
                      placeholder="e.g., Hello, Peace, Love..."
                      className="bg-slate-800/50 border-cyan-400/30 text-cyan-100 placeholder:text-cyan-400/50"
                    />
                  </div>

                  <Button
                    onClick={handleGenerateGesture}
                    className="w-full bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-700 hover:to-purple-700 text-white py-3"
                  >
                    <Play className="mr-2 h-4 w-4" /> Generate Gesture
                  </Button>

                  <Button
                    onClick={handleRandomGesture}
                    variant="outline"
                    className="w-full border-cyan-400/30 text-cyan-300 hover:bg-cyan-400/10 py-3"
                  >
                    <RotateCcw className="mr-2 h-4 w-4" /> Random Gesture
                  </Button>

                  <div className="space-y-3">
                    <Label className="text-cyan-100/90">Quick Gestures</Label>
                    <div className="grid grid-cols-2 gap-2">
                      {predefinedGestures.map((gesture) => (
                        <Button
                          key={gesture}
                          onClick={() => {
                            setCurrentGesture(gesture);
                            setInputGesture(gesture);
                          }}
                          variant="outline"
                          size="sm"
                          className="border-cyan-400/20 text-cyan-300 hover:bg-cyan-400/10 text-xs py-2"
                        >
                          {gesture}
                        </Button>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <Label className="text-cyan-100/90">Display Settings</Label>
                    <div className="space-y-3">
                      <Button
                        onClick={() => setIsAnimating(!isAnimating)}
                        variant="outline"
                        size="sm"
                        className="w-full border-cyan-400/20 text-cyan-300 hover:bg-cyan-400/10"
                      >
                        <Activity className="mr-2 h-4 w-4" />{" "}
                        {isAnimating ? "Pause" : "Resume"} Animation
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
              className="lg:col-span-3"
            >
              <Card className="bg-gradient-to-br from-slate-900/50 to-purple-900/30 border border-cyan-400/20 backdrop-blur-xl">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between text-cyan-100">
                    <span className="flex items-center">
                      <Hand className="mr-3 h-6 w-6 text-cyan-400" />
                      Gesture:
                      <span className="ml-2 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                        {currentGesture}
                      </span>
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-[700px] w-full rounded-lg overflow-hidden relative">
                    <Canvas shadows>
                      <PerspectiveCamera makeDefault position={[0, 2, 10]} />
                      <ambientLight intensity={0.6} />
                      <pointLight
                        position={[10, 10, 10]}
                        intensity={0.8}
                        color={"#ffffff"}
                        castShadow
                      />
                      <pointLight
                        position={[-10, -10, -10]}
                        intensity={0.4}
                        color={"#ffffff"}
                      />
                      <pointLight
                        position={[0, 15, 0]}
                        intensity={0.3}
                        color={"#ffffff"}
                      />
                      <Suspense
                        fallback={
                          <Html center>
                            <div className="text-cyan-400 text-xl font-mono">
                              <motion.div
                                animate={{ rotate: 360 }}
                                transition={{
                                  duration: 2,
                                  repeat: Number.POSITIVE_INFINITY,
                                  ease: "linear",
                                }}
                                className="inline-block mr-2"
                              >
                                <Hand className="h-6 w-6" />
                              </motion.div>
                              Loading 3D Viewer...
                            </div>
                          </Html>
                        }
                      >
                        <RealisticHandModel
                          gesture={currentGesture}
                          isAnimating={isAnimating}
                        />
                        <Text3D
                          font="/fonts/helvetiker_regular.typeface.json" 
                          size={0.6}
                          height={0.15}
                          position={[0, -3, 0]}
                          rotation={[0, 0, 0]}
                        >
                          {currentGesture}
                          <meshStandardMaterial
                            color="#00ffaa"
                            emissive="#333333"
                            emissiveIntensity={0.1}
                          />
                        </Text3D>
                        <ContactShadows
                          position={[0, -2.5, 0]}
                          opacity={0.4}
                          scale={10}
                          blur={2.5}
                          far={4}
                        />
                        <Environment preset="city" />
                      </Suspense>
                      <OrbitControls
                        enablePan={true}
                        enableZoom={true}
                        enableRotate={true}
                        minDistance={4}
                        maxDistance={20}
                        autoRotate={isAnimating}
                        autoRotateSpeed={0.5}
                      />
                    </Canvas>
                    <div className="absolute top-4 right-4 bg-gradient-to-r from-slate-900/80 to-purple-900/80 backdrop-blur-xl rounded-lg p-4 border border-cyan-400/30">
                      <div className="text-cyan-100 text-sm">
                        <div className="flex items-center mb-2">
                          <Activity className="h-4 w-4 text-cyan-400 mr-2" />
                          <span>Display Status</span>
                        </div>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-cyan-300/70">Mode:</span>
                            <span className="text-cyan-300">Standard</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-cyan-300/70">Animation:</span>
                            <span className="text-cyan-300">
                              {isAnimating ? "Active" : "Paused"}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-cyan-300/70">Gesture:</span>
                            <span className="text-cyan-300">
                              {currentGesture}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="mt-8"
          >
            <Card className="bg-gradient-to-br from-slate-900/50 to-purple-900/30 border border-cyan-400/20 backdrop-blur-xl">
              <CardContent className="p-6">
                <h3 className="text-cyan-100 font-semibold mb-4 flex items-center">
                  <Settings className="mr-2 h-5 w-5 text-cyan-400" />
                  3D Viewer Controls & Features
                </h3>
                <div className="grid md:grid-cols-4 gap-6 text-sm text-cyan-100/70">
                  <div>
                    <strong className="text-cyan-300">Rotate:</strong> Click and
                    drag to rotate the hand model.
                  </div>
                  <div>
                    <strong className="text-cyan-300">Zoom:</strong> Mouse wheel
                    or pinch to zoom in/out.
                  </div>
                  <div>
                    <strong className="text-cyan-300">Pan:</strong> Right-click
                    and drag to pan view.
                  </div>
                  <div>
                    <strong className="text-cyan-300">Auto-Rotate:</strong>{" "}
                    Enabled when animation is active.
                  </div>
                </div>
                <div className="mt-4 p-4 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-lg border border-cyan-400/20">
                  <p className="text-cyan-100/80 text-sm">
                    <strong className="text-cyan-300">
                      Enhanced Hand Model:
                    </strong>{" "}
                    The hand model features realistic finger articulation with
                    proper joint movement. The gesture-specific animations
                    provide an immersive experience.
                  </p>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
