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
  Network,
  Cpu,
  Activity,
  Zap,
  Eye,
  Settings,
} from "lucide-react";
import * as THREE from "three";

// Enhanced anatomical hand model with realistic finger articulation
function RealisticHandModel({
  gesture,
  isAnimating,
  neuralMode,
}: {
  gesture: string;
  isAnimating: boolean;
  neuralMode: boolean;
}) {
  // Added neuralMode prop
  const handRef = useRef<THREE.Group>(null); // [cite: 2]
  const [neuralPulse, setNeuralPulse] = useState(0);
  const { clock } = useThree();

  // Finger joint references
  const thumbRef = useRef<THREE.Group>(null); // [cite: 2]
  const indexRef = useRef<THREE.Group>(null); // [cite: 2]
  const middleRef = useRef<THREE.Group>(null); // [cite: 2]
  const ringRef = useRef<THREE.Group>(null); // [cite: 2]
  const pinkyRef = useRef<THREE.Group>(null); // [cite: 2]

  // Finger joint references - second knuckles
  const thumbKnuckleRef = useRef<THREE.Group>(null); // [cite: 2]
  const indexKnuckleRef = useRef<THREE.Group>(null); // [cite: 2]
  const middleKnuckleRef = useRef<THREE.Group>(null); // [cite: 2]
  const ringKnuckleRef = useRef<THREE.Group>(null); // [cite: 2]
  const pinkyKnuckleRef = useRef<THREE.Group>(null); // [cite: 2]

  // Finger tip references
  const thumbTipRef = useRef<THREE.Group>(null); // [cite: 2]
  const indexTipRef = useRef<THREE.Group>(null); // [cite: 2]
  const middleTipRef = useRef<THREE.Group>(null); // [cite: 2]
  const ringTipRef = useRef<THREE.Group>(null); // [cite: 3]
  const pinkyTipRef = useRef<THREE.Group>(null); // [cite: 3]

  // Neural glow effect
  const [glowIntensity, setGlowIntensity] = useState(0.2); // [cite: 3]
  const [glowColor, setGlowColor] = useState(new THREE.Color("#00ffaa")); // [cite: 3]

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
      case "Peace": // [cite: 4]
        applyPeaceGesture();
        setGlowColor(new THREE.Color("#00aaff")); // [cite: 4]
        break;
      case "Thumbs Up":
        applyThumbsUpGesture();
        setGlowColor(new THREE.Color("#ffaa00"));
        break;
      case "Thank You":
        applyThankYouGesture();
        setGlowColor(new THREE.Color("#ff00aa"));
        break;
      case "Please": // [cite: 5]
        applyPleaseGesture();
        setGlowColor(new THREE.Color("#aa00ff")); // [cite: 5]
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
        applyStopGesture(); // [cite: 6]
        setGlowColor(new THREE.Color("#ff0000")); // [cite: 6]
        break;
      default:
        applyDefaultGesture();
        setGlowColor(new THREE.Color("#00ffaa"));
    }
  }, [gesture]); // eslint-disable-line react-hooks/exhaustive-deps
  // Added eslint-disable for potentially missing dependencies if apply* functions were memoized differently

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
    if (!checkAllBaseRefs()) return; // [cite: 7]

    baseRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0));
    knuckleRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0));
    tipRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0)); // [cite: 8]
  };

  // Apply "Hello" gesture - all fingers extended and slight waving motion
  const applyHelloGesture = () => {
    if (!checkAllBaseRefs()) return; // [cite: 9]

    thumbRef.current!.rotation.set(0, 0, -0.3); // [cite: 9]
    indexRef.current!.rotation.set(0, 0, -0.1); // [cite: 9]
    middleRef.current!.rotation.set(0, 0, 0); // [cite: 9]
    ringRef.current!.rotation.set(0, 0, 0.1); // [cite: 9]
    pinkyRef.current!.rotation.set(0, 0, 0.2); // [cite: 9]

    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0, 0, -0.1); // [cite: 9]
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(-0.1, 0, 0); // [cite: 9]
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(-0.1, 0, 0); // [cite: 9]
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(-0.1, 0, 0); // [cite: 9]
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(-0.1, 0, 0); // [cite: 9]
  };

  // Apply "Peace" gesture - index and middle fingers extended, others curled
  const applyPeaceGesture = () => {
    // [cite: 10]
    if (!checkAllBaseRefs()) return; // [cite: 10]

    // Curl thumb, ring, and pinky
    thumbRef.current!.rotation.set(Math.PI / 4, Math.PI / 6, Math.PI / 3); // Adjusted for a more natural thumb curl [cite: 10]
    ringRef.current!.rotation.set(Math.PI / 2, 0, 0.1); // Increased curl [cite: 10]
    pinkyRef.current!.rotation.set(Math.PI / 2, 0, 0.15); // Increased curl [cite: 10]

    // Extend index and middle, slightly spread for V shape
    indexRef.current!.rotation.set(0, -0.1, -0.05); // Spread slightly
    middleRef.current!.rotation.set(0, 0.1, 0); // Spread slightly [cite: 10]

    // Knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(Math.PI / 3, 0, 0); // [cite: 10]
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(-0.1, 0, 0); // [cite: 10]
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(-0.1, 0, 0); // [cite: 11]
    if (ringKnuckleRef.current)
      ringKnuckleRef.current.rotation.set(Math.PI / 2, 0, 0); // [cite: 11]
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(Math.PI / 2, 0, 0); // [cite: 11]

    // Tips
    if (thumbTipRef.current)
      thumbTipRef.current.rotation.set(Math.PI / 4, 0, 0); // [cite: 11]
    // Index and Middle tips straight
    if (indexTipRef.current) indexTipRef.current.rotation.set(0, 0, 0); // [cite: 11]
    if (middleTipRef.current) middleTipRef.current.rotation.set(0, 0, 0); // [cite: 11]
    if (ringTipRef.current) ringTipRef.current.rotation.set(Math.PI / 3, 0, 0); // [cite: 11]
    if (pinkyTipRef.current)
      pinkyTipRef.current.rotation.set(Math.PI / 3, 0, 0); // [cite: 11]
  };

  // Apply "Thumbs Up" gesture
  const applyThumbsUpGesture = () => {
    if (!checkAllBaseRefs()) return; // [cite: 12]
    const curlAngle = (Math.PI / 2) * 0.9; // For fist
    const tipCurlAngle = Math.PI / 3;

    // Extend thumb upward
    thumbRef.current!.rotation.set(-0.5, 0, -Math.PI / 3); // Thumb up and slightly angled [cite: 12]
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(-0.2, 0, 0); // Slightly extend knuckle [cite: 12]
    if (thumbTipRef.current) thumbTipRef.current.rotation.set(-0.2, 0, 0); // Slightly extend tip [cite: 13]

    // Curl other fingers into a fist
    [indexRef, middleRef, ringRef, pinkyRef].forEach((ref) =>
      ref.current!.rotation.set(curlAngle, 0, 0)
    );
    [
      indexKnuckleRef,
      middleKnuckleRef,
      ringKnuckleRef,
      pinkyKnuckleRef,
    ].forEach((ref) => ref.current?.rotation.set(curlAngle, 0, 0)); // [cite: 12]
    [indexTipRef, middleTipRef, ringTipRef, pinkyTipRef].forEach(
      (ref) => ref.current?.rotation.set(tipCurlAngle, 0, 0)
    ); // [cite: 13]
  };

  // Apply "Thank You" gesture - palm open, slight bow
  const applyThankYouGesture = () => {
    if (!checkAllBaseRefs()) return; // [cite: 14]
    // Open palm with slight finger spread
    thumbRef.current!.rotation.set(0, 0, -0.4); // [cite: 14]
    indexRef.current!.rotation.set(0, 0, -0.2); // [cite: 14]
    middleRef.current!.rotation.set(0, 0, 0); // [cite: 14]
    ringRef.current!.rotation.set(0, 0, 0.2); // [cite: 14]
    pinkyRef.current!.rotation.set(0, 0, 0.4); // [cite: 14]

    // Slight bend at knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(-0.1, 0, 0); // [cite: 14]
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(-0.2, 0, 0); // [cite: 14]
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(-0.2, 0, 0); // [cite: 14]
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(-0.2, 0, 0); // [cite: 14]
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(-0.2, 0, 0); // [cite: 14]
  };

  // Apply "Please" gesture - flat hand (as if on chest or for prayer)
  const applyPleaseGesture = () => {
    // [cite: 15]
    if (!checkAllBaseRefs()) return; // [cite: 15]
    // Fingers straight and together
    thumbRef.current!.rotation.set(0.2, 0, -0.1); // Thumb slightly adducted and flexed [cite: 15]
    indexRef.current!.rotation.set(0, 0, 0); // [cite: 15]
    middleRef.current!.rotation.set(0, 0, 0); // [cite: 15]
    ringRef.current!.rotation.set(0, 0, 0); // [cite: 15]
    pinkyRef.current!.rotation.set(0, 0, 0); // [cite: 15]

    // Slight bend at knuckles for natural flat hand
    [
      thumbKnuckleRef,
      indexKnuckleRef,
      middleKnuckleRef,
      ringKnuckleRef,
      pinkyKnuckleRef,
    ].forEach((ref) => {
      if (ref.current) ref.current.rotation.set(-0.1, 0, 0); // [cite: 15, 16]
    });
  };

  // Apply "Sorry" gesture - fist (as if to rub chest)
  const applySorryGesture = () => {
    if (!checkAllBaseRefs()) return; // [cite: 17]
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
    ].forEach((ref) => ref.current?.rotation.set(curlAngle, 0, 0)); // [cite: 17]
    [indexTipRef, middleTipRef, ringTipRef, pinkyTipRef].forEach(
      (ref) => ref.current?.rotation.set(tipCurlAngle, 0, 0)
    );

    // Thumb curls over fingers
    thumbRef.current!.rotation.set(0.6, 0.2, -0.2); // Adjusted to curl over [cite: 17]
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0.5, 0, 0); // [cite: 17]
    if (thumbTipRef.current) thumbTipRef.current.rotation.set(0.5, 0, 0);
  };

  // Apply "Love" gesture - ILY sign
  const applyLoveGesture = () => {
    // [cite: 18]
    if (!checkAllBaseRefs()) return; // [cite: 18]
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
      middleKnuckleRef.current.rotation.set(curlAngle, 0, 0); // [cite: 18]
    if (middleTipRef.current)
      middleTipRef.current.rotation.set(tipCurlAngle, 0, 0);

    // Ring finger curled
    ringRef.current!.rotation.set(curlAngle, 0, 0);
    if (ringKnuckleRef.current)
      ringKnuckleRef.current.rotation.set(curlAngle, 0, 0); // [cite: 19]
    if (ringTipRef.current) ringTipRef.current.rotation.set(tipCurlAngle, 0, 0);

    // Pinky finger extended
    pinkyRef.current!.rotation.set(0, 0, 0);
    if (pinkyKnuckleRef.current) pinkyKnuckleRef.current.rotation.set(0, 0, 0); // [cite: 19]
    if (pinkyTipRef.current) pinkyTipRef.current.rotation.set(0, 0, 0);
  };

  // Apply "Stop" gesture - palm forward, fingers extended
  const applyStopGesture = () => {
    if (!checkAllBaseRefs()) return; // [cite: 20]
    // All fingers extended straight
    thumbRef.current!.rotation.set(0, 0, -0.4); // Abducted thumb [cite: 20]
    indexRef.current!.rotation.set(0, 0, -0.05); // Slightly adducted [cite: 20]
    middleRef.current!.rotation.set(0, 0, 0); // [cite: 20]
    ringRef.current!.rotation.set(0, 0, 0.05); // Slightly adducted [cite: 20]
    pinkyRef.current!.rotation.set(0, 0, 0.1); // Slightly adducted [cite: 20]

    // No bend at knuckles - straight and firm
    knuckleRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0)); // [cite: 20]
    tipRefs.forEach((ref) => ref.current?.rotation.set(0, 0, 0));
  };

  // Apply default gesture - relaxed open hand
  const applyDefaultGesture = () => {
    // [cite: 21]
    if (!checkAllBaseRefs()) return; // [cite: 21]
    // Relaxed finger positions
    thumbRef.current!.rotation.set(0.1, 0, -0.3); // [cite: 21]
    indexRef.current!.rotation.set(0.1, 0, -0.1); // [cite: 21]
    middleRef.current!.rotation.set(0.15, 0, 0); // [cite: 21]
    ringRef.current!.rotation.set(0.15, 0, 0.1); // [cite: 21]
    pinkyRef.current!.rotation.set(0.1, 0, 0.2); // [cite: 21]

    // Natural curve at knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0.1, 0, 0); // [cite: 21]
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(0.2, 0, 0); // [cite: 21]
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(0.25, 0, 0); // [cite: 21]
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(0.25, 0, 0); // [cite: 22]
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(0.2, 0, 0); // [cite: 22]

    // Slight bend at tips
    tipRefs.forEach((ref) => ref.current?.rotation.set(0.1, 0, 0)); // [cite: 22]
  };

  useFrame(() => {
    if (handRef.current && isAnimating) {
      handRef.current.rotation.y = Math.sin(clock.elapsedTime * 0.5) * 0.2; // [cite: 22]
      handRef.current.position.y = Math.sin(clock.elapsedTime * 0.7) * 0.1; // [cite: 23]

      if (neuralMode) {
        // Pulse effect only in neural mode
        const pulse = Math.sin(clock.elapsedTime * 2) * 0.5 + 0.5; // [cite: 23]
        setNeuralPulse(pulse);
        setGlowIntensity(0.2 + pulse * 0.3); // [cite: 23]
      } else {
        setNeuralPulse(0); // No pulse
        setGlowIntensity(0.1); // Minimal glow for non-neural
      }

      if (isAnimating && thumbRef.current) {
        thumbRef.current.rotation.z += Math.sin(clock.elapsedTime * 3) * 0.001; // [cite: 23]
        if (indexRef.current)
          indexRef.current.rotation.z +=
            Math.sin(clock.elapsedTime * 2.5) * 0.0005; // [cite: 24]
        if (middleRef.current)
          middleRef.current.rotation.z +=
            Math.sin(clock.elapsedTime * 2.7) * 0.0005; // [cite: 24]
        if (ringRef.current)
          ringRef.current.rotation.z +=
            Math.sin(clock.elapsedTime * 2.9) * 0.0005; // [cite: 24]
        if (pinkyRef.current)
          pinkyRef.current.rotation.z +=
            Math.sin(clock.elapsedTime * 3.1) * 0.0005; // [cite: 24]
      }
    } else if (handRef.current) {
      // Not animating, ensure glow intensity is stable
      setGlowIntensity(neuralMode ? 0.2 : 0.1);
    }
  });

  const FingerSegment = ({
    width,
    height,
    depth,
    position = [0, 0, 0], // [cite: 25]
    isJoint = false, // [cite: 25]
  }: {
    width: number;
    height: number;
    depth: number;
    position?: [number, number, number];
    isJoint?: boolean;
  }) => (
    <mesh position={position} castShadow receiveShadow>
      {isJoint ? ( // [cite: 26]
        <sphereGeometry args={[width * 0.6, 12, 12]} /> // [cite: 26]
      ) : (
        <capsuleGeometry args={[width * 0.5, height, 8, 16]} /> // [cite: 26]
      )}
      <meshStandardMaterial
        color={glowColor}
        emissive={glowColor}
        emissiveIntensity={neuralMode ? glowIntensity * 0.3 : 0.05} // Adjusted for neuralMode [cite: 26]
        roughness={0.3}
        metalness={0.7}
        transparent
        opacity={neuralMode ? 0.9 : 1.0} // Adjusted for neuralMode [cite: 27]
      />
    </mesh>
  );

  const currentOpacity = neuralMode ? 0.85 : 1.0;
  const currentEmissivePalm = neuralMode ? glowIntensity * 0.2 : 0.05; // [cite: 28]
  const currentEmissiveWrist = neuralMode ? glowIntensity * 0.15 : 0.05; // [cite: 29]
  const currentEmissiveTip = neuralMode ? glowIntensity : 0.1; // [cite: 32]
  const currentEmissiveKnuckleBump = neuralMode ? glowIntensity * 0.5 : 0.05; // [cite: 49]

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
          emissiveIntensity={currentEmissivePalm} // [cite: 28]
          roughness={0.4}
          metalness={0.6}
          transparent
          opacity={currentOpacity} // [cite: 28]
        />
      </mesh>
      <mesh position={[0, -1.2, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.4, 0.5, 0.6, 16]} />
        <meshStandardMaterial
          color={glowColor}
          emissive={glowColor}
          emissiveIntensity={currentEmissiveWrist} // [cite: 29]
          roughness={0.4}
          metalness={0.6}
          transparent
          opacity={currentOpacity} // [cite: 29]
        />
      </mesh>
      {/* Thumb */} // [cite: 30]
      <group
        ref={thumbRef}
        position={[-0.7, -0.2, 0.1]}
        rotation={[0, 0, -0.3]}
      >
        {" "}
        {/* [cite: 30] */}
        <FingerSegment width={0.25} height={0.6} depth={0.25} />
        <group
          ref={thumbKnuckleRef}
          position={[0, 0.6, 0]}
          rotation={[0, 0, -0.1]}
        >
          <FingerSegment width={0.22} height={0.4} depth={0.22} isJoint />
          <group ref={thumbTipRef} position={[0, 0.4, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              {" "}
              {/* [cite: 31] */}
              <sphereGeometry args={[0.12, 16, 16]} /> {/* [cite: 31] */}
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={currentEmissiveTip} // [cite: 31]
                roughness={0.2} // [cite: 32]
                metalness={0.8} // [cite: 32]
                opacity={neuralMode ? 1.0 : 1.0} // Tips should be opaque
              />
            </mesh>
          </group>
        </group>
      </group>
      {/* Index finger */}
      <group ref={indexRef} position={[-0.4, 0.9, 0.1]} rotation={[0, 0, -0.1]}>
        <FingerSegment width={0.22} height={0.7} depth={0.22} />{" "}
        {/* [cite: 33] */}
        <group
          ref={indexKnuckleRef}
          position={[0, 0.7, 0]}
          rotation={[-0.1, 0, 0]}
        >
          <FingerSegment width={0.2} height={0.5} depth={0.2} />
          <group ref={indexTipRef} position={[0, 0.5, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.11, 16, 16]} /> {/* [cite: 34] */}
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={currentEmissiveTip} // [cite: 34]
                roughness={0.2} // [cite: 34]
                metalness={0.8} // [cite: 35]
                opacity={neuralMode ? 1.0 : 1.0}
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
          {" "}
          {/* [cite: 36] */}
          <FingerSegment width={0.2} height={0.5} depth={0.2} />
          <group ref={middleTipRef} position={[0, 0.5, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.11, 16, 16]} />
              <meshStandardMaterial // [cite: 37]
                color={glowColor} // [cite: 37]
                emissive={glowColor} // [cite: 37]
                emissiveIntensity={currentEmissiveTip} // [cite: 37]
                roughness={0.2} // [cite: 37]
                metalness={0.8} // [cite: 38]
                opacity={neuralMode ? 1.0 : 1.0}
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
          <FingerSegment width={0.2} height={0.5} depth={0.2} />{" "}
          {/* [cite: 39] */}
          <group ref={ringTipRef} position={[0, 0.5, 0]} rotation={[0, 0, 0]}>
            {" "}
            {/* [cite: 3] */}
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.11, 16, 16]} />
              <meshStandardMaterial
                color={glowColor} // [cite: 40]
                emissive={glowColor} // [cite: 40]
                emissiveIntensity={currentEmissiveTip} // [cite: 40]
                roughness={0.2} // [cite: 40]
                metalness={0.8} // [cite: 40]
                opacity={neuralMode ? 1.0 : 1.0}
              />
            </mesh>
          </group>{" "}
          {/* [cite: 41] */}
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
            {" "}
            {/* [cite: 3, 42] */}
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.1, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor} // [cite: 43]
                emissiveIntensity={currentEmissiveTip} // [cite: 43]
                roughness={0.2} // [cite: 43]
                metalness={0.8} // [cite: 43]
                opacity={neuralMode ? 1.0 : 1.0}
              />
            </mesh>
          </group>
        </group>
      </group>{" "}
      {/* [cite: 44] */}
      {/* Neural connection lines - only in neuralMode and if animating */}
      {neuralMode &&
        isAnimating && ( // [cite: 44]
          <group>
            {Array.from({ length: 12 }).map((_, i) => (
              <mesh
                key={i}
                position={[0, 0, 0.2]}
                rotation={[0, 0, (i * Math.PI) / 6]}
              >
                <cylinderGeometry args={[0.01, 0.01, 2 + neuralPulse * 0.5]} />
                <meshStandardMaterial // [cite: 45]
                  color={glowColor} // [cite: 45]
                  emissive={glowColor} // [cite: 45]
                  emissiveIntensity={glowIntensity} // [cite: 45]
                  transparent
                  opacity={0.6} // [cite: 45]
                />
              </mesh> // [cite: 46]
            ))}
          </group>
        )}
      {/* Neural energy core - only in neuralMode */}
      {neuralMode && ( // [cite: 46]
        <mesh position={[0, 0, 0]} castShadow>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshStandardMaterial
            color={glowColor}
            emissive={glowColor}
            emissiveIntensity={glowIntensity * 1.5} // [cite: 47]
            transparent
            opacity={0.7} // [cite: 47]
            roughness={0.1} // [cite: 47]
            metalness={0.9} // [cite: 47]
          />
        </mesh>
      )}
      {/* Anatomical details - knuckle bumps, opacity tied to neuralMode */}
      {[
        [-0.4, 0.4, 0.15],
        [0, 0.4, 0.15],
        [0.4, 0.4, 0.15], // [cite: 48]
        [0.7, 0.3, 0.15], // [cite: 48]
      ].map((pos, i) => (
        <mesh key={i} position={pos as [number, number, number]} castShadow>
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshStandardMaterial
            color={glowColor}
            emissive={glowColor}
            emissiveIntensity={currentEmissiveKnuckleBump}
            transparent
            opacity={neuralMode ? 0.6 : 0.8} // Adjusted opacity [cite: 49]
            roughness={0.5} // [cite: 49]
            metalness={0.5} // [cite: 49]
          />
        </mesh>
      ))}
    </group>
  );
}

// Enhanced floating neural particles
function NeuralParticles() {
  const particlesRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y = state.clock.elapsedTime * 0.05; // [cite: 50]
      particlesRef.current.rotation.x =
        Math.sin(state.clock.elapsedTime * 0.1) * 0.1; // [cite: 50]
    }
  });

  return (
    <group ref={particlesRef}>
      {Array.from({ length: 80 }).map((_, i) => (
        <mesh
          key={i}
          position={[
            (Math.random() - 0.5) * 25,
            (Math.random() - 0.5) * 25,
            (Math.random() - 0.5) * 25,
          ]}
        >
          <sphereGeometry args={[0.03 + Math.random() * 0.05]} />
          <meshStandardMaterial
            color={
              i % 3 === 0 ? "#00ffaa" : i % 3 === 1 ? "#00aaff" : "#ffaa00"
            } // [cite: 51, 52]
            emissive={
              i % 3 === 0 ? "#00ffaa" : i % 3 === 1 ? "#00aaff" : "#ffaa00"
            } // [cite: 53]
            emissiveIntensity={0.4 + Math.random() * 0.3} // [cite: 53]
          />
        </mesh>
      ))}
    </group>
  );
}

// Neural connection grid
function NeuralGrid() {
  return (
    <group>
      {Array.from({ length: 20 }).map((_, i) => (
        <group key={i}>
          <mesh position={[0, 0, -10 + i]} rotation={[Math.PI / 2, 0, 0]}>
            {" "}
            {/* [cite: 54] */}
            <planeGeometry args={[20, 20]} />
            <meshBasicMaterial
              color="#00ffaa"
              transparent
              opacity={0.02}
              wireframe
            />
          </mesh>
        </group>
      ))}
    </group>
  );
}

export default function VisualizePage() {
  const [inputGesture, setInputGesture] = useState("");
  const [currentGesture, setCurrentGesture] = useState("Hello");
  const [isAnimating, setIsAnimating] = useState(true);
  const [neuralMode, setNeuralMode] = useState(true);

  const predefinedGestures = [
    "Hello",
    "Peace",
    "Thumbs Up",
    "Thank You",
    "Please",
    "Sorry",
    "Love",
    "Stop",
  ]; // [cite: 55]

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

  // Effect to get window dimensions for background animation
  // Ensure this runs client-side only where window is available
  const [windowSize, setWindowSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (typeof window !== "undefined") {
      const handleResize = () => {
        setWindowSize({ width: window.innerWidth, height: window.innerHeight });
      };
      window.addEventListener("resize", handleResize);
      handleResize(); // Initial size
      return () => window.removeEventListener("resize", handleResize);
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-cyan-950 relative overflow-hidden">
      {/* Animated neural background - conditional on neuralMode and windowSize */}
      {neuralMode &&
        windowSize.width > 0 && ( // [cite: 56]
          <div className="fixed inset-0 pointer-events-none">
            {Array.from({ length: 40 }).map((_, i) => (
              <motion.div
                key={i}
                className="absolute w-1 h-1 bg-cyan-400/20 rounded-full"
                animate={{
                  x: [0, Math.random() * windowSize.width], // [cite: 56]
                  y: [0, Math.random() * windowSize.height], // [cite: 56]
                  scale: [1, 1.5, 1], // [cite: 57]
                  opacity: [0.2, 0.8, 0.2], // [cite: 57]
                }}
                transition={{
                  duration: Math.random() * 15 + 10,
                  repeat: Number.POSITIVE_INFINITY,
                  ease: "linear", // [cite: 58]
                }}
                style={{
                  left: Math.random() * windowSize.width, // [cite: 58]
                  top: Math.random() * windowSize.height, // [cite: 58]
                }}
              />
            ))}
          </div>
        )}
      {/* Navigation */} {/* [cite: 59] */}
      <div className="pt-24 pb-12 px-6 relative z-10">
        <div className="container mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            {" "}
            {/* [cite: 65] */}
            <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-cyan-300 via-purple-300 to-pink-300 bg-clip-text text-transparent">
              {neuralMode
                ? "Neural 3D Visualization"
                : "3D Gesture Visualization"}
            </h1>
            <p className="text-cyan-100/70 text-xl max-w-3xl mx-auto">
              {neuralMode
                ? "Experience gesture recognition in immersive 3D space with real-time neural network visualization"
                : "Explore 3D hand gestures with detailed articulation and controls."}{" "}
              {/* [cite: 66] */}
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-4 gap-8">
            <motion.div
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }} // [cite: 67]
              transition={{ delay: 0.2 }} // [cite: 67]
              className="lg:col-span-1"
            >
              <Card className="bg-gradient-to-br from-slate-900/50 to-purple-900/30 border border-cyan-400/20 backdrop-blur-xl">
                <CardHeader>
                  <CardTitle className="flex items-center text-cyan-100">
                    {" "}
                    {/* [cite: 68] */}
                    <Sparkles className="mr-2 h-6 w-6 text-cyan-400" />{" "}
                    {/* [cite: 68] */}
                    {neuralMode ? "Neural Controls" : "Gesture Controls"}
                    {neuralMode /* Only show rotating Cpu if neuralMode is on */ && (
                      <motion.div // [cite: 69]
                        className="ml-auto" // [cite: 69]
                        animate={{ rotate: 360 }} // [cite: 69]
                        transition={{
                          duration: 6,
                          repeat: Number.POSITIVE_INFINITY,
                          ease: "linear",
                        }} // [cite: 69]
                      >
                        <Cpu className="h-5 w-5 text-purple-400" />{" "}
                        {/* [cite: 70] */}
                      </motion.div>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <Label
                      htmlFor="gesture-input"
                      className="text-cyan-100/90 mb-3 block"
                    >
                      {" "}
                      {/* [cite: 71] */}
                      {neuralMode
                        ? "Neural Gesture Input"
                        : "Gesture Input"}{" "}
                      {/* [cite: 71] */}
                    </Label>
                    <Input
                      id="gesture-input"
                      value={inputGesture} // [cite: 72]
                      onChange={(e) => setInputGesture(e.target.value)} // [cite: 72]
                      placeholder="e.g., Hello, Peace, Love..."
                      className="bg-slate-800/50 border-cyan-400/30 text-cyan-100 placeholder:text-cyan-400/50" // [cite: 72]
                    />{" "}
                    {/* [cite: 73] */}
                  </div>

                  <Button
                    onClick={handleGenerateGesture}
                    className="w-full bg-gradient-to-r from-cyan-600 to-purple-600 hover:from-cyan-700 hover:to-purple-700 text-white py-3"
                  >
                    {" "}
                    {/* [cite: 74] */}
                    <Play className="mr-2 h-4 w-4" /> {/* [cite: 74] */}
                    {neuralMode
                      ? "Generate Neural Gesture"
                      : "Generate Gesture"}
                  </Button>

                  <Button
                    onClick={handleRandomGesture} // [cite: 75]
                    variant="outline"
                    className="w-full border-cyan-400/30 text-cyan-300 hover:bg-cyan-400/10 py-3"
                  >
                    <RotateCcw className="mr-2 h-4 w-4" />
                    {neuralMode
                      ? "Random Neural Pattern"
                      : "Random Gesture"}{" "}
                    {/* [cite: 76] */}
                  </Button>

                  <div className="space-y-3">
                    <Label className="text-cyan-100/90">
                      {neuralMode ? "Quick Neural Patterns" : "Quick Gestures"}
                    </Label>
                    <div className="grid grid-cols-2 gap-2">
                      {predefinedGestures.map(
                        (
                          gesture // [cite: 77]
                        ) => (
                          <Button
                            key={gesture}
                            onClick={() => {
                              setCurrentGesture(gesture); // [cite: 78]
                              setInputGesture(gesture); // [cite: 78]
                            }}
                            variant="outline" // [cite: 79]
                            size="sm"
                            className="border-cyan-400/20 text-cyan-300 hover:bg-cyan-400/10 text-xs py-2"
                          >
                            {gesture} {/* [cite: 80] */}
                          </Button>
                        )
                      )}
                    </div>
                  </div>

                  <div className="space-y-4">
                    {" "}
                    {/* [cite: 81] */}
                    <Label className="text-cyan-100/90">
                      {neuralMode ? "Neural Settings" : "Display Settings"}
                    </Label>{" "}
                    {/* [cite: 81] */}
                    <div className="space-y-3">
                      <Button
                        onClick={() => setIsAnimating(!isAnimating)} // [cite: 82]
                        variant="outline"
                        size="sm" // [cite: 83]
                        className="w-full border-cyan-400/20 text-cyan-300 hover:bg-cyan-400/10" // [cite: 82]
                      >
                        {" "}
                        {/* [cite: 83] */}
                        <Activity className="mr-2 h-4 w-4" /> {/* [cite: 83] */}
                        {isAnimating ? "Pause" : "Resume"}{" "}
                        {neuralMode ? "Neural Activity" : "Animation"}{" "}
                        {/* [cite: 84] */}
                      </Button>

                      <Button
                        onClick={() => setNeuralMode(!neuralMode)}
                        variant="outline"
                        size="sm" // [cite: 85]
                        className="w-full border-purple-400/20 text-purple-300 hover:bg-purple-400/10" // [cite: 85]
                      >
                        <Zap className="mr-2 h-4 w-4" />
                        {neuralMode ? "Disable" : "Enable"} Neural Mode{" "}
                        {/* [cite: 86, 87] */}
                      </Button>
                    </div>
                  </div>

                  {neuralMode /* Neural activity indicator only in neural mode */ && (
                    <div className="bg-gradient-to-r from-cyan-500/10 to-purple-500/10 p-4 rounded-lg border border-cyan-400/20">
                      {" "}
                      {/* [cite: 88] */}
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-cyan-100/70 text-sm">
                          Neural Activity
                        </span>
                        <Network className="h-4 w-4 text-cyan-400" />{" "}
                        {/* [cite: 89] */}
                      </div>
                      <div className="flex space-x-1">
                        {" "}
                        {/* [cite: 89] */}
                        {Array.from({ length: 8 }).map((_, i) => (
                          <motion.div
                            key={i} // [cite: 90]
                            className="w-2 h-8 bg-gradient-to-t from-cyan-400 to-purple-400 rounded-full" // [cite: 90]
                            animate={
                              // [cite: 91]
                              isAnimating
                                ? {
                                    scaleY: [0.3, 1, 0.3], // [cite: 91]
                                    opacity: [0.5, 1, 0.5], // [cite: 91]
                                  }
                                : {} // [cite: 92]
                            }
                            transition={{
                              // [cite: 93]
                              duration: 1, // [cite: 93]
                              repeat: Number.POSITIVE_INFINITY, // [cite: 93]
                              delay: i * 0.1, // [cite: 94]
                            }}
                          />
                        ))}
                      </div>
                    </div> /* [cite: 95] */
                  )}
                </CardContent>
              </Card>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }} // [cite: 96]
              transition={{ delay: 0.4 }} // [cite: 96]
              className="lg:col-span-3"
            >
              <Card className="bg-gradient-to-br from-slate-900/50 to-purple-900/30 border border-cyan-400/20 backdrop-blur-xl">
                <CardHeader>
                  <CardTitle className="flex items-center justify-between text-cyan-100">
                    {" "}
                    {/* [cite: 97] */}
                    <span className="flex items-center">
                      <Hand className="mr-3 h-6 w-6 text-cyan-400" />
                      {neuralMode ? "Neural Gesture:" : "Gesture:"}
                      <span className="ml-2 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                        {currentGesture} {/* [cite: 98] */}
                      </span>
                    </span>
                    {neuralMode /* Only show neural active status if neuralMode is on */ && (
                      <div className="flex items-center space-x-2">
                        {" "}
                        {/* [cite: 99] */}
                        <motion.div
                          animate={{ scale: [1, 1.2, 1] }} // [cite: 99]
                          transition={{
                            duration: 2,
                            repeat: Number.POSITIVE_INFINITY,
                          }} // [cite: 99]
                        >
                          {" "}
                          {/* [cite: 100] */}
                          <Eye className="h-5 w-5 text-green-400" />{" "}
                          {/* [cite: 100] */}
                        </motion.div>
                        <span className="text-sm text-cyan-300/70 font-mono">
                          NEURAL ACTIVE
                        </span>{" "}
                        {/* [cite: 100] */}
                      </div> /* [cite: 101] */
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-[700px] w-full rounded-lg overflow-hidden relative">
                    <Canvas shadows>
                      {" "}
                      {/* [cite: 102] */}
                      <PerspectiveCamera makeDefault position={[0, 2, 10]} />
                      <ambientLight intensity={neuralMode ? 0.3 : 0.6} />{" "}
                      {/* Brighter ambient for non-neural */}
                      <pointLight
                        position={[10, 10, 10]}
                        intensity={neuralMode ? 1 : 0.8}
                        color={neuralMode ? "#00ffaa" : "#ffffff"}
                        castShadow
                      />{" "}
                      {/* [cite: 103] */}
                      <pointLight
                        position={[-10, -10, -10]}
                        intensity={neuralMode ? 0.8 : 0.4}
                        color={neuralMode ? "#00aaff" : "#ffffff"}
                      />{" "}
                      {/* [cite: 103] */}
                      <pointLight
                        position={[0, 15, 0]}
                        intensity={neuralMode ? 0.6 : 0.3}
                        color={neuralMode ? "#ffaa00" : "#ffffff"}
                      />{" "}
                      {/* [cite: 103] */}
                      <Suspense
                        fallback={
                          /* [cite: 104] */
                          <Html center>
                            <div className="text-cyan-400 text-xl font-mono">
                              <motion.div
                                animate={{ rotate: 360 }} // [cite: 105]
                                transition={{
                                  duration: 2,
                                  repeat: Number.POSITIVE_INFINITY,
                                  ease: "linear",
                                }} // [cite: 105]
                                className="inline-block mr-2" // [cite: 106]
                              >
                                <Network className="h-6 w-6" />{" "}
                                {/* [cite: 106] */}
                              </motion.div>
                              {neuralMode
                                ? "Loading Neural Space..."
                                : "Loading 3D Viewer..."}{" "}
                              {/* [cite: 107] */}
                            </div>
                          </Html>
                        } /* [cite: 108] */
                      >
                        {neuralMode && <NeuralGrid />} {/* [cite: 108] */}
                        <RealisticHandModel
                          gesture={currentGesture}
                          isAnimating={isAnimating}
                          neuralMode={neuralMode}
                        />
                        {neuralMode && <NeuralParticles />} {/* [cite: 109] */}
                        <Text3D
                          font="/fonts/Geist_Bold.json"
                          size={0.6} // [cite: 110]
                          height={0.15} // [cite: 110]
                          position={[0, -3, 0]} // Adjusted Y position
                          rotation={[0, 0, 0]}
                        >
                          {" "}
                          {/* [cite: 111] */}
                          {currentGesture}
                          <meshStandardMaterial
                            color="#00ffaa"
                            emissive={neuralMode ? "#00ffaa" : "#333333"}
                            emissiveIntensity={neuralMode ? 0.3 : 0.1}
                          />{" "}
                          {/* [cite: 111] */}
                        </Text3D>
                        {neuralMode /* Neural subtitle only in neuralMode */ && ( // [cite: 112]
                          <Text3D
                            font="/fonts/Geist_Regular.json" // [cite: 112]
                            size={0.2} // [cite: 113]
                            height={0.05} // [cite: 113]
                            position={[0, -3.8, 0]} // Adjusted Y position [cite: 113]
                            rotation={[0, 0, 0]}
                          >
                            {" "}
                            {/* [cite: 114] */}
                            NEURAL PATTERN ACTIVE
                            <meshStandardMaterial
                              color="#ff00aa"
                              emissive="#ff00aa"
                              emissiveIntensity={0.2}
                            />{" "}
                            {/* [cite: 114] */}
                          </Text3D>
                        )}
                        <ContactShadows
                          position={[0, -2.5, 0]}
                          opacity={0.4}
                          scale={10}
                          blur={2.5}
                          far={4}
                        />{" "}
                        {/* Adjusted Y position slightly */} {/* [cite: 115] */}
                        <Environment
                          preset={neuralMode ? "night" : "city"}
                        />{" "}
                        {/* Different preset for non-neural for better lighting */}{" "}
                        {/* [cite: 115] */}
                      </Suspense>
                      <OrbitControls
                        enablePan={true} // [cite: 116]
                        enableZoom={true} // [cite: 116]
                        enableRotate={true} // [cite: 116]
                        minDistance={4} // [cite: 116]
                        maxDistance={20} // [cite: 117]
                        autoRotate={isAnimating} // [cite: 117]
                        autoRotateSpeed={0.5} // [cite: 117]
                      />
                    </Canvas>{" "}
                    {/* [cite: 118] */}
                    <div className="absolute top-4 right-4 bg-gradient-to-r from-slate-900/80 to-purple-900/80 backdrop-blur-xl rounded-lg p-4 border border-cyan-400/30">
                      <div className="text-cyan-100 text-sm">
                        <div className="flex items-center mb-2">
                          {" "}
                          {/* [cite: 119] */}
                          <Activity className="h-4 w-4 text-cyan-400 mr-2" />{" "}
                          {/* [cite: 119] */}
                          <span>
                            {neuralMode ? "Neural Status" : "Display Status"}
                          </span>{" "}
                          {/* [cite: 119] */}
                        </div>
                        <div className="space-y-1 text-xs">
                          {" "}
                          {/* [cite: 120] */}
                          <div className="flex justify-between">
                            <span className="text-cyan-300/70">Mode:</span>
                            <span className="text-cyan-300">
                              {neuralMode ? "Enhanced" : "Standard"}
                            </span>{" "}
                            {/* [cite: 121] */}
                          </div>
                          <div className="flex justify-between">
                            <span className="text-cyan-300/70">Animation:</span>
                            <span className="text-cyan-300">
                              {isAnimating ? "Active" : "Paused"}
                            </span>{" "}
                            {/* [cite: 122, 123] */}
                          </div>
                          <div className="flex justify-between">
                            <span className="text-cyan-300/70">Gesture:</span>
                            <span className="text-cyan-300">
                              {currentGesture}
                            </span>{" "}
                            {/* [cite: 124] */}
                          </div>
                        </div>
                      </div>
                    </div>{" "}
                    {/* [cite: 125] */}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 30 }} // [cite: 126]
            animate={{ opacity: 1, y: 0 }} // [cite: 126]
            transition={{ delay: 0.6 }} // [cite: 126]
            className="mt-8"
          >
            <Card className="bg-gradient-to-br from-slate-900/50 to-purple-900/30 border border-cyan-400/20 backdrop-blur-xl">
              <CardContent className="p-6">
                {" "}
                {/* [cite: 127] */}
                <h3 className="text-cyan-100 font-semibold mb-4 flex items-center">
                  <Settings className="mr-2 h-5 w-5 text-cyan-400" />
                  {neuralMode
                    ? "Neural 3D Controls & Features"
                    : "3D Viewer Controls & Features"}
                </h3>
                <div className="grid md:grid-cols-4 gap-6 text-sm text-cyan-100/70">
                  {" "}
                  {/* [cite: 128] */}
                  <div>
                    <strong className="text-cyan-300">Rotate:</strong> Click and
                    drag to rotate the neural space {/* [cite: 128] */}
                  </div>
                  <div>
                    <strong className="text-cyan-300">Zoom:</strong> Mouse wheel
                    or pinch to zoom in/out {/* [cite: 129] */}
                  </div>
                  <div>
                    <strong className="text-cyan-300">Pan:</strong> Right-click
                    and drag to pan view
                  </div>
                  <div>
                    <strong className="text-cyan-300">Auto-Rotate:</strong>{" "}
                    Enabled when {neuralMode ? "neural " : ""}animation is
                    active {/* [cite: 130] */}
                  </div>
                </div>
                <div className="mt-4 p-4 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-lg border border-cyan-400/20">
                  {" "}
                  {/* [cite: 131] */}
                  <p className="text-cyan-100/80 text-sm">
                    <strong className="text-cyan-300">
                      Enhanced Hand Model:
                    </strong>{" "}
                    The {neuralMode ? "new anatomically correct" : ""} hand
                    model features realistic finger articulation with proper
                    joint movement
                    {neuralMode
                      ? ", enhanced materials with metallic and emissive properties,"
                      : "."}{" "}
                    {/* [cite: 131] */}
                    The gesture-specific animations provide an immersive
                    experience. {/* [cite: 132, 133] */}
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
