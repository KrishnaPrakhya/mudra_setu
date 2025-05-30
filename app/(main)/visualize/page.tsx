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
}: {
  gesture: string;
  isAnimating: boolean;
}) {
  const handRef = useRef<THREE.Group>(null);
  const [neuralPulse, setNeuralPulse] = useState(0);
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

  // Neural glow effect
  const [glowIntensity, setGlowIntensity] = useState(0.2);
  const [glowColor, setGlowColor] = useState(new THREE.Color("#00ffaa"));

  // Apply gesture-specific finger positions
  useEffect(() => {
    if (!handRef.current) return;

    // Reset all finger positions
    resetFingerPositions();

    // Apply specific gesture
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
      case "Love":
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
  }, [gesture]);

  // Reset all finger positions to default
  const resetFingerPositions = () => {
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // Reset base joints
    thumbRef.current.rotation.set(0, 0, 0);
    indexRef.current.rotation.set(0, 0, 0);
    middleRef.current.rotation.set(0, 0, 0);
    ringRef.current.rotation.set(0, 0, 0);
    pinkyRef.current.rotation.set(0, 0, 0);

    // Reset knuckles
    if (thumbKnuckleRef.current) thumbKnuckleRef.current.rotation.set(0, 0, 0);
    if (indexKnuckleRef.current) indexKnuckleRef.current.rotation.set(0, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(0, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(0, 0, 0);
    if (pinkyKnuckleRef.current) pinkyKnuckleRef.current.rotation.set(0, 0, 0);

    // Reset tips
    if (thumbTipRef.current) thumbTipRef.current.rotation.set(0, 0, 0);
    if (indexTipRef.current) indexTipRef.current.rotation.set(0, 0, 0);
    if (middleTipRef.current) middleTipRef.current.rotation.set(0, 0, 0);
    if (ringTipRef.current) ringTipRef.current.rotation.set(0, 0, 0);
    if (pinkyTipRef.current) pinkyTipRef.current.rotation.set(0, 0, 0);
  };

  // Apply "Hello" gesture - all fingers extended and slight waving motion
  const applyHelloGesture = () => {
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // Extend all fingers
    thumbRef.current.rotation.set(0, 0, -0.3);
    indexRef.current.rotation.set(0, 0, -0.1);
    middleRef.current.rotation.set(0, 0, 0);
    ringRef.current.rotation.set(0, 0, 0.1);
    pinkyRef.current.rotation.set(0, 0, 0.2);

    // Slight bend at knuckles for natural look
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
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // Curl thumb, ring, and pinky
    thumbRef.current.rotation.set(0.3, 0.5, 0.8);
    indexRef.current.rotation.set(0, 0, -0.1);
    middleRef.current.rotation.set(0, 0, 0);
    ringRef.current.rotation.set(0.7, 0, 0.2);
    pinkyRef.current.rotation.set(0.8, 0, 0.3);

    // Bend knuckles appropriately
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0.5, 0, 0);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(0.8, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(0.8, 0, 0);

    // Bend tips
    if (thumbTipRef.current) thumbTipRef.current.rotation.set(0.5, 0, 0);
    if (indexTipRef.current) indexTipRef.current.rotation.set(0, 0, 0);
    if (middleTipRef.current) middleTipRef.current.rotation.set(0, 0, 0);
    if (ringTipRef.current) ringTipRef.current.rotation.set(0.8, 0, 0);
    if (pinkyTipRef.current) pinkyTipRef.current.rotation.set(0.8, 0, 0);
  };

  // Apply "Thumbs Up" gesture
  const applyThumbsUpGesture = () => {
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // Extend thumb upward, curl other fingers
    thumbRef.current.rotation.set(-1.2, 0, -0.3);
    indexRef.current.rotation.set(0.7, 0, 0.1);
    middleRef.current.rotation.set(0.8, 0, 0);
    ringRef.current.rotation.set(0.8, 0, -0.1);
    pinkyRef.current.rotation.set(0.8, 0, -0.2);

    // Bend knuckles
    if (thumbKnuckleRef.current) thumbKnuckleRef.current.rotation.set(0, 0, 0);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(0.8, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(0.8, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(0.8, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(0.8, 0, 0);

    // Bend tips
    if (thumbTipRef.current) thumbTipRef.current.rotation.set(0, 0, 0);
    if (indexTipRef.current) indexTipRef.current.rotation.set(0.5, 0, 0);
    if (middleTipRef.current) middleTipRef.current.rotation.set(0.5, 0, 0);
    if (ringTipRef.current) ringTipRef.current.rotation.set(0.5, 0, 0);
    if (pinkyTipRef.current) pinkyTipRef.current.rotation.set(0.5, 0, 0);
  };

  // Apply "Thank You" gesture - palm open, slight bow
  const applyThankYouGesture = () => {
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // Open palm with slight finger spread
    thumbRef.current.rotation.set(0, 0, -0.4);
    indexRef.current.rotation.set(0, 0, -0.2);
    middleRef.current.rotation.set(0, 0, 0);
    ringRef.current.rotation.set(0, 0, 0.2);
    pinkyRef.current.rotation.set(0, 0, 0.4);

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

  // Apply "Please" gesture - hands together in prayer position
  const applyPleaseGesture = () => {
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // Fingers straight and together
    thumbRef.current.rotation.set(0, 0, -0.1);
    indexRef.current.rotation.set(0, 0, 0);
    middleRef.current.rotation.set(0, 0, 0);
    ringRef.current.rotation.set(0, 0, 0);
    pinkyRef.current.rotation.set(0, 0, 0.1);

    // Slight bend at knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(-0.1, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(-0.1, 0, 0);
  };

  // Apply "Sorry" gesture - hand over heart
  const applySorryGesture = () => {
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // Fingers slightly curled
    thumbRef.current.rotation.set(0.2, 0, -0.3);
    indexRef.current.rotation.set(0.3, 0, -0.1);
    middleRef.current.rotation.set(0.3, 0, 0);
    ringRef.current.rotation.set(0.3, 0, 0.1);
    pinkyRef.current.rotation.set(0.3, 0, 0.2);

    // Bend knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0.2, 0, 0);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(0.3, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(0.3, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(0.3, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(0.3, 0, 0);
  };

  // Apply "Love" gesture - heart shape with hands
  const applyLoveGesture = () => {
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // Thumb and index extended to form half heart
    thumbRef.current.rotation.set(-0.5, 0.5, -0.5);
    indexRef.current.rotation.set(-0.5, 0, -0.3);
    middleRef.current.rotation.set(0.7, 0, 0);
    ringRef.current.rotation.set(0.8, 0, 0.1);
    pinkyRef.current.rotation.set(0.8, 0, 0.2);

    // Bend knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0, 0.3, 0);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(-0.3, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(0.7, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(0.7, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(0.7, 0, 0);
  };

  // Apply "Stop" gesture - palm forward, fingers extended
  const applyStopGesture = () => {
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // All fingers extended straight
    thumbRef.current.rotation.set(0, 0, -0.4);
    indexRef.current.rotation.set(0, 0, -0.1);
    middleRef.current.rotation.set(0, 0, 0);
    ringRef.current.rotation.set(0, 0, 0.1);
    pinkyRef.current.rotation.set(0, 0, 0.2);

    // No bend at knuckles - straight and firm
    if (thumbKnuckleRef.current) thumbKnuckleRef.current.rotation.set(0, 0, 0);
    if (indexKnuckleRef.current) indexKnuckleRef.current.rotation.set(0, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(0, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(0, 0, 0);
    if (pinkyKnuckleRef.current) pinkyKnuckleRef.current.rotation.set(0, 0, 0);
  };

  // Apply default gesture - relaxed open hand
  const applyDefaultGesture = () => {
    if (
      !thumbRef.current ||
      !indexRef.current ||
      !middleRef.current ||
      !ringRef.current ||
      !pinkyRef.current
    )
      return;

    // Relaxed finger positions
    thumbRef.current.rotation.set(0, 0, -0.3);
    indexRef.current.rotation.set(0.1, 0, -0.1);
    middleRef.current.rotation.set(0.1, 0, 0);
    ringRef.current.rotation.set(0.1, 0, 0.1);
    pinkyRef.current.rotation.set(0.1, 0, 0.2);

    // Natural curve at knuckles
    if (thumbKnuckleRef.current)
      thumbKnuckleRef.current.rotation.set(0.1, 0, 0);
    if (indexKnuckleRef.current)
      indexKnuckleRef.current.rotation.set(0.2, 0, 0);
    if (middleKnuckleRef.current)
      middleKnuckleRef.current.rotation.set(0.2, 0, 0);
    if (ringKnuckleRef.current) ringKnuckleRef.current.rotation.set(0.2, 0, 0);
    if (pinkyKnuckleRef.current)
      pinkyKnuckleRef.current.rotation.set(0.2, 0, 0);

    // Slight bend at tips
    if (thumbTipRef.current) thumbTipRef.current.rotation.set(0.1, 0, 0);
    if (indexTipRef.current) indexTipRef.current.rotation.set(0.1, 0, 0);
    if (middleTipRef.current) middleTipRef.current.rotation.set(0.1, 0, 0);
    if (ringTipRef.current) ringTipRef.current.rotation.set(0.1, 0, 0);
    if (pinkyTipRef.current) pinkyTipRef.current.rotation.set(0.1, 0, 0);
  };

  useFrame(() => {
    if (handRef.current && isAnimating) {
      // Subtle hand movement
      handRef.current.rotation.y = Math.sin(clock.elapsedTime * 0.5) * 0.2;
      handRef.current.position.y = Math.sin(clock.elapsedTime * 0.7) * 0.1;

      // Neural pulse effect
      const pulse = Math.sin(clock.elapsedTime * 2) * 0.5 + 0.5;
      setNeuralPulse(pulse);
      setGlowIntensity(0.2 + pulse * 0.3);

      // Subtle finger movement for more lifelike appearance
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
    }
  });

  // Create enhanced finger segment with rounded edges
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
        emissiveIntensity={glowIntensity * 0.3}
        roughness={0.3}
        metalness={0.7}
        transparent
        opacity={0.9}
      />
    </mesh>
  );

  return (
    <group
      ref={handRef}
      position={[0, 0, 0]}
      rotation={[0.2, 0, 0]}
      scale={1.5}
    >
      {/* Enhanced Palm with anatomical shape */}
      <mesh castShadow receiveShadow>
        <boxGeometry args={[1.2, 1.8, 0.4]} />
        <meshStandardMaterial
          color={glowColor}
          emissive={glowColor}
          emissiveIntensity={glowIntensity * 0.2}
          roughness={0.4}
          metalness={0.6}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Wrist with natural taper */}
      <mesh position={[0, -1.2, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.4, 0.5, 0.6, 16]} />
        <meshStandardMaterial
          color={glowColor}
          emissive={glowColor}
          emissiveIntensity={glowIntensity * 0.15}
          roughness={0.4}
          metalness={0.6}
          transparent
          opacity={0.85}
        />
      </mesh>

      {/* Thumb with realistic proportions */}
      <group
        ref={thumbRef}
        position={[-0.7, -0.2, 0.1]}
        rotation={[0, 0, -0.3]}
      >
        <FingerSegment width={0.25} height={0.6} depth={0.25} />

        {/* Thumb knuckle joint */}
        <group
          ref={thumbKnuckleRef}
          position={[0, 0.6, 0]}
          rotation={[0, 0, -0.1]}
        >
          <FingerSegment width={0.22} height={0.4} depth={0.22} isJoint />

          {/* Thumb tip */}
          <group ref={thumbTipRef} position={[0, 0.4, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.12, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={glowIntensity}
                roughness={0.2}
                metalness={0.8}
              />
            </mesh>
          </group>
        </group>
      </group>

      {/* Index finger with enhanced detail */}
      <group ref={indexRef} position={[-0.4, 0.9, 0.1]} rotation={[0, 0, -0.1]}>
        <FingerSegment width={0.22} height={0.7} depth={0.22} />

        {/* Index knuckle */}
        <group
          ref={indexKnuckleRef}
          position={[0, 0.7, 0]}
          rotation={[-0.1, 0, 0]}
        >
          <FingerSegment width={0.2} height={0.5} depth={0.2} />

          {/* Index tip */}
          <group ref={indexTipRef} position={[0, 0.5, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.11, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={glowIntensity}
                roughness={0.2}
                metalness={0.8}
              />
            </mesh>
          </group>
        </group>
      </group>

      {/* Middle finger - longest finger */}
      <group ref={middleRef} position={[0, 0.9, 0.1]} rotation={[0, 0, 0]}>
        <FingerSegment width={0.22} height={0.8} depth={0.22} />

        {/* Middle knuckle */}
        <group
          ref={middleKnuckleRef}
          position={[0, 0.8, 0]}
          rotation={[-0.1, 0, 0]}
        >
          <FingerSegment width={0.2} height={0.5} depth={0.2} />

          {/* Middle tip */}
          <group ref={middleTipRef} position={[0, 0.5, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.11, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={glowIntensity}
                roughness={0.2}
                metalness={0.8}
              />
            </mesh>
          </group>
        </group>
      </group>

      {/* Ring finger */}
      <group ref={ringRef} position={[0.4, 0.9, 0.1]} rotation={[0, 0, 0.1]}>
        <FingerSegment width={0.22} height={0.7} depth={0.22} />

        {/* Ring knuckle */}
        <group
          ref={ringKnuckleRef}
          position={[0, 0.7, 0]}
          rotation={[-0.1, 0, 0]}
        >
          <FingerSegment width={0.2} height={0.5} depth={0.2} />

          {/* Ring tip */}
          <group ref={ringTipRef} position={[0, 0.5, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.11, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={glowIntensity}
                roughness={0.2}
                metalness={0.8}
              />
            </mesh>
          </group>
        </group>
      </group>

      {/* Pinky finger - smallest */}
      <group ref={pinkyRef} position={[0.7, 0.8, 0.1]} rotation={[0, 0, 0.2]}>
        <FingerSegment width={0.2} height={0.6} depth={0.2} />

        {/* Pinky knuckle */}
        <group
          ref={pinkyKnuckleRef}
          position={[0, 0.6, 0]}
          rotation={[-0.1, 0, 0]}
        >
          <FingerSegment width={0.18} height={0.4} depth={0.18} />

          {/* Pinky tip */}
          <group ref={pinkyTipRef} position={[0, 0.4, 0]} rotation={[0, 0, 0]}>
            <mesh castShadow receiveShadow>
              <sphereGeometry args={[0.1, 16, 16]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={glowIntensity}
                roughness={0.2}
                metalness={0.8}
              />
            </mesh>
          </group>
        </group>
      </group>

      {/* Neural connection lines */}
      {isAnimating && (
        <group>
          {Array.from({ length: 12 }).map((_, i) => (
            <mesh
              key={i}
              position={[0, 0, 0.2]}
              rotation={[0, 0, (i * Math.PI) / 6]}
            >
              <cylinderGeometry args={[0.01, 0.01, 2 + neuralPulse * 0.5]} />
              <meshStandardMaterial
                color={glowColor}
                emissive={glowColor}
                emissiveIntensity={glowIntensity}
                transparent
                opacity={0.6}
              />
            </mesh>
          ))}
        </group>
      )}

      {/* Neural energy core */}
      <mesh position={[0, 0, 0]} castShadow>
        <sphereGeometry args={[0.3, 16, 16]} />
        <meshStandardMaterial
          color={glowColor}
          emissive={glowColor}
          emissiveIntensity={glowIntensity * 1.5}
          transparent
          opacity={0.7}
          roughness={0.1}
          metalness={0.9}
        />
      </mesh>

      {/* Anatomical details - knuckle bumps */}
      {(
        [
          [-0.4, 0.4, 0.15],
          [0, 0.4, 0.15],
          [0.4, 0.4, 0.15],
          [0.7, 0.3, 0.15],
        ] as [number, number, number][]
      ).map((pos, i) => (
        <mesh key={i} position={pos} castShadow>
          <sphereGeometry args={[0.08, 8, 8]} />
          <meshStandardMaterial
            color={glowColor}
            emissive={glowColor}
            emissiveIntensity={glowIntensity * 0.5}
            transparent
            opacity={0.6}
            roughness={0.5}
            metalness={0.5}
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
      particlesRef.current.rotation.y = state.clock.elapsedTime * 0.05;
      particlesRef.current.rotation.x =
        Math.sin(state.clock.elapsedTime * 0.1) * 0.1;
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
            }
            emissive={
              i % 3 === 0 ? "#00ffaa" : i % 3 === 1 ? "#00aaff" : "#ffaa00"
            }
            emissiveIntensity={0.4 + Math.random() * 0.3}
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
      {/* Animated neural background */}
      <div className="fixed inset-0 pointer-events-none">
        {Array.from({ length: 40 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400/20 rounded-full"
            animate={{
              x: [0, Math.random() * window.innerWidth],
              y: [0, Math.random() * window.innerHeight],
              scale: [1, 1.5, 1],
              opacity: [0.2, 0.8, 0.2],
            }}
            transition={{
              duration: Math.random() * 15 + 10,
              repeat: Number.POSITIVE_INFINITY,
              ease: "linear",
            }}
            style={{
              left: Math.random() * window.innerWidth,
              top: Math.random() * window.innerHeight,
            }}
          />
        ))}
      </div>

      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 bg-black/10 backdrop-blur-xl border-b border-cyan-400/20">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center space-x-3">
              <div className="relative">
                <Hand className="h-8 w-8 text-cyan-400" />
                <motion.div
                  className="absolute inset-0 h-8 w-8 text-cyan-400"
                  animate={{ rotate: 360 }}
                  transition={{
                    duration: 8,
                    repeat: Number.POSITIVE_INFINITY,
                    ease: "linear",
                  }}
                >
                  <Network className="h-8 w-8" />
                </motion.div>
              </div>
              <div>
                <span className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  NeuroGesture
                </span>
                <div className="text-xs text-cyan-400/70 font-mono">
                  3D NEURAL SPACE
                </div>
              </div>
            </Link>

            <div className="flex items-center space-x-6">
              <Link
                href="/predict"
                className="text-cyan-300/80 hover:text-cyan-300 transition-colors"
              >
                Neural Predict
              </Link>
              <Link
                href="/dashboard"
                className="text-cyan-300/80 hover:text-cyan-300 transition-colors"
              >
                Analytics
              </Link>
              <Link
                href="/"
                className="text-cyan-300/80 hover:text-cyan-300 transition-colors"
              >
                <ArrowLeft className="h-5 w-5" />
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <div className="pt-24 pb-12 px-6 relative z-10">
        <div className="container mx-auto max-w-7xl">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-cyan-300 via-purple-300 to-pink-300 bg-clip-text text-transparent">
              Neural 3D Visualization
            </h1>
            <p className="text-cyan-100/70 text-xl max-w-3xl mx-auto">
              Experience gesture recognition in immersive 3D space with
              real-time neural network visualization
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-4 gap-8">
            {/* Enhanced Controls */}
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
                    Neural Controls
                    <motion.div
                      className="ml-auto"
                      animate={{ rotate: 360 }}
                      transition={{
                        duration: 6,
                        repeat: Number.POSITIVE_INFINITY,
                        ease: "linear",
                      }}
                    >
                      <Cpu className="h-5 w-5 text-purple-400" />
                    </motion.div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div>
                    <Label
                      htmlFor="gesture-input"
                      className="text-cyan-100/90 mb-3 block"
                    >
                      Neural Gesture Input
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
                    <Play className="mr-2 h-4 w-4" />
                    Generate Neural Gesture
                  </Button>

                  <Button
                    onClick={handleRandomGesture}
                    variant="outline"
                    className="w-full border-cyan-400/30 text-cyan-300 hover:bg-cyan-400/10 py-3"
                  >
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Random Neural Pattern
                  </Button>

                  <div className="space-y-3">
                    <Label className="text-cyan-100/90">
                      Quick Neural Patterns
                    </Label>
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
                    <Label className="text-cyan-100/90">Neural Settings</Label>
                    <div className="space-y-3">
                      <Button
                        onClick={() => setIsAnimating(!isAnimating)}
                        variant="outline"
                        size="sm"
                        className="w-full border-cyan-400/20 text-cyan-300 hover:bg-cyan-400/10"
                      >
                        <Activity className="mr-2 h-4 w-4" />
                        {isAnimating ? "Pause" : "Resume"} Neural Activity
                      </Button>

                      <Button
                        onClick={() => setNeuralMode(!neuralMode)}
                        variant="outline"
                        size="sm"
                        className="w-full border-purple-400/20 text-purple-300 hover:bg-purple-400/10"
                      >
                        <Zap className="mr-2 h-4 w-4" />
                        {neuralMode ? "Disable" : "Enable"} Neural Mode
                      </Button>
                    </div>
                  </div>

                  {/* Neural activity indicator */}
                  <div className="bg-gradient-to-r from-cyan-500/10 to-purple-500/10 p-4 rounded-lg border border-cyan-400/20">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-cyan-100/70 text-sm">
                        Neural Activity
                      </span>
                      <Network className="h-4 w-4 text-cyan-400" />
                    </div>
                    <div className="flex space-x-1">
                      {Array.from({ length: 8 }).map((_, i) => (
                        <motion.div
                          key={i}
                          className="w-2 h-8 bg-gradient-to-t from-cyan-400 to-purple-400 rounded-full"
                          animate={
                            isAnimating
                              ? {
                                  scaleY: [0.3, 1, 0.3],
                                  opacity: [0.5, 1, 0.5],
                                }
                              : {}
                          }
                          transition={{
                            duration: 1,
                            repeat: Number.POSITIVE_INFINITY,
                            delay: i * 0.1,
                          }}
                        />
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Enhanced 3D Visualization */}
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
                      Neural Gesture:
                      <span className="ml-2 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                        {currentGesture}
                      </span>
                    </span>
                    <div className="flex items-center space-x-2">
                      <motion.div
                        animate={{ scale: [1, 1.2, 1] }}
                        transition={{
                          duration: 2,
                          repeat: Number.POSITIVE_INFINITY,
                        }}
                      >
                        <Eye className="h-5 w-5 text-green-400" />
                      </motion.div>
                      <span className="text-sm text-cyan-300/70 font-mono">
                        NEURAL ACTIVE
                      </span>
                    </div>
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-[700px] w-full rounded-lg overflow-hidden relative">
                    <Canvas shadows>
                      <PerspectiveCamera makeDefault position={[0, 2, 10]} />

                      {/* Enhanced lighting */}
                      <ambientLight intensity={0.3} />
                      <pointLight
                        position={[10, 10, 10]}
                        intensity={1}
                        color="#00ffaa"
                        castShadow
                      />
                      <pointLight
                        position={[-10, -10, -10]}
                        intensity={0.8}
                        color="#00aaff"
                      />
                      <pointLight
                        position={[0, 15, 0]}
                        intensity={0.6}
                        color="#ffaa00"
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
                                <Network className="h-6 w-6" />
                              </motion.div>
                              Loading Neural Space...
                            </div>
                          </Html>
                        }
                      >
                        {neuralMode && <NeuralGrid />}
                        <RealisticHandModel
                          gesture={currentGesture}
                          isAnimating={isAnimating}
                        />
                        <NeuralParticles />

                        {/* Enhanced 3D Text */}
                        <Text3D
                          font="/fonts/Geist_Bold.json"
                          size={0.6}
                          height={0.15}
                          position={[0, -5, 0]}
                          rotation={[0, 0, 0]}
                        >
                          {currentGesture}
                          <meshStandardMaterial
                            color="#00ffaa"
                            emissive="#00ffaa"
                            emissiveIntensity={0.3}
                          />
                        </Text3D>

                        {/* Neural subtitle */}
                        <Text3D
                          font="/fonts/Geist_Regular.json"
                          size={0.2}
                          height={0.05}
                          position={[0, -6, 0]}
                          rotation={[0, 0, 0]}
                        >
                          NEURAL PATTERN ACTIVE
                          <meshStandardMaterial
                            color="#ff00aa"
                            emissive="#ff00aa"
                            emissiveIntensity={0.2}
                          />
                        </Text3D>

                        <ContactShadows
                          position={[0, -2, 0]}
                          opacity={0.4}
                          scale={10}
                          blur={2.5}
                          far={4}
                        />
                        <Environment preset="night" />
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

                    {/* Neural overlay UI */}
                    <div className="absolute top-4 right-4 bg-gradient-to-r from-slate-900/80 to-purple-900/80 backdrop-blur-xl rounded-lg p-4 border border-cyan-400/30">
                      <div className="text-cyan-100 text-sm">
                        <div className="flex items-center mb-2">
                          <Activity className="h-4 w-4 text-cyan-400 mr-2" />
                          <span>Neural Status</span>
                        </div>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-cyan-300/70">Mode:</span>
                            <span className="text-cyan-300">
                              {neuralMode ? "Enhanced" : "Standard"}
                            </span>
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

          {/* Enhanced Instructions */}
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
                  Neural 3D Controls & Features
                </h3>
                <div className="grid md:grid-cols-4 gap-6 text-sm text-cyan-100/70">
                  <div>
                    <strong className="text-cyan-300">Rotate:</strong> Click and
                    drag to rotate the neural space
                  </div>
                  <div>
                    <strong className="text-cyan-300">Zoom:</strong> Mouse wheel
                    or pinch to zoom in/out
                  </div>
                  <div>
                    <strong className="text-cyan-300">Pan:</strong> Right-click
                    and drag to pan view
                  </div>
                  <div>
                    <strong className="text-cyan-300">Auto-Rotate:</strong>{" "}
                    Enabled when neural animation is active
                  </div>
                </div>
                <div className="mt-4 p-4 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-lg border border-cyan-400/20">
                  <p className="text-cyan-100/80 text-sm">
                    <strong className="text-cyan-300">
                      Enhanced Hand Model:
                    </strong>{" "}
                    The new anatomically correct hand model features realistic
                    finger articulation with proper joint movement, enhanced
                    materials with metallic and emissive properties, and
                    gesture-specific animations for a more immersive experience.
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
