"use client";

import { motion, useScroll, useTransform } from "framer-motion";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Brain,
  Hand,
  ArrowRight,
  Upload,
  Sparkles,
  Zap,
  Camera,
  ImageIcon,
  Cpu,
  Activity,
  Target,
} from "lucide-react";
import { useRef, useEffect, useState } from "react";

// Animated background component
function AnimatedBackground() {
  const [nodePositions, setNodePositions] = useState<
    { x: number; y: number }[]
  >([]);
  const [shapePositions, setShapePositions] = useState<
    { x: number; y: number }[]
  >([]);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    if (typeof window !== "undefined") {
      setNodePositions(
        Array.from({ length: 20 }).map(() => ({
          x: Math.random() * window.innerWidth,
          y: Math.random() * window.innerHeight,
        }))
      );
      setShapePositions(
        Array.from({ length: 8 }).map(() => ({
          x: Math.random() * window.innerWidth,
          y: Math.random() * window.innerHeight,
        }))
      );
    }
  }, []);

  if (!isClient || nodePositions.length === 0 || shapePositions.length === 0)
    return null;

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      {/* Neural network nodes */}

      {nodePositions.map((pos, i) => (
        <motion.div
          key={i}
          className="absolute w-2 h-2 bg-cyan-400/30 rounded-full"
          initial={{ x: pos.x, y: pos.y }}
          animate={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
          }}
          transition={{
            duration: Math.random() * 20 + 10,
            repeat: Number.POSITIVE_INFINITY,
            ease: "linear",
          }}
        />
      ))}

      {/* Floating geometric shapes */}
      {shapePositions.map((pos, i) => (
        <motion.div
          key={`shape-${i}`}
          className="absolute w-16 h-16 border border-purple-400/20 rounded-lg"
          initial={{ x: pos.x, y: pos.y, rotate: 0 }}
          animate={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            rotate: 360,
          }}
          transition={{
            duration: Math.random() * 30 + 20,
            repeat: Number.POSITIVE_INFINITY,
            ease: "linear",
          }}
        />
      ))}
    </div>
  );
}

// Glowing orb component
function GlowingOrb({ delay = 0 }: { delay?: number }) {
  return (
    <motion.div
      className="absolute w-32 h-32 rounded-full bg-gradient-to-r from-cyan-400/20 to-purple-400/20 blur-xl"
      animate={{
        scale: [1, 1.5, 1],
        opacity: [0.3, 0.6, 0.3],
      }}
      transition={{
        duration: 4,
        repeat: Number.POSITIVE_INFINITY,
        delay,
      }}
    />
  );
}

export default function HomePage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end start"],
  });

  const y = useTransform(scrollYProgress, [0, 1], ["0%", "50%"]);
  const opacity = useTransform(scrollYProgress, [0, 0.5], [1, 0]);

  return (
    <div
      ref={containerRef}
      className="min-h-screen bg-gradient-to-br from-slate-950 via-purple-950 to-cyan-950 relative overflow-hidden"
    >
      <AnimatedBackground />

      {/* Hero Section */}
      <motion.section
        style={{ y, opacity }}
        className="pt-32 pb-20 px-6 relative"
      >
        <div className="container mx-auto text-center relative">
          {/* Glowing orbs */}
          <GlowingOrb delay={0} />
          <GlowingOrb delay={2} />

          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 1 }}
            className="max-w-5xl mx-auto relative z-10"
          >
            <motion.div
              className="mb-8"
              animate={{
                backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
              }}
              transition={{ duration: 5, repeat: Number.POSITIVE_INFINITY }}
            >
              <h1 className="text-7xl md:text-8xl font-black mb-6 bg-gradient-to-r from-cyan-300 via-purple-300 to-pink-300 bg-clip-text text-transparent leading-tight">
                Mudra
                <span className="bg-gradient-to-r from-pink-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  Setu
                </span>
                <br />
                <span className="text-6xl md:text-7xl bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                  Intelligence
                </span>
              </h1>
            </motion.div>

            <motion.p
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="text-xl text-cyan-100/80 mb-12 max-w-3xl mx-auto leading-relaxed font-light"
            >
              Experience the future of sign language recognition with our
              advanced AI-powered platform. Real-time neural processing,
              immersive 3D visualization, and cutting-edge deep learning models
              converge to create the ultimate gesture classification experience.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
              className="flex flex-col sm:flex-row gap-6 justify-center items-center"
            >
              <Link href="/predict">
                <Button
                  size="lg"
                  className="group relative bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-600 hover:to-purple-700 text-white px-10 py-6 text-lg font-semibold rounded-xl overflow-hidden hover:cursor-pointer"
                >
                  <motion.div
                    className="absolute inset-0 bg-gradient-to-r from-cyan-400 to-purple-500 opacity-0 group-hover:opacity-100 transition-opacity "
                    whileHover={{ scale: 1.05 }}
                  />
                  <span className="relative flex items-center">
                    <Brain className="mr-3 h-6 w-6" />
                    Start Neural Analysis
                    <ArrowRight className="ml-3 h-6 w-6 group-hover:translate-x-1 transition-transform" />
                  </span>
                </Button>
              </Link>

              <Link href="/visualize">
                <Button
                  size="lg"
                  variant="outline"
                  className="border-2 border-cyan-400/50 text-cyan-300 hover:bg-cyan-400/10 hover:border-cyan-400 px-10 py-6 text-lg font-semibold rounded-xl backdrop-blur-sm"
                >
                  <Sparkles className="mr-3 h-6 w-6" />
                  Explore 3D Space
                </Button>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </motion.section>

      {/* Features Section */}
      <section className="py-32 px-6 relative">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-20"
          >
            <h2 className="text-5xl font-bold bg-gradient-to-r from-cyan-300 to-purple-300 bg-clip-text text-transparent mb-6">
              Advanced Neural Capabilities
            </h2>
            <p className="text-cyan-100/70 text-xl max-w-3xl mx-auto">
              Harness the power of artificial intelligence with our
              comprehensive suite of gesture recognition tools
            </p>
          </motion.div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: Camera,
                title: "Real-time Neural Processing",
                description:
                  "Live webcam integration with instant AI-powered gesture recognition and confidence scoring",
                gradient: "from-cyan-500 to-blue-600",
                delay: 0,
              },
              {
                icon: ImageIcon,
                title: "Deep Learning Analysis",
                description:
                  "Upload and analyze images using your custom-trained neural networks with detailed insights",
                gradient: "from-purple-500 to-pink-600",
                delay: 0.2,
              },
              {
                icon: Cpu,
                title: "3D Neural Visualization",
                description:
                  "Immersive three-dimensional gesture representation with real-time neural network feedback",
                gradient: "from-pink-500 to-cyan-600",
                delay: 0.4,
              },
            ].map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 50, rotateX: -15 }}
                whileInView={{ opacity: 1, y: 0, rotateX: 0 }}
                transition={{ delay: feature.delay, duration: 0.8 }}
                whileHover={{
                  y: -10,
                  rotateX: 5,
                  transition: { duration: 0.3 },
                }}
                viewport={{ once: true }}
                className="group perspective-1000"
              >
                <Card className="bg-gradient-to-br from-slate-900/50 to-purple-900/30 border border-cyan-400/20 backdrop-blur-xl hover:border-cyan-400/50 transition-all duration-500 overflow-hidden relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-cyan-400/5 to-purple-400/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                  <CardContent className="p-8 text-center relative z-10">
                    <motion.div
                      className={`inline-flex p-6 rounded-2xl bg-gradient-to-r ${feature.gradient} mb-8 relative`}
                      whileHover={{ scale: 1.1, rotate: 5 }}
                      transition={{ duration: 0.3 }}
                    >
                      <feature.icon className="h-10 w-10 text-white" />
                      <motion.div
                        className="absolute inset-0 rounded-2xl bg-white/20"
                        animate={{
                          scale: [1, 1.2, 1],
                          opacity: [0, 0.5, 0],
                        }}
                        transition={{
                          duration: 2,
                          repeat: Number.POSITIVE_INFINITY,
                          delay: index * 0.5,
                        }}
                      />
                    </motion.div>

                    <h3 className="text-2xl font-bold text-cyan-100 mb-4">
                      {feature.title}
                    </h3>
                    <p className="text-cyan-100/70 leading-relaxed">
                      {feature.description}
                    </p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Interactive Demo Section */}
      <section className="py-32 px-6 relative">
        <div className="container mx-auto">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="bg-gradient-to-r from-cyan-900/20 to-purple-900/20 rounded-3xl p-12 text-center border border-cyan-400/20 backdrop-blur-xl relative overflow-hidden"
          >
            {/* Animated background pattern */}
            <div className="absolute inset-0 opacity-10">
              {Array.from({ length: 12 }).map((_, i) => (
                <motion.div
                  key={i}
                  className="absolute w-32 h-32 border border-cyan-400 rounded-full"
                  style={{
                    left: `${(i % 4) * 25}%`,
                    top: `${Math.floor(i / 4) * 33}%`,
                  }}
                  animate={{
                    scale: [1, 1.2, 1],
                    rotate: [0, 180, 360],
                  }}
                  transition={{
                    duration: 8,
                    repeat: Number.POSITIVE_INFINITY,
                    delay: i * 0.5,
                  }}
                />
              ))}
            </div>

            <div className="relative z-10">
              <motion.h2
                className="text-5xl font-bold text-cyan-100 mb-6"
                animate={{
                  backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
                }}
                transition={{ duration: 3, repeat: Number.POSITIVE_INFINITY }}
              >
                Ready to Experience the Future?
              </motion.h2>

              <p className="text-cyan-100/80 text-xl mb-10 max-w-3xl mx-auto">
                Join thousands of researchers, developers, and innovators using
                MudraSetu to push the boundaries of human-computer interaction
                through advanced AI.
              </p>

              <div className="flex flex-col sm:flex-row gap-6 justify-center">
                <Link href="/predict">
                  <Button
                    size="lg"
                    className="bg-gradient-to-r from-cyan-500 to-purple-600 hover:from-cyan-600 hover:to-purple-700 text-white px-12 py-6 text-lg font-semibold rounded-xl"
                  >
                    <Upload className="mr-3 h-6 w-6" />
                    Upload Neural Model
                  </Button>
                </Link>

                <Link href="/dashboard">
                  <Button
                    size="lg"
                    variant="outline"
                    className="border-2 border-cyan-400/50 text-cyan-300 hover:bg-cyan-400/10 px-12 py-6 text-lg font-semibold rounded-xl"
                  >
                    <Activity className="mr-3 h-6 w-6" />
                    View Analytics
                  </Button>
                </Link>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 px-6">
        <div className="container mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {[
              { value: "99.2%", label: "Neural Accuracy", icon: Target },
              { value: "0.3s", label: "Processing Speed", icon: Zap },
              { value: "50K+", label: "Gestures Analyzed", icon: Hand },
              { value: "24/7", label: "AI Availability", icon: Activity },
            ].map((stat, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                viewport={{ once: true }}
                className="text-center"
              >
                <div className="bg-gradient-to-br from-cyan-500/10 to-purple-500/10 rounded-2xl p-6 border border-cyan-400/20 backdrop-blur-sm">
                  <stat.icon className="h-8 w-8 text-cyan-400 mx-auto mb-4" />
                  <div className="text-3xl font-bold text-cyan-100 mb-2">
                    {stat.value}
                  </div>
                  <div className="text-cyan-100/70 text-sm">{stat.label}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-cyan-400/20 bg-black/20 backdrop-blur-xl">
        <div className="container mx-auto text-center">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <Hand className="h-8 w-8 text-cyan-400" />
            <span className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              MudraSetu
            </span>
          </div>
          <p className="text-cyan-100/50 font-mono">
            © 2025 MudraSetu AI. Advancing human-computer interaction through
            neural intelligence.
          </p>
        </div>
      </footer>
    </div>
  );
}
