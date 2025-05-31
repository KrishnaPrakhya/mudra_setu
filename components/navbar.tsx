"use client";

import { motion } from "framer-motion";
import { Hand } from "lucide-react";
import Link from "next/link";
import React from "react";

export default function Navbar() {
  return (
    <nav className="fixed top-0 w-full z-50 bg-black/10 backdrop-blur-xl border-b border-cyan-400/20">
      <div className="container mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-3"
          >
            <div className="relative">
              <Hand className="h-10 w-10 text-cyan-400" />
              <motion.div
                className="absolute inset-0 h-10 w-10 text-cyan-400"
                animate={{ rotate: 360 }}
                transition={{
                  duration: 8,
                  repeat: Number.POSITIVE_INFINITY,
                  ease: "linear",
                }}
              >
                {/* <Network className="h-10 w-10" /> */}
              </motion.div>
            </div>
            <div>
              <Link href={"/"}>
                <span className="text-2xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                  MudraSetu
                </span>
                <div className="text-xs text-cyan-400/70 font-mono">
                  AI POWERED
                </div>
              </Link>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="flex items-center space-x-8"
          >
            <Link
              href="/predict"
              className="text-cyan-300/80 hover:text-cyan-300 transition-colors font-medium"
            >
              Neural Predict
            </Link>
            <Link
              href="/visualize"
              className="text-cyan-300/80 hover:text-cyan-300 transition-colors font-medium"
            >
              3D Visualize
            </Link>
          </motion.div>
        </div>
      </div>
    </nav>
  );
}
