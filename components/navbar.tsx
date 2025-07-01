"use client";

import { motion } from "framer-motion";
import { Hand } from "lucide-react";
import Link from "next/link";
import React from "react";
import { Button } from "./ui/button";
import { SignIn, useClerk } from "@clerk/nextjs";
import { SignedIn, SignedOut, SignInButton, UserButton } from "@clerk/nextjs";
import { useRouter } from "next/navigation";
interface Props {
  isLoggedIn: boolean | null;
}

export default function Navbar({ isLoggedIn }: Props) {
  const { signOut } = useClerk();
  const router = useRouter();
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
              ></motion.div>
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
            <SignedIn>
              <Link
                href="/predict"
                className="text-cyan-300/80 hover:text-cyan-300 transition-colors font-medium"
              >
                Neural Predict
              </Link>
              <Link
                href="/videoInput"
                className="text-cyan-300/80 hover:text-cyan-300 transition-colors font-medium"
              >
                Video Predict
              </Link>
              <Link
                href="/visualize"
                className="text-cyan-300/80 hover:text-cyan-300 transition-colors font-medium"
              >
                3D Visualize
              </Link>
              <Link
                href="/video-call"
                className="text-cyan-300/80 hover:text-cyan-300 transition-colors font-medium"
              >
                Video Call
              </Link>
            </SignedIn>
            <SignedOut>
              <SignInButton>
                <Button
                  variant="secondary"
                  className="relative overflow-hidden px-8 py-2 text-cyan-300/90 hover:text-white bg-gradient-to-r from-cyan-500/20 via-purple-500/10 to-pink-500/20 border border-cyan-400/40 shadow-md hover:shadow-lg transition-all duration-300 hover:cursor-pointer  text-xl rounded-full group"
                >
                  <span className="absolute left-0 top-0 w-full h-full bg-cyan-400/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-sm"></span>
                  <span className="relative z-10">Sign In</span>
                </Button>
              </SignInButton>
            </SignedOut>
            <SignedIn>
              <UserButton
                appearance={{
                  elements: {
                    avatarBox: "w-9 h-9 border-2 border-primary/20",
                    userButtonPopoverCard: "shadow-lg",
                    userPreviewMainIdentifier: "font-semibold",
                  },
                }}
                afterSignOutUrl="/"
              />
            </SignedIn>
          </motion.div>
        </div>
      </div>
    </nav>
  );
}
