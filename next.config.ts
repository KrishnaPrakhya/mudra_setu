import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    domains: ["vercel.com"],
  },
  eslint:{
    ignoreDuringBuilds:true
  }
};

export default nextConfig;
