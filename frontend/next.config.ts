import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  // Note: rewrites are not supported with 'output: export'
  // and will be ignored in the build.
  async rewrites() {
    const backendUrl = process.env.INTERNAL_API_URL || "http://backend:8000";
    return [
      {
        source: "/v1/:path*",
        destination: `${backendUrl}/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
