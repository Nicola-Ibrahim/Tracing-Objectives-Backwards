import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

import Navigation from "@/components/Navigation";

import { ToastProvider } from "@/components/ui/ToastContext";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Pareto Optimization",
  description: "Tracing Objectives Backwards - Candidate Generation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen bg-background`}
      >
        <ToastProvider>
          <div className="max-w-6xl mx-auto px-4 py-8">
            <header className="mb-12">
              <h1 className="text-3xl font-bold text-foreground mb-2">
                Tracing Objectives Backwards
              </h1>
              <p className="text-slate-500">Pareto-Optimized Candidate Generation</p>
            </header>
            <Navigation />
            <main>{children}</main>
          </div>
        </ToastProvider>
      </body>
    </html>
  );
}
