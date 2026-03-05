"use client";

import React from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  ArrowRight,
  Database,
  Sparkles,
  Target,
  BarChart3,
  Zap,
  ShieldCheck,
  ChevronRight,
  SearchCode,
  Globe,
  Layers
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Logo } from "@/components/ui/Logo";

const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-slate-50 text-slate-900 selection:bg-indigo-100 selection:text-indigo-900 overflow-x-hidden font-sans">
      {/* Dynamic Background Decor */}
      <div className="absolute top-0 left-0 w-full h-[1000px] overflow-hidden z-0 pointer-events-none">
        <div className="absolute top-[-10%] right-[-5%] w-[60%] h-[60%] bg-indigo-100/40 rounded-full blur-[120px] animate-pulse" />
        <div className="absolute top-[20%] left-[-10%] w-[40%] h-[40%] bg-blue-100/30 rounded-full blur-[100px]" />

        {/* Grid pattern overlay */}
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-size-[40px_40px]"></div>
      </div>

      {/* Navigation / Header */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="relative z-50 container mx-auto px-6 py-8 flex items-center justify-between"
      >
        <Logo className="scale-110" />
        <nav className="hidden md:flex items-center gap-8 text-sm font-bold text-slate-500 uppercase tracking-widest">
          <Link href="#features" className="hover:text-indigo-600 transition-colors">Features</Link>
          <Link href="/datasets" className="hover:text-indigo-600 transition-colors">Data Hub</Link>
          <Button variant="outline" className="rounded-2xl border-indigo-100 bg-white shadow-sm hover:shadow-md transition-all font-bold" asChild>
            <Link href="/datasets">Open App</Link>
          </Button>
        </nav>
      </motion.header>

      {/* Hero Section */}
      <section className="relative pt-16 pb-24 md:pt-32 md:pb-48 z-10">
        <div className="container mx-auto px-6">
          <div className="max-w-5xl mx-auto text-center">
            <motion.div
              {...fadeIn}
              className="inline-block"
            >
              <Badge variant="outline" className="mb-8 px-5 py-2 border-indigo-200 bg-indigo-50/50 text-indigo-700 font-bold tracking-widest uppercase text-[10px] shadow-sm">
                <Sparkles className="h-3 w-3 mr-2 text-indigo-500" />
                Advanced Inverse Mapping Protocol v1.2.0
              </Badge>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
              className="text-7xl md:text-9xl font-black tracking-tighter text-slate-900 mb-10 leading-[0.85] text-balance"
            >
              Trace Your Objectives <br />
              <span className="bg-clip-text text-transparent bg-linear-to-r from-indigo-600 via-indigo-500 to-blue-600">
                Backwards
              </span>
            </motion.h1>

            <motion.p
              {...fadeIn}
              transition={{ delay: 0.2 }}
              className="text-xl md:text-2xl text-slate-500 mb-14 leading-relaxed max-w-3xl mx-auto font-medium"
            >
              Invert high-dimensional models with surgical precision.
              Find the shortest path from your target outcomes back to their optimal decision parameters.
            </motion.p>

            <motion.div
              {...fadeIn}
              transition={{ delay: 0.4 }}
              className="flex flex-wrap items-center justify-center gap-8"
            >
              <Button size="lg" className="rounded-3xl h-20 px-12 text-2xl font-black shadow-2xl shadow-indigo-200 transition-all hover:scale-105 active:scale-95 bg-slate-950 hover:bg-slate-900 text-white border-0 group" asChild>
                <Link href="/inverse/generate">
                  Launch Explorer
                  <ArrowRight className="ml-3 h-8 w-8 group-hover:translate-x-2 transition-transform" />
                </Link>
              </Button>
              <Button size="lg" variant="ghost" className="rounded-3xl h-20 px-10 text-xl font-bold text-slate-600 hover:text-indigo-600 hover:bg-indigo-50/50 transition-all" asChild>
                <Link href="/datasets">
                  View Data Workbench
                </Link>
              </Button>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Feature Bento Grid */}
      <section id="features" className="py-32 relative z-10 bg-white/60 backdrop-blur-3xl border-y border-slate-200/50">
        <div className="container mx-auto px-6">
          <motion.div
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-6 gap-6"
          >
            {/* Main Block */}
            <motion.div variants={fadeIn} className="md:col-span-3 bg-white border border-slate-200 rounded-[3rem] p-12 shadow-sm hover:shadow-xl transition-all group overflow-hidden relative">
              <div className="absolute bottom-0 right-0 p-12 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity">
                <Target size={300} strokeWidth={1} />
              </div>
              <div className="relative z-10">
                <div className="w-16 h-16 rounded-[1.5rem] bg-indigo-600 text-white flex items-center justify-center mb-10 shadow-lg shadow-indigo-100">
                  <Target className="h-8 w-8" />
                </div>
                <h3 className="text-4xl font-black text-slate-900 mb-6 tracking-tight leading-none italic">Inverse Mapping Engine</h3>
                <p className="text-slate-500 text-lg leading-relaxed mb-10 font-medium">
                  Convert target objectives into precise decision vectors using GBPI solvers.
                  Navigate the Pareto front with visual cohort tracking and real-time convergence diagnostics.
                </p>
                <Link href="/inverse/generate" className="inline-flex items-center gap-2 font-black text-indigo-600 tracking-widest uppercase text-xs hover:gap-4 transition-all">
                  Infrastructure Deep Dive <ArrowRight className="h-4 w-4" />
                </Link>
              </div>
            </motion.div>

            {/* Secondary Block 1 */}
            <motion.div variants={fadeIn} className="md:col-span-3 bg-slate-950 rounded-[3rem] p-12 shadow-2xl overflow-hidden relative group">
              <div className="absolute top-0 right-0 w-full h-full bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,0.02)_50%,transparent_75%)] bg-size-[400%_400%] animate-[shimmer_8s_infinite]"></div>
              <div className="relative z-10">
                <div className="w-16 h-16 rounded-[1.5rem] bg-white/10 flex items-center justify-center mb-10 backdrop-blur-xl border border-white/10">
                  <Database className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-4xl font-black text-white mb-6 tracking-tight leading-none">Data Intelligence</h3>
                <p className="text-slate-400 text-lg leading-relaxed mb-10 font-medium">
                  Seamlessly manage reference populations. Automated profiling of X and Y spaces with 1D/2D distribution analysis.
                </p>
                <Link href="/datasets" className="inline-flex items-center gap-2 font-black text-indigo-400 tracking-widest uppercase text-xs">
                  Access Workspace <Globe className="h-4 w-4 ml-1" />
                </Link>
              </div>
            </motion.div>

            {/* Diagnostics Block */}
            <motion.div variants={fadeIn} className="md:col-span-4 bg-indigo-50 border border-indigo-100 rounded-[3rem] p-12 shadow-sm flex flex-col md:flex-row gap-12 items-center">
              <div className="flex-1">
                <h3 className="text-3xl font-black text-slate-950 mb-6 tracking-tight">Advanced Diagnostics</h3>
                <p className="text-slate-600 text-base leading-relaxed mb-8 font-medium">
                  Verify your inverse models with MACE, PIT, and ECDF calibration metrics. Ensure every generation is statistically robust and numerically stable.
                </p>
                <div className="flex gap-4">
                  <div className="bg-white p-4 rounded-2xl shadow-sm border border-indigo-100 flex-1">
                    <BarChart3 className="text-indigo-600 mb-2 h-5 w-5" />
                    <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Error Metrics</p>
                    <p className="text-xl font-bold text-slate-900">MACE-Suite</p>
                  </div>
                  <div className="bg-white p-4 rounded-2xl shadow-sm border border-indigo-100 flex-1">
                    <Layers className="text-indigo-600 mb-2 h-5 w-5" />
                    <p className="text-[10px] font-black uppercase tracking-widest text-slate-400">Distribution</p>
                    <p className="text-xl font-bold text-slate-900">PIT-Analysis</p>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Performance Block */}
            <motion.div variants={fadeIn} className="md:col-span-2 bg-linear-to-br from-indigo-600 to-indigo-800 rounded-[3rem] p-12 text-white shadow-xl flex flex-col justify-end">
              <Zap className="h-10 w-10 text-indigo-200 mb-8" />
              <h3 className="text-5xl font-black mb-2 tracking-tighter leading-none italic">&lt;2ms</h3>
              <p className="text-indigo-100 font-bold uppercase tracking-widest text-xs opacity-80">Inference Latency</p>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Social Proof / Stats */}
      <section className="py-32">
        <div className="container mx-auto px-6">
          <div className="max-w-4xl mx-auto flex flex-col md:flex-row items-center justify-between gap-16 md:gap-8 opacity-40 grayscale">
            <Logo simple className="scale-125" />
            <div className="h-px w-20 bg-slate-200 hidden md:block"></div>
            <div className="flex items-center gap-4">
              <ShieldCheck className="h-6 w-6" />
              <span className="font-black uppercase tracking-tighter text-lg italic">Certified Robustness</span>
            </div>
            <div className="h-px w-20 bg-slate-200 hidden md:block"></div>
            <div className="flex items-center gap-4">
              <SearchCode className="h-6 w-6" />
              <span className="font-black uppercase tracking-tighter text-lg italic">Statistically Verified</span>
            </div>
          </div>
        </div>
      </section>

      {/* Modern Footer */}
      <footer className="py-24 border-t border-slate-200 bg-white relative z-10">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-1 md:grid-cols-12 gap-16 mb-20">
            <div className="md:col-span-5">
              <Logo className="mb-8" />
              <p className="text-slate-500 text-lg leading-relaxed max-w-sm font-medium">
                Advanced mathematical playground for multi-objective optimization and inverse mapping.
                Built for researchers and engineers who demand precision.
              </p>
            </div>
            <div className="md:col-span-2">
              <h4 className="font-black uppercase tracking-widest text-xs text-slate-400 mb-8">Navigation</h4>
              <ul className="space-y-4 font-bold text-slate-600">
                <li><Link href="/datasets" className="hover:text-indigo-600 transition-all">Data Hub</Link></li>
                <li><Link href="/inverse/generate" className="hover:text-indigo-600 transition-all">Inference</Link></li>
                <li><Link href="/evaluation" className="hover:text-indigo-600 transition-all">Evaluation</Link></li>
              </ul>
            </div>
            <div className="md:col-span-5 flex flex-col items-end">
              <div className="bg-slate-50 border border-slate-200 p-8 rounded-[2rem] w-full max-w-sm flex items-center justify-between group cursor-pointer hover:bg-slate-100 transition-all shadow-sm">
                <div className="flex flex-col">
                  <span className="font-black text-2xl tracking-tighter italic text-slate-400 group-hover:text-indigo-600 transition-colors">Start Generating</span>
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Target Objective v1.2</span>
                </div>
                <div className="w-12 h-12 rounded-2xl bg-slate-900 flex items-center justify-center text-white group-hover:scale-110 transition-transform">
                  <ArrowRight />
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col md:flex-row items-center justify-between gap-6 pt-12 border-t border-slate-100">
            <p className="text-slate-400 text-xs font-black uppercase tracking-[0.3em]">
              © 2026 Tracing Objectives Backwards
            </p>
            <div className="flex items-center gap-8 text-[10px] font-bold text-slate-400 uppercase tracking-widest">
              <span className="hover:text-slate-600 cursor-pointer">Security Protocol</span>
              <span className="hover:text-slate-600 cursor-pointer">API Integration</span>
              <span className="hover:text-slate-600 cursor-pointer">Documentation</span>
            </div>
          </div>
        </div>
      </footer>

      {/* Global Transition Overlay */}
      <motion.div
        initial={{ opacity: 1 }}
        animate={{ opacity: 0 }}
        transition={{ duration: 0.5, ease: "easeOut" }}
        className="fixed inset-0 bg-slate-950 z-100 pointer-events-none"
      />
    </div>
  );
}
