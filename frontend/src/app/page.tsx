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
  Layers,
  Cpu
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
      <div className="absolute top-0 left-0 w-full h-[1200px] overflow-hidden z-0 pointer-events-none">
        <div className="absolute top-[-20%] right-[-10%] w-[80%] h-[80%] bg-indigo-100/40 rounded-full blur-[160px] animate-pulse" />
        <div className="absolute top-[10%] left-[-10%] w-[60%] h-[60%] bg-blue-100/30 rounded-full blur-[140px]" />
        <div className="absolute top-[40%] right-[10%] w-[40%] h-[40%] bg-purple-100/20 rounded-full blur-[120px]" />

        {/* Grid pattern overlay */}
        <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 mix-blend-overlay"></div>
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#80808012_1px,transparent_1px),linear-gradient(to_bottom,#80808012_1px,transparent_1px)] bg-size-[60px_60px]"></div>
      </div>

      {/* Navigation / Header */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="relative z-50 container mx-auto px-6 py-8 flex items-center justify-between"
      >
        <Logo className="scale-110" />
        <nav className="hidden md:flex items-center gap-8 text-sm font-bold text-slate-500 uppercase tracking-widest">
          <Link href="#features" className="hover:text-indigo-600 transition-colors">Infrastructure</Link>
          <Link href="/datasets" className="hover:text-indigo-600 transition-colors">Data Hub</Link>
          <Button variant="outline" className="rounded-2xl border-slate-200 bg-white shadow-sm hover:shadow-md transition-all font-bold" asChild>
            <Link href="/datasets">Open Console</Link>
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
              <span className="bg-clip-text text-transparent bg-linear-to-r from-indigo-600 via-blue-600 to-indigo-600">
                Backwards
              </span>
            </motion.h1>

            <motion.p
              {...fadeIn}
              transition={{ delay: 0.2 }}
              className="text-xl md:text-2xl text-slate-500 mb-14 leading-relaxed max-w-3xl mx-auto font-medium"
            >
              Invert high-dimensional models with mathematical precision.
              Find the optimal path from target outcomes back to their decision parameters.
            </motion.p>

            <motion.div
              {...fadeIn}
              transition={{ delay: 0.4 }}
              className="flex flex-wrap items-center justify-center gap-8"
            >
              <Button size="lg" className="rounded-3xl h-20 px-12 text-2xl font-black shadow-2xl shadow-indigo-100 transition-all hover:scale-105 active:scale-95 bg-slate-950 text-white hover:bg-slate-900 border-0 group" asChild>
                <Link href="/inverse/generate">
                  Launch Explorer
                  <ArrowRight className="ml-3 h-8 w-8 group-hover:translate-x-2 transition-transform" />
                </Link>
              </Button>
              <Button size="lg" variant="ghost" className="rounded-3xl h-20 px-10 text-xl font-bold text-slate-600 hover:text-indigo-600 hover:bg-indigo-50/50 transition-all outline outline-slate-200" asChild>
                <Link href="/datasets">
                  Data Workbench
                </Link>
              </Button>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Infrastructure Core Section */}
      <section id="features" className="py-32 relative z-10 bg-white/60 backdrop-blur-3xl border-y border-slate-200/50">
        <div className="container mx-auto px-6">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-24"
          >
            <h2 className="text-5xl md:text-6xl font-black tracking-tight mb-6 text-slate-900">Built for Researchers</h2>
            <p className="text-xl text-slate-500 max-w-2xl mx-auto font-medium">
              Four specialized environments designed for high-fidelity multi-objective optimization and inverse mapping.
            </p>
          </motion.div>

          <motion.div
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
          >
            {/* Data Hub */}
            <motion.div variants={fadeIn} className="bg-white border border-slate-200 rounded-[2.5rem] p-8 hover:shadow-xl hover:border-indigo-100 transition-all group relative overflow-hidden">
              <div className="absolute top-0 right-0 p-8 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity text-slate-900">
                <Database size={120} />
              </div>
              <div className="w-12 h-12 rounded-2xl bg-indigo-600 text-white flex items-center justify-center mb-8 shadow-lg shadow-indigo-100">
                <Database className="h-6 w-6" />
              </div>
              <h3 className="text-2xl font-black mb-4 tracking-tight text-slate-900">Data Analytics Hub</h3>
              <p className="text-slate-500 text-sm leading-relaxed mb-8 font-medium">
                Manage high-dimensional reference populations. Profiling of X and Y spaces with automated distribution analysis.
              </p>
              <Link href="/datasets" className="inline-flex items-center gap-2 font-black text-indigo-600 tracking-widest uppercase text-[10px] hover:gap-4 transition-all">
                Enter Hub <ArrowRight className="h-3 w-3" />
              </Link>
            </motion.div>

            {/* Train Engine */}
            <motion.div variants={fadeIn} className="bg-white border border-slate-200 rounded-[2.5rem] p-8 hover:shadow-xl hover:border-blue-100 transition-all group relative overflow-hidden">
              <div className="absolute top-0 right-0 p-8 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity text-slate-900">
                <Cpu size={120} />
              </div>
              <div className="w-12 h-12 rounded-2xl bg-blue-600 text-white flex items-center justify-center mb-8 shadow-lg shadow-blue-100">
                <Cpu className="h-6 w-6" />
              </div>
              <h3 className="text-2xl font-black mb-4 tracking-tight text-slate-900">Engine Construction</h3>
              <p className="text-slate-500 text-sm leading-relaxed mb-8 font-medium">
                Construct and train specialized Inverse Solvers. Monitor loss convergence and architectural benchmarks.
              </p>
              <Link href="/inverse/train" className="inline-flex items-center gap-2 font-black text-blue-600 tracking-widest uppercase text-[10px] hover:gap-4 transition-all">
                Configure Solver <ArrowRight className="h-3 w-3" />
              </Link>
            </motion.div>

            {/* Inference Hub */}
            <motion.div variants={fadeIn} className="bg-white border border-slate-200 rounded-[2.5rem] p-8 hover:shadow-xl hover:border-purple-100 transition-all group relative overflow-hidden">
              <div className="absolute top-0 right-0 p-8 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity text-slate-900">
                <Sparkles size={120} />
              </div>
              <div className="w-12 h-12 rounded-2xl bg-purple-600 text-white flex items-center justify-center mb-8 shadow-lg shadow-purple-100">
                <Sparkles className="h-6 w-6" />
              </div>
              <h3 className="text-2xl font-black mb-4 tracking-tight text-slate-900">Inference Registry</h3>
              <p className="text-slate-500 text-sm leading-relaxed mb-8 font-medium">
                Generate high-fidelity candidate solutions for target objectives using GBPI and Probabilistic solvers.
              </p>
              <Link href="/inverse/generate" className="inline-flex items-center gap-2 font-black text-purple-600 tracking-widest uppercase text-[10px] hover:gap-4 transition-all">
                Generate Solutions <ArrowRight className="h-3 w-3" />
              </Link>
            </motion.div>

            {/* Evaluation */}
            <motion.div variants={fadeIn} className="bg-white border border-slate-200 rounded-[2.5rem] p-8 hover:shadow-xl hover:border-emerald-100 transition-all group relative overflow-hidden">
              <div className="absolute top-0 right-0 p-8 opacity-[0.03] group-hover:opacity-[0.08] transition-opacity text-slate-900">
                <BarChart3 size={120} />
              </div>
              <div className="w-12 h-12 rounded-2xl bg-emerald-600 text-white flex items-center justify-center mb-8 shadow-lg shadow-emerald-100">
                <BarChart3 className="h-6 w-6" />
              </div>
              <h3 className="text-2xl font-black mb-4 tracking-tight text-slate-900">Leaderboard Metrics</h3>
              <p className="text-slate-500 text-sm leading-relaxed mb-8 font-medium">
                Verify performance with MACE, PIT, and ECDF calibration. Compare solver robustness across datasets.
              </p>
              <Link href="/evaluation" className="inline-flex items-center gap-2 font-black text-emerald-600 tracking-widest uppercase text-[10px] hover:gap-4 transition-all">
                View Benchmarks <ArrowRight className="h-3 w-3" />
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Registry Summary Section */}
      <section className="py-32 bg-slate-50/50">
        <div className="container mx-auto px-6">
          <div className="bg-white border border-slate-200 rounded-[4rem] p-16 flex flex-col items-center text-center relative overflow-hidden group shadow-2xl shadow-slate-200">
            <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(circle_at_50%_50%,rgba(99,102,241,0.03),transparent_70%)]"></div>
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              whileInView={{ scale: 1, opacity: 1 }}
              viewport={{ once: true }}
              className="relative z-10"
            >
              <Logo simple className="scale-150 mb-12 opacity-30 grayscale hover:grayscale-0 transition-all duration-700" />
              <h2 className="text-4xl md:text-5xl font-black tracking-tight mb-8 text-slate-900">Ready to invert your models?</h2>
              <div className="flex flex-wrap justify-center gap-6">
                <Button size="lg" className="rounded-2xl h-16 px-8 bg-indigo-600 hover:bg-indigo-700 text-white font-bold shadow-lg shadow-indigo-100 transition-all active:scale-95" asChild>
                  <Link href="/inverse/train">Initialize Training</Link>
                </Button>
                <Button size="lg" variant="outline" className="rounded-2xl h-16 px-8 border-slate-200 bg-white hover:bg-slate-50 font-bold transition-all active:scale-95" asChild>
                  <Link href="/datasets">Explore Datasets</Link>
                </Button>
              </div>
            </motion.div>
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
                Advanced mathematical protocol for multi-objective optimization and inverse mapping.
                Built for researchers who demand precision.
              </p>
            </div>
            <div className="md:col-span-2">
              <h4 className="font-black uppercase tracking-widest text-[10px] text-slate-400 mb-8">Navigation</h4>
              <ul className="space-y-4 font-bold text-slate-500 text-sm">
                <li><Link href="/datasets" className="hover:text-indigo-600 transition-all">Data Hub</Link></li>
                <li><Link href="/inverse/generate" className="hover:text-indigo-600 transition-all">Explorer</Link></li>
                <li><Link href="/evaluation" className="hover:text-indigo-600 transition-all">Leaderboard</Link></li>
              </ul>
            </div>
            <div className="md:col-span-5 flex flex-col items-end">
              <div className="bg-slate-50 border border-slate-200 p-8 rounded-[2.5rem] w-full max-w-sm flex items-center justify-between group cursor-pointer hover:bg-slate-100 transition-all shadow-sm">
                <div className="flex flex-col">
                  <span className="font-black text-2xl tracking-tighter italic text-slate-400 group-hover:text-indigo-600 transition-colors">Start Generating</span>
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Target Objective v1.2</span>
                </div>
                <div className="w-12 h-12 rounded-2xl bg-slate-950 flex items-center justify-center text-white group-hover:scale-110 transition-transform">
                  <ArrowRight />
                </div>
              </div>
            </div>
          </div>

          <div className="flex flex-col md:flex-row items-center justify-between gap-6 pt-12 border-t border-slate-100">
            <p className="text-slate-400 text-[10px] font-black uppercase tracking-[0.3em]">
              © 2026 TRACING OBJECTIVES BACKWARDS
            </p>
            <div className="flex items-center gap-8 text-[10px] font-bold text-slate-400 uppercase tracking-widest">
              <span className="hover:text-slate-600 cursor-pointer transition-colors">Security</span>
              <span className="hover:text-slate-600 cursor-pointer transition-colors">API</span>
              <span className="hover:text-slate-600 cursor-pointer transition-colors">Docs</span>
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
