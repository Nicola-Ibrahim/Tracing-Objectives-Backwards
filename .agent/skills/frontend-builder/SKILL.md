---
name: frontend-builder
description: Generates UI components, landing pages, and dashboard layouts using Next.js (App Router), Tailwind CSS, Recharts, and Framer Motion. Use for all frontend architecture and interactive elements.
---
# Frontend Builder Skill

Your goal is to build a scalable Next.js application, cleanly separating marketing pages from the application dashboard, while delivering a highly interactive and animated user experience.

## Core Principles
1. **App Router Structure:** Use Next.js Route Groups (e.g., `(marketing)` for the landing page and `(dashboard)` for the authenticated app) to share layouts within specific sections without affecting the URL path.
2. **Interactivity & Animation:** Use `framer-motion` for complex interactive and animated components (page transitions, complex micro-interactions). Use Tailwind CSS for simple hover/focus states and basic transitions.
3. **Data Visualization:** Use `recharts` wrapped in `<ResponsiveContainer>` for dashboard metrics.
4. **Server vs. Client Components:** Default to Server Components for layouts and data fetching. Extract interactive elements, charts, and Framer Motion animations into isolated Client Components (using `"use client"`).

## Execution Workflow
1. **Identify the Route Context:** Determine if the component belongs in the `(marketing)` landing page (focus on SEO and conversions) or the `(dashboard)` (focus on dense data, charts, and state management).
2. **Scaffold with Tailwind:** Build the semantic HTML structure and style it mobile-first.
3. **Add Interactivity:** If animating, wrap elements in Framer Motion's `<motion.div>` and define animation variants.

## Constraints
* Do not mix marketing layouts with dashboard layouts; keep them isolated in their respective Route Groups.
* Do not make entire page layouts Client Components just to animate one element; isolate the animated component.