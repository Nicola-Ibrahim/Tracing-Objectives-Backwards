/* global Reveal, RevealMarkdown, RevealNotes, RevealHighlight */

Reveal.initialize({
  hash: true,
  slideNumber: true,
  progress: true,
  controls: true,
  center: false,
  transition: "convex",
  backgroundTransition: "fade",
  transitionSpeed: "fast",
  autoAnimate: true,
  plugins: [RevealMarkdown, RevealNotes, RevealHighlight],
  markdown: {
    smartypants: true,
  },
});
