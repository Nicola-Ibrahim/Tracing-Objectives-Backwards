/* global Reveal, RevealMarkdown, RevealNotes, RevealHighlight */

Reveal.initialize({
  width: 2048,
  height: 1152,
  margin: 0.04,
  minScale: 0.9,
  maxScale: 6.0,
  hash: true,
  slideNumber: "c/t",
  progress: true,
  controls: true,
  center: true,
  transition: "fade",
  backgroundTransition: "fade",
  transitionSpeed: "normal",
  autoAnimate: true,
  plugins: [RevealMarkdown, RevealNotes, RevealHighlight, RevealMath.KaTeX],
  markdown: {
    smartypants: true,
  },
});
