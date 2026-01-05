/* global Reveal, RevealMarkdown, RevealNotes, RevealHighlight */

Reveal.initialize({
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
