# Thesis Presentation (reveal.js)

## Structure
- `index.html` loads reveal.js and the markdown slides
- `slides.md` holds slide content
- `css/custom.css` stores theme overrides
- `js/presentation.js` configures reveal.js
- `assets/` for images/figures

## Run locally
```bash
npm start
```

Then open `http://localhost:8000`.

## Authoring tips
- Use `---` for horizontal slides and `--` for vertical stacks.
- Add speaker notes with `Note:` in `slides.md`.
- Keep plots in `assets/` and reference them with relative paths.
