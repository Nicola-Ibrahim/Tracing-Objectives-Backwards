# ✍️ Doc-as-Code Guide

To ensure our documentation remains a premium resource, please follow these "Doc-as-Code" principles.

---

## 🏛️ Structure: The Diátaxis Framework

All documentation must fall into one of these four categories:

1.  **Tutorials**: Learning-oriented. Step-by-step for newcomers.
2.  **How-to Guides**: Problem-oriented. "How do I add a new model?"
3.  **Reference**: Information-oriented. Mathematical formulas, API schemas, CLI flags.
4.  **Explanation**: Understanding-oriented. Architecture, DDD theory, "Why we use GPBI".

---

## 🎨 Style Guidelines

- **Active Voice**: Use "The system generates..." instead of "The generation is performed by...".
- **Absolute File Links**: Always use `[file basename](file:///absolute/path/to/file)` to link to code.
- **Visuals First**: If you can explain it with a **Mermaid** diagram, do so.
- **No Placeholders**: Never leave "TO DO" or "TBD" in a merged document.

---

## 🏗️ Architectural Integrity

- **Centralized Theory**: Do not repeat DDD layer definitions in module docs. Link to the **[Central DDD Guide](../concepts/ddd-architecture-guide.md)**.
- **Metric Consistency**: Ensure mathematical metrics match the implementation in `shared/reasons.py`.

---

## 🛠️ Maintenance Workflow

Every time a core module is refactored:
1.  Update the corresponding `architecture/module.md`.
2.  Verify the `architecture/README.md` diagram still reflects current dependencies.
3.  If a new concept is introduced, add it to `concepts/`.
