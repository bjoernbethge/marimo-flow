# Reference Documentation - Getting Started

Welcome to the marimo-flow reference documentation library. This directory contains comprehensive, LLM-friendly reference materials for the core technologies used in this project.

## Start Here

1. **New to these technologies?** → Start with [INDEX.md](INDEX.md)
2. **Looking for quick syntax?** → Jump to the "API Reference" section in any quickstart guide
3. **Need a working example?** → See "Common Patterns" sections
4. **Building something complex?** → Read [integration-patterns.md](integration-patterns.md)
5. **Debugging an issue?** → Check "Common Issues & Solutions" in relevant guide

---

## Files in This Directory

### Navigation & Overview
- **[INDEX.md](INDEX.md)** - Main navigation guide and technology matrix
  - Technology overview matrix
  - Quick command reference
  - Integration patterns checklist
  - Document guide for each technology

- **[README.md](README.md)** - This file
  - Quick orientation
  - File descriptions
  - Usage recommendations

- **[RESEARCH_SUMMARY.md](RESEARCH_SUMMARY.md)** - Documentation research report
  - What was researched and why
  - Quality metrics
  - Version information
  - Maintenance guidelines

### Technology Reference Guides

- **[marimo-quickstart.md](marimo-quickstart.md)** - Reactive Python Notebooks
  - Core concepts (reactivity, state, UI elements)
  - State management with `mo.state()`
  - UI components (sliders, buttons, dropdowns, etc.)
  - Layout and composition patterns
  - Deployment and export options
  - Integration with Polars, Plotly, DuckDB
  - **When to use**: Building interactive notebooks, UIs, dashboards

- **[polars-quickstart.md](polars-quickstart.md)** - High-Performance DataFrames
  - Eager vs lazy evaluation
  - Expression API (pl.col, .select, .filter, etc.)
  - DataFrame operations (group_by, join, with_columns)
  - String and date/time operations
  - Performance optimization tips
  - Window functions and advanced features
  - **When to use**: Data loading, transformation, aggregation, analysis

- **[plotly-quickstart.md](plotly-quickstart.md)** - Interactive Visualizations
  - Plotly Express API (simple, declarative)
  - Graph Objects API (advanced, customizable)
  - 30+ chart types explained
  - Layout customization and theming
  - Subplots and multi-view dashboards
  - Export options (HTML, PNG, PDF)
  - **When to use**: Creating interactive charts, dashboards, data exploration

- **[pina-quickstart.md](pina-quickstart.md)** - Scientific Machine Learning
  - Problem definition and domain setup
  - Model architectures (FeedForward, ResNet, GraphNet)
  - Solvers (Supervised, PINN, DeepONet)
  - Training with PyTorch Lightning
  - Physics-Informed Neural Networks
  - PDE solving examples
  - **When to use**: Neural networks, physics constraints, differential equations

### Advanced Reference

- **[integration-patterns.md](integration-patterns.md)** - Real-World Integration Patterns
  - Interactive data explorer (marimo + polars + plotly)
  - Multi-view dashboard with state synchronization
  - Real-time data pipeline with streaming updates
  - ML model training and evaluation (polars + pina + plotly)
  - Scientific problem solving (PDE with PINA + visualization)
  - Batch processing pipelines
  - Error handling and validation patterns
  - Performance optimization across all technologies
  - Common pitfalls and solutions

---

## Quick Start by Use Case

### "I want to build an interactive notebook"
1. Read [marimo-quickstart.md](marimo-quickstart.md) - Core Concepts
2. Look at patterns in the file
3. Use INDEX.md → Quick Command Reference for syntax

### "I need to load and transform data"
1. Read [polars-quickstart.md](polars-quickstart.md) - Core Concepts
2. Check Common Patterns section for your use case
3. Use API Reference - Quick Lookup for exact syntax

### "I want to create visualizations"
1. Read [plotly-quickstart.md](plotly-quickstart.md) - Core Concepts
2. Browse chart types in the file
3. Copy patterns from Common Patterns section

### "I'm building a machine learning solution"
1. Read [pina-quickstart.md](pina-quickstart.md) - Core Concepts
2. Follow the four-step workflow
3. Look at relevant pattern examples

### "I'm combining multiple technologies"
1. Start with [integration-patterns.md](integration-patterns.md)
2. Find pattern matching your use case
3. Reference specific guides for detailed API info

---

## Key Features of This Documentation

✅ **Complete** - Covers core concepts through advanced patterns
✅ **Current** - All information as of November 2025
✅ **Practical** - 100+ working code examples
✅ **Integrated** - Shows how to use technologies together
✅ **Optimized for LLMs** - Structured for both human and AI reading
✅ **Well-Indexed** - Easy navigation with clear section headings
✅ **Error-Focused** - Includes common issues and solutions
✅ **Performance-Aware** - Includes optimization tips and benchmarks

---

## Documentation Structure

Each technology guide follows this structure:

```
1. Quick Reference
   - Installation commands
   - Basic syntax examples

2. Core Concepts
   - Fundamental ideas with examples
   - Mental models and architecture

3. Common Patterns
   - Real-world use cases
   - Full working implementations

4. Best Practices
   - DO recommendations with examples
   - DON'T anti-patterns with rationale

5. Common Issues & Solutions
   - Real problems and their fixes

6. API Reference - Quick Lookup
   - Most important functions/methods
   - Signatures and basic examples
```

---

## How to Use These Docs

### For Quick Answers
- Use **API Reference** sections for exact syntax
- Check **Quick Reference** for installation/setup
- See **Common Issues** for debugging

### For Learning
- Read **Core Concepts** section in order
- Try the code examples
- Work through **Common Patterns**
- Review **Best Practices**

### For Implementation
- Find relevant **Common Pattern**
- Adapt code to your use case
- Check **Best Practices** before finalizing
- Use **API Reference** for exact syntax

### For Debugging
- Search **Common Issues & Solutions** first
- Check **Best Practices** for preventable errors
- Review **Common Pitfalls** in integration guide
- Verify API usage in **API Reference**

---

## Technology Versions

All documentation is current for:
- **marimo**: Latest (marimo-team/marimo)
- **polars**: Latest (pola-rs/polars)
- **plotly**: Latest (plotly/plotly.py)
- **pina**: Latest (mathLab/PINA)

Versions checked: November 21, 2025

---

## Official Documentation Links

If you need more detail beyond these references:

- **marimo**: https://docs.marimo.io
- **polars**: https://docs.pola.rs/
- **plotly**: https://plotly.com/python/
- **pina**: https://mathlab.github.io/PINA/

---

## Tips for Using This Documentation with LLMs

These documents are optimized for LLM context:

1. **Structured Format** - Clear headings and sections
2. **Code Fences** - Language-tagged code blocks
3. **DO/DON'T Markers** - Easy to parse patterns
4. **Version Info** - Explicit version information
5. **Complete Examples** - Full, runnable code
6. **High Density** - Lots of information per section

You can ask LLMs to:
- "Show me a pattern from integration-patterns.md for..."
- "What's the best practice for... in polars-quickstart.md?"
- "How do I integrate marimo with polars based on these docs?"
- "What does this error mean according to these references?"

---

## Feedback & Updates

### Found an issue?
- Check if docs match the official documentation
- Verify you're using the latest library version
- Check the relevant "Common Issues & Solutions" section

### Want to improve these docs?
- Update with latest library changes
- Add patterns you've found useful
- Improve examples based on feedback
- Correct any inaccuracies

---

## Navigation Quick Links

By Technology:
- [marimo Guide](marimo-quickstart.md)
- [polars Guide](polars-quickstart.md)
- [plotly Guide](plotly-quickstart.md)
- [pina Guide](pina-quickstart.md)

By Use Case:
- [Data Explorer Pattern](integration-patterns.md#pattern-1-interactive-data-explorer)
- [Dashboard Pattern](integration-patterns.md#pattern-2-multi-view-dashboard)
- [Real-Time Pipeline](integration-patterns.md#pattern-3-real-time-data-pipeline)
- [ML Training Pattern](integration-patterns.md#pattern-4-ml-model-training--evaluation)
- [Scientific Computing](integration-patterns.md#pattern-5-scientific-problem-solving)

By Topic:
- [Marimo State Management](marimo-quickstart.md#2-state-management-with-mostate)
- [Polars Expressions](polars-quickstart.md#2-expressions---the-core-language)
- [Plotly Chart Types](plotly-quickstart.md#common-patterns)
- [PINA Workflow](pina-quickstart.md#1-four-step-workflow)
- [Error Handling](integration-patterns.md#pattern-7-error-handling--validation)

---

## File Organization

```
refs/
├── README.md                   ← You are here
├── INDEX.md                    ← Navigation & overview
├── RESEARCH_SUMMARY.md         ← Research details
├── marimo-quickstart.md        ← Marimo reference
├── polars-quickstart.md        ← Polars reference
├── plotly-quickstart.md        ← Plotly reference
├── pina-quickstart.md          ← PINA reference
└── integration-patterns.md     ← Combined patterns
```

---

## Summary

This documentation library is your complete reference for:
- **marimo**: Building reactive Python notebooks
- **polars**: High-performance data operations
- **plotly**: Creating interactive visualizations
- **pina**: Machine learning and scientific computing

Each guide includes working code examples, best practices, and real-world patterns. Use the navigation links above to find exactly what you need.

**Happy coding!**
