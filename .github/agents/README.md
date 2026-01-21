# marimo-flow Agent Architecture

Multi-agent system for AI-assisted development inspired by [Cursor's autonomous agent research](https://cursor.com/blog/agents).

## Architecture Principles

Based on Cursor's findings from building a web browser with 100s of autonomous agents:

1. **Clear role separation** - Planners, Workers, Judge (not flat hierarchy)
2. **Reduce complexity** - Remove bottlenecks, let workers self-coordinate
3. **Model-role matching** - Different models excel at different tasks
4. **Prompts > Harness** - Prompt engineering is more important than infrastructure
5. **Sub-specialization** - Planners can spawn sub-planners for specific domains

## Agent Roles

### 🎯 Planner Agent (Chief Architect)

**Responsibility**: Explore codebase, create high-level tasks, spawn sub-planners

**Model**: GPT-5.2 (better at planning than GPT-5.1-Codex)

**Skills**:
- Analyze requirements and break down into tasks
- Understand marimo-flow architecture
- Spawn sub-planners for specialized domains
- Maintain project vision and coherence

**Prompts**: See `planner-agent.md`

**When to use**:
- Starting new features
- Refactoring large code sections
- Architectural decisions

---

### 👷 Worker Agents (Specialists)

**Responsibility**: Take tasks, execute them, push results (no big picture thinking)

**Model**: GPT-5.1-Codex or Opus 4.5 (task-dependent)

**Types**:

#### 1. Notebook Worker
- Creates/modifies marimo notebooks (`.py` files)
- Ensures reactivity and idempotence
- Implements UI elements (sliders, forms, etc.)

#### 2. MLflow Worker
- Implements experiment tracking
- Manages model registry
- Handles artifact logging

#### 3. PINA Worker
- Implements Physics-Informed Neural Networks
- Solves PDEs with neural networks
- Integrates with marimo reactivity

#### 4. Data Worker
- Works with Polars/Pandas DataFrames
- Implements DuckDB queries
- Creates data visualizations (Altair/Plotly)

#### 5. Testing Worker
- Writes pytest tests
- Ensures code coverage
- Validates reactive behavior

**Prompts**: See `worker-*.md` files

**When to use**:
- Executing specific tasks from Planner
- Focused implementation work
- Conflict resolution (workers handle this better than integrators)

---

### ⚖️ Judge Agent (Quality Control)

**Responsibility**: Evaluate work, decide if project is complete or needs another iteration

**Model**: GPT-5.2 (better at following instructions and staying focused)

**Skills**:
- Verify task completion
- Check code quality and reactivity
- Decide: ship, iterate, or escalate
- No implementation - only judgment

**Prompts**: See `judge-agent.md`

**When to use**:
- End of each work cycle
- Before merging PRs
- Quality gates

---

## Agent Workflows

### Feature Development

```
1. Planner Agent
   ├─ Analyzes requirements
   ├─ Creates task list
   └─ Spawns Sub-Planners if needed
       ├─ Notebook Sub-Planner (for complex notebook logic)
       ├─ MLflow Sub-Planner (for experiment tracking)
       └─ PINA Sub-Planner (for PDE solving)

2. Worker Agents (parallel execution)
   ├─ Take tasks from queue
   ├─ Execute independently
   ├─ Push results
   └─ Self-coordinate on conflicts

3. Judge Agent
   ├─ Reviews all changes
   ├─ Checks quality metrics
   └─ Decides: ✓ Done | ↻ Iterate | ⚠ Escalate
```

### Bug Fix

```
1. Planner Agent
   ├─ Diagnoses issue
   └─ Creates focused task

2. Relevant Worker (direct assignment)
   ├─ Fixes bug
   └─ Adds test

3. Judge Agent
   ├─ Verifies fix
   └─ Approves merge
```

---

## Anti-Patterns (Learned from Cursor)

### ❌ Don't: Flat hierarchy with locks
**Problem**: Agents wait for locks, avoid hard tasks, no accountability

### ❌ Don't: Integrator role
**Problem**: Creates bottlenecks, workers handle conflicts better themselves

### ❌ Don't: Same model for all roles
**Problem**: Models have different strengths (planning vs. coding vs. judging)

### ❌ Don't: Add complexity
**Problem**: More roles/steps = more failure points

---

## Model Selection Guide

| Role | Recommended Model | Why |
|------|-------------------|-----|
| **Planner** | GPT-5.2 | Better at planning, maintaining focus, avoiding drift |
| **Notebook Worker** | GPT-5.1-Codex | Specialized for code generation |
| **MLflow Worker** | Opus 4.5 | Good at structured API calls |
| **PINA Worker** | GPT-5.1-Codex | Math-heavy code generation |
| **Data Worker** | Opus 4.5 | Excellent at data transformations |
| **Testing Worker** | GPT-5.1-Codex | Code-focused |
| **Judge** | GPT-5.2 | Best at following instructions, staying focused |

**Note**: Opus 4.5 tends to finish faster (gives back control earlier), GPT-5.2 holds focus longer for autonomous work.

---

## Prompt Engineering Guidelines

> "A surprisingly large part of system behavior depends on how we prompt the agents. The harness and models matter, but prompts matter more." - Cursor

### Key Principles

1. **Be specific about role boundaries**
   - Planners: Don't implement, only plan
   - Workers: Don't plan, only execute
   - Judge: Don't fix, only evaluate

2. **Include marimo-flow context**
   - Reactivity requirements
   - MLflow tracking patterns
   - PINA integration patterns

3. **Provide examples**
   - Good: Idempotent cells
   - Bad: Hidden state

4. **Set clear completion criteria**
   - Workers: Task is done when...
   - Judge: Ship if all criteria met

5. **Encourage responsibility**
   - Workers: Take ownership of hard problems
   - Don't avoid difficult tasks
   - Implement end-to-end solutions

---

## Usage

### GitHub Copilot
Reference agent in comments:
```python
# @agent planner: Design experiment tracking for this notebook
# @agent notebook-worker: Implement reactive UI for hyperparameters
# @agent judge: Review this cell for reactivity issues
```

### Claude Code (via MCP)
Agents are available via custom instructions in `.github/workflows/claude-code.yml`

### Cursor
Use `.cursorrules` which references this agent architecture

---

## Configuration

- **GitHub Copilot**: `.vscode/mcp.json` + `.github/copilot-instructions.md`
- **Claude Code**: `.github/workflows/claude-code.yml` (custom_instructions)
- **Cursor**: `.cursorrules` (references agent patterns)
- **Claude Desktop**: `~/.config/Claude/claude_desktop_config.json`

---

## Metrics

Track agent effectiveness:
- **Planner**: Tasks created vs. tasks completed
- **Workers**: Completion rate, conflict resolution time
- **Judge**: False positive/negative rate on quality checks

---

## Evolution

This architecture will evolve based on:
- Real-world usage in marimo-flow
- Model improvements (GPT-5.x, Opus 5.x)
- User feedback
- Agent performance metrics

**Last updated**: 2026-01-21
**Based on**: [Cursor Agent Research](https://cursor.com/blog/agents)
