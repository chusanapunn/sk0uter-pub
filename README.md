# Sk0uter: Project Context Prompt Brewery

![Brewery](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbTl3M3pjYjRvbjZnNGFiejQwcDhoemt4YzYxd3NxaDg0ZGZmdWZuNCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/wNDa1OZtvl6Fi/giphy.gif)

## Distilled Prompt generation on local model using LightRAG

Standard AI tools treat your codebase like a pile of your weeks old laundry. They dig through the pile, find the **cheese touched sock** (a code snippet), and try to guess what the rest of that **cheese touched skins looks like.** They hit the **RAG Wall**—a point where context windows flood with irrelevant noise, and the AI starts hallucinating connections that don't exist.

Sk0uter is a **Prompt Brewery** built contribute by **Claude Code**. We don't just "retrieve" data; we distill it. By fusing **Hybrid Search**, with **Relational Topology**, we generate "Distilled Project Context" that allows even local 8B models to reason like Senior Architects. ( just for a few questions on local model for now...)

Sk0uter now work especially on **Godot4.4**. Indexed the Godot Project to VectorDB(Qdrant) and GraphRAG(Kuzu), then perform Hybrid Search and Graph Retrieval. SymCode context Compression to generate best prompt for Local/Cloud AI project development.

---
## Quick Start
```
0. Ensure you have Ollama installed
1. Download the release file, Extract and Run Start_Scout.bat # on Windows
```
---

# WHY A "BREWERY"?
I didn't build this to be another "AI Chat" clone. I built it because I needed a refinery.

Most AI tools give you "raw" output. Sk0uter gives you a Distilled Prompt. It’s like a brewery: you put in the raw complexity of a growing Godot project, and Sk0uter outputs a smooth, high-proof Concentrated prompt. You can then use that prompt locally or hand it off to a cloud giant like Claude Sonnet.

---

## THE PROBLEM: The "Huge Context" Trap

20-trillion-token windows? Maybe one day. But right now, dumping a raw repository into a chat box is a recipe for disaster.

* **Claude’s "Project Blindness":** Even the most agentic models can’t "see" what isn't in the immediate prompt. Without a Graph-RAG map, Claude doesn't know that refactoring `Player.gd` will silently annihilate a signal listener buried three folders deep in a UI scene.
* **Gemini’s "Middle-Loss":** Huge context windows suffer from a mathematical phenomenon where the AI prioritizes the start and end of a prompt. The critical "glue code" in the middle of your 100-file dump gets ignored, leading to broken logic and session memory resets.
* **The Hallucination Wall:** Without explicit architecture (who owns what, who fires which signal), AI interprets code as a flat list of text. Sk0uter adds the **Relational Layer** required for the AI to actually understand the "Game Loop."

---

## THE LIGHT-RAG ARCHITECTURE

Sk0uter is built on the principles of **LightRAG** that solves the "Incomplete Context" problem.

### How it Works:
* **Global Retrieval (The Graph):** When you ask a broad question ("How does my combat system work?"), the Graph provides the high-level summary of relationships across multiple scripts.
* **Local Retrieval (The Vector):** When you ask a specific question ("Why is my jump height wrong?"), the Vector DB grabs the specific line of code.
* **The Synergy:** Sk0uter performs a "Multi-Hop" traversal. It finds the target code via Vectors, then "hops" through the Graph to see every other script that interacts with it. You get the **Surgical Implementation** and the **Architectural Impact** in one prompt.

---

## PRODUCT DIFFERENTIATION

| Feature | Standard RAG / Wrappers | **Sk0uter (The Brewery)** |
| :--- | :--- | :--- |
| **Logic** | Simple keyword lookup | **Topology-Aware GraphRAG** |
| **Context** | Full-file dumps (VRAM killer) | **Dynamic Context Compression** |
| **Search** | Flat Vector Search | **Dual-Engine (Hybrid Search + Relational)** |
| **Philosophy** | "Talk to your code" | **"Brew the perfect Handshake Prompt"** |
| **Privacy** | Usually sends data to Cloud | **100% Local Sovereignty** |

---
## CONFIGURATION

### Qdrant Vector Database

By default, Sk0uter expects Qdrant to be running **locally** on `localhost:6333`.

If your Qdrant instance is on a **different machine** (e.g. a NAS, home server, or cloud VM), or on a **non-default port**, edit `config.py`:

```python
# config.py
QDRANT_HOST = "192.168.1.44"   # Replace with your Qdrant server's IP
QDRANT_PORT = 6333             # Replace with the port Qdrant is listening on
```

To run Qdrant locally via Docker:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Ollama LLM Engine

Ollama is expected at `http://localhost:11434` by default. If you're running Ollama on a remote machine, update `OLLAMA_URL` in `config.py`:

```python
OLLAMA_URL = "http://192.168.1.44:11434"
```

---

## CURRENT SYSTEM STATE

Sk0uter is currently operational with:
* **Hybrid Vector Indexing:** Every chunk of your code is mathematically mapped for semantic and lexical retrieval.
* **Relational Graphing:** Your Godot signals, script inheritance, and scene hierarchies are mapped into a high-performance relational database.
* **Surgical Prompting:** Our "Brewery" distills these two data sources into a single structured payload for local (Ollama) or Cloud (Claude/Gemini) execution.

## Patch note v0.9 
Current state still mostly works on Cloud model, local 7b will lose context FAST, however, there are still many gaps to be fix.
And if it starts to work on small model, It's likely going to be significantly better on Cloud model as well.

- improve graph map relations layers, node details, ui,
- add Qdrant Local mode/sever toggle
- add prompt evaluation
- Hybrid search (BM25+dense) on vector db with RRF
- Graph Map real Three-tier context utilization + SymCode compression


## Plan to improve next:

- Change stack from streamlit UI to fastAPI + Golang Wails ( current streamlit is at UX limitation)
- DSPy on prompt optimization
- HyDE for vague queries
- Adaptive token budgeting to fix silent truncation
- Query Intent Router
- Ollama Reranker
- Prompt generation/Surgery Tab to focus on one prompt unlike chat memory manner
- Context distillation for further compression
- Agentic
---
*Architected by qua.liap*

> **WARNING:** > ![Do not develop my app](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExc2ExOWdmcHBwc2gydmMyOGN5ODFrOTdyamJiOXZsbHM1OGVxYTh0dSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L3bj6t3opdeNddYCyl/giphy.gif)
> *Seriously, don't develop my app. But if you want to fund the Brewery...*

If Sk0uter saved your project from a spaghetti-code meltdown, consider throwing some caffeine money my way. It takes a lot of late nights to wrangle these local models.

why name sk0uter? it kinda scout your project, to create knowledge context... and 0 is cool

☕ **[Buy me a Coffee on Ko-fi](https://ko-fi.com/qualiap/goal?g=0)**
