# **Agentic Hybrid HPC Scheduler**

[![Version](https://img.shields.io/badge/version-1.0-green.svg)](https://github.com/lemoinep/Agentic-HPC-Scheduler)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

---

<p align="center">
<img src="Images/P0002.jpg" width="100%" />
</p>

# **Introduction**

**Agentic Hybrid HPC Scheduler with Ollama and SLURM** is a tool that combines local LLM inference with SLURM-aware cluster management to support HPC scheduling workflows. It reads the live SLURM queue, inspects node and GPU topology, ranks jobs with a simple scheduling heuristic, estimates energy-related signals, and can optionally submit jobs through `sbatch` when submission is enabled. The project is designed to help explore **agentic** decision-making in HPC environments while keeping execution practical and operational through local tools and deterministic system commands.


...


```mermaid
flowchart TD
    %% Nodes
    A[Start script - AgenticHybridHPCScheduler.py] --> B[Parse CLI args: Model, URL, AllowSubmit, FilePrompt]
    B --> C[Set Ollama URL and ALLOW_SUBMIT]
    C --> D[Check SLURM squeue JSON support]
    D -->|Failure| Z[Exit]
    D --> E[Check Ollama reachable and model available]
    E -->|Failure| Z

    E --> F[Build user prompt from file or default]
    F --> G[Build tools: get_slurm_queue, inspect_topology, optimize_jobs, estimate_energy_efficiency, submit_slurm_job]
    G --> H[Call Ollama chat with system and user messages and tools]

    H --> I{Tool calls requested?}
    I -->|No| J[Print final LLM answer with scheduling recommendation]
    I -->|Yes| K[Execute each tool via execute_tool]

    K --> L[Collect tool results as JSON]
    L --> M[Append tool results as tool messages]
    M --> H

    %% Class definitions
    classDef startNode fill:#6AA84F,stroke:#333,stroke-width:1px,color:#ffffff;
    classDef checkNode fill:#FFD966,stroke:#333,stroke-width:1px,color:#000000;
    classDef toolNode fill:#9FC5E8,stroke:#333,stroke-width:1px,color:#000000;
    classDef loopNode fill:#CFE2F3,stroke:#333,stroke-width:1px,color:#000000;
    classDef endNode fill:#CC0000,stroke:#333,stroke-width:1px,color:#ffffff;

    %% Class assignments
    class A startNode;
    class B,C,D,E checkNode;
    class F,G toolNode;
    class H,I,K,L,M loopNode;
    class J,Z endNode;
```

*The scheduler starts by validating SLURM JSON output and the local Ollama model, then builds a user prompt from a file or a default template. It calls Ollama with a set of HPC tools (SLURM queue, topology, job optimization, energy estimation, and optional submission), loops over tool calls, and finally prints an agentic scheduling recommendation to stdout.*

---

## Command line usage

```bash
python AgenticHybridHPCScheduler.py 
  --Model model 
  --URL http://localhost:11434 
  --Temperature 0.0 
  --AllowSubmit 
  --FilePrompt tasks.md
```

- `--Model
Name of the local Ollama model to use (default: llama3.1).

- `--URL` – Base URL of the Ollama server (default: http://localhost:11434).
- `--Speech` – Enable speech-related behavior in the agent (currently a placeholder flag).
- `--Temperature` – Sampling temperature between 0.0 and 1.0 for the LLM (default: 0.0).
- `--AllowSubmit` – If set, allow the agent to actually submit jobs with sbatch; otherwise, submit_slurm_job returns an error and does not touch SLURM.
- `--FilePrompt` – Path to a text/Markdown file containing the user tasks and instructions for the agent; if omitted, a built‑in default prompt about queue analysis and job optimization is used.

---

## 📝 **Author**

**Dr. Patrick Lemoine**  
*Engineer Expert in Scientific Computing*  
[LinkedIn](https://www.linkedin.com/in/patrick-lemoine-7ba11b72/)

---















"# Agentic-HPC-Scheduler" 
