# Agentic Hybrid HPC Scheduler with Ollama and SLURM
# Tools 1
# Author: Dr. Patrick Lemoine

# Features:
# - Reads the live SLURM queue
# - Inspects node and GPU topology
# - Optimizes a job list
# - Estimates energy impact
# - Optionally submits a SLURM job

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import requests
from ollama import chat

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"
DEFAULT_PARTITION = "gpu"
APP_DIR = Path.home() / ".agentic_hpc_scheduler"
APP_DIR.mkdir(parents=True, exist_ok=True)

ALLOW_SUBMIT = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("agentic_hpc_scheduler")


@dataclass
class Job:
    job_id: str
    gpus: int
    duration: int
    priority: int
    node: Optional[str] = None


def run_command(cmd: List[str], timeout: int = 20) -> dict:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": 124,
            "stdout": "",
            "stderr": f"Timeout running command: {' '.join(cmd)}",
        }


def check_slurm_json_support() -> bool:
    result = run_command(["squeue", "--json"], timeout=10)
    if result["returncode"] != 0:
        logger.error("SLURM JSON output is not available: %s", result["stderr"])
        return False
    try:
        json.loads(result["stdout"])
        return True
    except json.JSONDecodeError:
        logger.error("squeue --json returned invalid JSON.")
        return False


def ollama_is_available() -> bool:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False


def list_models() -> List[str]:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", []) if "name" in m]
    except Exception as exc:
        logger.error("Failed to fetch Ollama models: %s", exc)
        return []


def validate_ollama_and_model(model_name: str) -> bool:
    if not ollama_is_available():
        logger.error("Ollama is not reachable at %s", OLLAMA_BASE_URL)
        return False

    models = list_models()
    if model_name not in models:
        logger.error("Model %s is not available locally. Available models: %s", model_name, models)
        return False

    return True


def get_slurm_queue() -> str:
    result = run_command(["squeue", "--json"], timeout=10)
    if result["returncode"] != 0:
        return json.dumps({"error": result["stderr"]}, indent=2)

    try:
        data = json.loads(result["stdout"])
        jobs = data.get("jobs", [])
        running = [j for j in jobs if j.get("job_state") == "R"]
        pending = [j for j in jobs if j.get("job_state") == "PD"]

        return json.dumps(
            {
                "running_jobs": len(running),
                "pending_jobs": len(pending),
                "total_jobs": len(jobs),
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"error": f"Failed to parse squeue JSON: {exc}"}, indent=2)


def inspect_topology() -> str:
    hostname = run_command(["hostname"], timeout=5)
    gpu = run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw",
            "--format=csv,noheader",
        ],
        timeout=10,
    )
    cpu = run_command(["lscpu"], timeout=10)

    info = {
        "hostname": hostname["stdout"] if hostname["returncode"] == 0 else "unknown",
        "gpu_info": gpu["stdout"] if gpu["returncode"] == 0 else gpu["stderr"],
        "cpu_info": cpu["stdout"] if cpu["returncode"] == 0 else cpu["stderr"],
        "nccl_available": shutil.which("nccl-tests") is not None
        or os.path.exists("/usr/lib/x86_64-linux-gnu/libnccl.so")
        or os.path.exists("/usr/lib64/libnccl.so"),
        "rccl_available": os.path.exists("/opt/rocm/lib/librccl.so"),
    }
    return json.dumps(info, indent=2)


def optimize_jobs(jobs: List[Job]) -> str:
    ranked = sorted(
        jobs,
        key=lambda j: (j.priority * j.duration, j.gpus),
        reverse=True,
    )

    total_gpus = sum(j.gpus for j in ranked)
    total_duration = sum(j.duration for j in ranked)

    return json.dumps(
        {
            "optimized_order": [j.job_id for j in ranked],
            "total_gpus_required": total_gpus,
            "estimated_total_duration": total_duration,
            "jobs": [asdict(j) for j in ranked],
            "recommendation": "Schedule high-priority, long jobs first to reduce queue fragmentation.",
        },
        indent=2,
    )


def estimate_energy_efficiency() -> str:
    gpu_power = run_command(
        [
            "nvidia-smi",
            "--query-gpu=power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ],
        timeout=10,
    )

    return json.dumps(
        {
            "gpu_power": gpu_power["stdout"] if gpu_power["returncode"] == 0 else gpu_power["stderr"],
            "recommendation": [
                "Use GPU nodes for training only.",
                "Prefer lighter accelerators for preprocessing or inference if available.",
                "Track power draw in production runs.",
            ],
        },
        indent=2,
    )


def submit_slurm_job(job_script: str, partition: str = DEFAULT_PARTITION) -> str:
    if not ALLOW_SUBMIT:
        return json.dumps(
            {"error": "Job submission is disabled. Relaunch with --AllowSubmit to enable sbatch."},
            indent=2,
        )

    script_path = APP_DIR / "agentic_job.slurm"
    script_path.write_text(job_script, encoding="utf-8")

    result = run_command(["sbatch", "-p", partition, str(script_path)], timeout=30)

    try:
        script_path.unlink(missing_ok=True)
    except OSError:
        pass

    if result["returncode"] != 0:
        return json.dumps({"error": result["stderr"]}, indent=2)

    return json.dumps(
        {
            "status": "submitted",
            "message": result["stdout"],
        },
        indent=2,
    )


def build_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_slurm_queue",
                "description": "Get the live SLURM queue summary.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inspect_topology",
                "description": "Inspect node and GPU topology.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "optimize_jobs",
                "description": "Optimize a job list with a scheduling heuristic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "jobs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "job_id": {"type": "string"},
                                    "gpus": {"type": "integer"},
                                    "duration": {"type": "integer"},
                                    "priority": {"type": "integer"},
                                    "node": {"type": ["string", "null"]},
                                },
                                "required": ["job_id", "gpus", "duration", "priority"],
                            },
                        }
                    },
                    "required": ["jobs"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "estimate_energy_efficiency",
                "description": "Estimate energy efficiency from available node metrics.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_slurm_job",
                "description": "Submit a SLURM job script with sbatch.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_script": {"type": "string"},
                        "partition": {"type": "string"},
                    },
                    "required": ["job_script"],
                },
            },
        },
    ]


def execute_tool(name: str, args: dict) -> str:
    if name == "get_slurm_queue":
        return get_slurm_queue()
    if name == "inspect_topology":
        return inspect_topology()
    if name == "optimize_jobs":
        jobs = [Job(**j) for j in args["jobs"]]
        return optimize_jobs(jobs)
    if name == "estimate_energy_efficiency":
        return estimate_energy_efficiency()
    if name == "submit_slurm_job":
        return submit_slurm_job(args["job_script"], args.get("partition", DEFAULT_PARTITION))
    return json.dumps({"error": f"Unknown tool: {name}"}, indent=2)


def agentic_run(model_target: str, user_prompt: str, q_speech: bool = False, temperature: float = 0.0) -> None:
    tools = build_tools()
    messages = [
        {
            "role": "system",
            "content": (
                "You are an autonomous HPC scheduling assistant. "
                "Use tools when needed. Keep answers concise and operational."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    response = chat(
        model=model_target,
        messages=messages,
        tools=tools,
        options={"temperature": temperature},
    )

    message = response["message"]
    tool_calls = message.get("tool_calls", [])

    while tool_calls:
        messages.append(message)

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"].get("arguments", {})
            if isinstance(fn_args, str):
                fn_args = json.loads(fn_args)

            tool_result = execute_tool(fn_name, fn_args)
            messages.append(
                {
                    "role": "tool",
                    "name": fn_name,
                    "content": tool_result,
                }
            )

        response = chat(
            model=model_target,
            messages=messages,
            tools=tools,
            options={"temperature": temperature},
        )
        message = response["message"]
        tool_calls = message.get("tool_calls", [])

    print(message["content"])

def load_tasks_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--URL", type=str, default=OLLAMA_BASE_URL, help="Ollama base URL")
    parser.add_argument("--Speech", action="store_true", help="Enable speech output")
    parser.add_argument("--Temperature", type=float, default=0.0, help="Temperature between 0.0 and 1.0")
    parser.add_argument("--AllowSubmit", action="store_true", help="Allow sbatch submission")
    parser.add_argument("--FilePrompt", type=str, default=None, help="Path to the file containing tasks")
    args = parser.parse_args()

    global OLLAMA_BASE_URL
    global ALLOW_SUBMIT
    OLLAMA_BASE_URL = args.URL
    ALLOW_SUBMIT = args.AllowSubmit

    if not check_slurm_json_support():
        return

    if not validate_ollama_and_model(args.Model):
        return

    if args.FilePrompt:
        if os.path.isfile(args.FilePrompt) and os.path.getsize(args.FilePrompt) > 0:
            user_prompt = load_tasks_from_file(args.FilePrompt)
        else:
            print("Error: The file does not exist or is empty. Please provide a valid file path.")
            return
    else:
        user_prompt = """
Analyze my HPC cluster and propose a scheduling plan.

Tasks:
1. Read the live SLURM queue.
2. Inspect the node topology.
3. Optimize these jobs:
   - job_A: 4 GPUs, 300 seconds, priority 3
   - job_B: 8 GPUs, 150 seconds, priority 2
   - job_C: 2 GPUs, 600 seconds, priority 1
   - job_D: 6 GPUs, 200 seconds, priority 3
   - job_E: 4 GPUs, 400 seconds, priority 2
4. Estimate energy efficiency.
5. Give a final recommendation.
"""

    agentic_run(
        model_target=args.Model,
        user_prompt=user_prompt,
        q_speech=args.Speech,
        temperature=args.Temperature,
    )


if __name__ == "__main__":
    main()
    
