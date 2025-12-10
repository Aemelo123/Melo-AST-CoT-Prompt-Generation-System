# Melo-AST-CoT-Guided-Prompting-Experiment

A research project implementing an **AST (Abstract Syntax Tree) Chain-of-Thought Guided Prompting approach** for secure code generation using Large Language Models.

## Overview

This project compares two code generation strategies:

- **AST_COT**: Generates code via AST JSON representation with structured reasoning and built-in security validation
- **NL_COT**: Natural Language Chain-of-Thought baseline using "Let's think step by step" prompting

Both approaches are evaluated on GPT-4 and Claude Sonnet 4.5 using tasks from the [SecurityEval](https://huggingface.co/datasets/s2e-lab/SecurityEval) dataset.

## Requirements

- Python 3.10+
- OpenAI API key (for GPT-4)
- Anthropic API key (for Claude)

## Quick Start

### 1. Create Virtual Environment

Open a terminal in the project folder and run:

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install All Dependencies

```bash
pip install -e .
```

This single command installs all required packages:
- `datasets` - For loading SecurityEval from HuggingFace
- `openai` - GPT-4 API client
- `anthropic` - Claude API client
- `ast2json` - AST to JSON conversion
- `bandit` - Python security linter
- `semgrep` - Static analysis tool

### 3. Set API Keys

```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

On Windows:
```cmd
set OPENAI_API_KEY=your-openai-api-key
set ANTHROPIC_API_KEY=your-anthropic-api-key
```

### 4. Run an Experiment

```bash
python -m melo_ast_cot.main 1
```

## Usage

### Running Experiments

The project runs 5 iterations, each processing 25 random tasks:

```bash
# Run a specific iteration (1-5)
python -m melo_ast_cot.main 1
python -m melo_ast_cot.main 2
# ... continue through iteration 5
```

Each iteration:
1. Loads 25 random tasks from SecurityEval
2. Generates code using both AST_COT and NL_COT conditions
3. Tests with both GPT-4 and Claude
4. Scans output with Bandit and Semgrep
5. Exports results to CSV

### Output Structure

```
results/
├── iteration_1/
│   ├── GPT4/
│   │   ├── AST_COT/
│   │   │   └── {task_id}_GPT4_AST_COT_1.json
│   │   └── NL_COT/
│   │       └── {task_id}_GPT4_NL_COT_1.json
│   └── CLAUDE/
│       ├── AST_COT/
│       └── NL_COT/
└── csv_exports/
    └── experiment_results_iteration_1.csv
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4 |
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key for Claude |

### LLM Parameters

Default settings in `llm_client.py`:
- Temperature: 0.7
- Max tokens: 2048
- Top-p: 0.95

## Results

Results are exported to CSV with the following metrics:

| Metric | Description |
|--------|-------------|
| `num_vulnerabilities` | Count of security issues found |
| `loc` | Lines of code generated |
| `vulnerability_density` | Vulnerabilities per 1000 LOC |
| `security_violations_count` | Internal validation violations |
| `success` | Whether generation completed |

## Research Background

This project is based on:

- **Li et al. (2023)** - Structured Chain-of-Thought Prompting for Code Generation ([arXiv:2305.06599](https://arxiv.org/abs/2305.06599))
- **Wei et al. (2022)** - Chain-of-Thought Prompting Elicits Reasoning in Large Language Models ([arXiv:2201.11903](https://arxiv.org/abs/2201.11903))
- **Kojima et al. (2022)** - Large Language Models are Zero-Shot Reasoners ([arXiv:2205.11916](https://arxiv.org/abs/2205.11916))
- **SecurityEval Dataset** - [s2e-lab/SecurityEval](https://huggingface.co/datasets/s2e-lab/SecurityEval)

