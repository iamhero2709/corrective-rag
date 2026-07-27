"""
Corrective RAG — Professional CLI
"""
import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.markdown import Markdown
from rich.columns import Columns
from rich import box

console = Console()

LOGO = """[bold violet]
   ____                _           ____   ____ ______
  / ___|___  _ __ ___ (_)_ __     |  _ \\ / ___/ ___|
 | |   / _ \\| '_ ` _ \\| | '_ \\   | |_) | |  | |
 | |__| (_) | | | | | | | | | |  |  _ <| |__| |___
  \\____\\___/|_| |_| |_|_|_| |_|  |_| \\_\\____\\____|
[/]"""

TAGLINE = "[dim]Self-Corrective RAG with HRR + NLI + Embedding Verification[/]"

BANNER = f"{LOGO}\n{TAGLINE}"


def _banner():
    console.print(BANNER)
    console.print()


def _resolve_device(device: str) -> str:
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _build_rag(args):
    from src.config import Settings
    from src.retriever import DenseRetriever
    from src.verifier import RetrievalVerifier
    from src.generator import SmallModelGenerator
    from src.pipeline import CorrectiveRAG

    config = Settings().validate()
    device = _resolve_device(config.device)
    use_hybrid = getattr(args, "hybrid", config.use_hybrid)

    with console.status("[bold violet]Loading retriever...[/]", spinner="dots"):
        retriever = DenseRetriever(config.embed_model, device=device, use_hybrid=use_hybrid)
        if Path(config.index_dir).exists():
            retriever.load(config.index_dir)

    with console.status("[bold violet]Loading verifier (NLI)...[/]", spinner="dots"):
        verifier = RetrievalVerifier(
            hrr_dim=config.hrr_dim, use_structural=config.use_structural,
            use_entailment=config.use_entailment, nli_model=config.nli_model,
            device=device, thresholds=(config.t_correct, config.t_incorrect),
            weights=(config.w_embedding, config.w_structural, config.w_entailment),
        )

    gen_model = getattr(args, "model", "") or config.gen_model
    quantize = getattr(args, "no_quant", False) is not True

    with console.status(f"[bold violet]Loading generator [cyan]{gen_model}[/]...[/]", spinner="dots"):
        generator = SmallModelGenerator(gen_model, device=device,
                                         max_new_tokens=config.max_new_tokens, quantize=quantize)

    rag = CorrectiveRAG(retriever, verifier, generator, top_k=config.top_k,
                        min_correct=config.min_correct, max_retries=config.max_retries)
    return rag, config


DEMO_DOCS = {
    "nolan": "Christopher Nolan directed Inception, released in 2010. The film stars "
             "Leonardo DiCaprio as Dom Cobb, a thief who steals corporate secrets "
             "through dream-sharing technology. It won four Academy Awards.",
    "dreams": "Dreams have fascinated humanity for millennia. Lucid dreaming is the "
              "practice of becoming aware within a dream. Many films explore dream "
              "themes and the subconscious mind in visually striking ways.",
    "dicaprio": "Leonardo DiCaprio won his first Academy Award for Best Actor for "
                "The Revenant in 2016, directed by Alejandro Inarritu.",
}


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_query(args):
    _banner()
    rag, config = _build_rag(args)

    if not rag.retriever.chunks:
        console.print("[yellow]No index found — using demo corpus[/]")
        rag.retriever.add_documents(DEMO_DOCS)

    console.print(f"\n[bold]Question:[/] [cyan]{args.question}[/]\n")

    with console.status("[bold violet]Thinking...[/]", spinner="dots"):
        t0 = time.time()
        result = rag.run(args.question)
        elapsed = time.time() - t0

    conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red", "ABSTAIN": "dim"}
    c = conf_color.get(result.confidence, "white")

    answer_panel = Panel(
        Markdown(result.answer) if result.answer else "[dim]No answer generated[/]",
        title="[bold]Answer[/]",
        border_style=c,
        padding=(1, 2),
    )
    console.print(answer_panel)

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Confidence", f"[{c}]{result.confidence}[/]")
    info.add_row("Latency", f"{elapsed:.1f}s")
    info.add_row("Chunks used", str(len(result.used_chunks)))
    info.add_row("Mode", getattr(args, "mode", "corrective"))
    console.print(info)

    if result.citations:
        console.print()
        cite_table = Table(title="Sources", box=box.ROUNDED, show_header=True)
        cite_table.add_column("#", style="dim", width=3)
        cite_table.add_column("Document")
        cite_table.add_column("Score", justify="right")
        cite_table.add_column("Snippet", style="dim")
        for i, c in enumerate(result.citations, 1):
            cite_table.add_row(str(i), c.doc_id, f"{c.score:.2f}", c.text[:80] + "...")
        console.print(cite_table)

    if args.verbose:
        console.print()
        trace_table = Table(title="Decision Trace", box=box.ROUNDED, show_lines=True)
        trace_table.add_column("Step", style="bold violet", width=10)
        trace_table.add_column("Detail", style="dim")
        for step in result.trace.steps:
            step_name = step.pop("step", "?")
            detail = "  ".join(f"{k}={v}" for k, v in step.items())
            trace_table.add_row(step_name, detail)
        console.print(trace_table)


def cmd_serve(args):
    _banner()
    import uvicorn
    port = args.port
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    info = Table(show_header=False, box=None)
    info.add_column(style="bold")
    info.add_column()
    info.add_row("URL", f"[bold cyan]http://localhost:{port}[/]")
    info.add_row("Mode", "Auto-reload" if args.reload else "Production")
    info.add_row("API Docs", f"[dim]http://localhost:{port}/docs[/]")

    console.print(Panel(info, title="[bold violet]Starting Server[/]", border_style="violet", padding=(1, 2)))
    console.print()

    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=args.reload, log_level="info")


def cmd_index(args):
    _banner()
    from src.config import Settings
    from src.retriever import DenseRetriever
    from src.multi_loader import load_file, load_directory, chunk_documents

    config = Settings().validate()

    src_path = Path(args.source)
    is_dir = src_path.is_dir()

    with console.status(f"[bold violet]{'Scanning directory' if is_dir else 'Loading file'}...[/]", spinner="dots"):
        if is_dir:
            loaded = load_directory(str(src_path))
        else:
            loaded = load_file(str(src_path))

    if not loaded:
        console.print("[red]No documents found or unsupported format[/]")
        sys.exit(1)

    with console.status("[bold violet]Chunking & embedding...[/]", spinner="dots"):
        docs = chunk_documents(loaded)
        if not docs:
            console.print("[red]No chunks produced[/]")
            sys.exit(1)
        retriever = DenseRetriever(config.embed_model, device=config.device, use_hybrid=args.hybrid)
        retriever.add_documents(docs)
        retriever.save(config.index_dir)

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Source", str(src_path))
    info.add_row("Pages/Sections", str(len(loaded)))
    info.add_row("Chunks", str(len(docs)))
    info.add_row("Total indexed", str(len(retriever.chunks)))
    info.add_row("Saved to", config.index_dir)
    console.print(Panel(info, title="[bold green]Indexed Successfully[/]", border_style="green", padding=(1, 2)))


def cmd_index_url(args):
    _banner()
    from src.config import Settings
    from src.retriever import DenseRetriever
    from src.multi_loader import load_url, chunk_documents

    config = Settings().validate()

    with console.status(f"[bold violet]Fetching URL...[/]", spinner="dots"):
        docs = load_url(args.url)

    if not docs:
        console.print("[red]Could not extract text from URL[/]")
        sys.exit(1)

    with console.status("[bold violet]Chunking & embedding...[/]", spinner="dots"):
        chunked = chunk_documents(docs)
        retriever = DenseRetriever(config.embed_model, device=config.device, use_hybrid=args.hybrid)
        retriever.add_documents(chunked)
        retriever.save(config.index_dir)

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("URL", args.url)
    info.add_row("Chunks", str(len(chunked)))
    info.add_row("Total indexed", str(len(retriever.chunks)))
    console.print(Panel(info, title="[bold green]URL Indexed[/]", border_style="green", padding=(1, 2)))


def cmd_demo(args):
    _banner()
    rag, _ = _build_rag(args)
    rag.retriever.add_documents(DEMO_DOCS)

    queries = [
        "Who directed Inception?",
        "What did DiCaprio win an Oscar for?",
        "What is the box office revenue of Interstellar?",
    ]

    for q in queries:
        console.rule(f"[bold cyan]{q}[/]")
        t0 = time.time()
        result = rag.run(q)
        elapsed = time.time() - t0

        conf_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red", "ABSTAIN": "dim"}
        c = conf_color.get(result.confidence, "white")

        console.print(f"\n[bold]{result.answer}[/]")
        console.print(f"  [{c}]{result.confidence}[/]  [dim]{elapsed:.1f}s[/]")
        if args.verbose:
            for step in result.trace.steps:
                console.print(f"    [dim]{step}[/]")
        console.print()


def cmd_benchmark(args):
    _banner()
    from src.benchmarks import HotpotQAEvaluator, ASQAEvaluator, PopQAEvaluator, AblationStudy
    from src.persistence import DocumentStore

    rag, config = _build_rag(args)
    db = DocumentStore("data/benchmark.db")
    dataset = args.dataset
    samples = args.samples

    info = Table(show_header=False, box=None, padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Dataset", f"[cyan]{dataset}[/]")
    info.add_row("Samples", str(samples))
    info.add_row("Ablation", "Yes (A0-A4)" if args.ablation else "No")
    console.print(Panel(info, title="[bold violet]Benchmark Configuration[/]", border_style="violet", padding=(1, 2)))
    console.print()

    if args.ablation:
        with console.status("[bold violet]Running ablation study...[/]", spinner="dots"):
            study = AblationStudy(dataset, samples)
            results = study.run(rag, db)

        console.print()
        study.print_table()
    else:
        evaluators = {
            "hotpot_qa": ("HotpotQA", HotpotQAEvaluator),
            "asqa": ("ASQA", ASQAEvaluator),
            "popqa": ("PopQA", PopQAEvaluator),
        }

        if dataset not in evaluators:
            console.print(f"[red]Unknown dataset: {dataset}[/]")
            console.print(f"[dim]Available: {', '.join(evaluators.keys())}[/]")
            sys.exit(1)

        name, EvaluatorCls = evaluators[dataset]
        evaluator = EvaluatorCls(rag, db)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"[violet]Evaluating {name}...[/]", total=samples)
            result = evaluator.evaluate(samples)
            progress.update(task, completed=samples)

        _print_benchmark_result(result)


def _print_benchmark_result(result):
    table = Table(title="Results", box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Dataset", result.dataset)
    table.add_row("Samples", str(result.num_samples))
    table.add_row("Exact Match", f"[green]{result.exact_match:.1f}%[/]")
    table.add_row("F1 Score", f"[cyan]{result.f1_score:.3f}[/]")
    table.add_row("Latency (mean)", f"{result.latency_mean:.0f}ms")
    table.add_row("Latency (p95)", f"{result.latency_p95:.0f}ms")
    table.add_row("Config", result.config)

    if result.confidence_distribution:
        dist_str = "  ".join(f"[dim]{k}:[/] {v}" for k, v in result.confidence_distribution.items())
        table.add_row("Confidence", dist_str)

    console.print()
    console.print(table)


def cmd_download_models(args):
    _banner()
    from huggingface_hub import snapshot_download
    import os

    models = {
        "embed": ("BAAI/bge-small-en-v1.5", "Embedding model"),
        "nli": ("cross-encoder/nli-deberta-v3-xsmall", "NLI cross-encoder"),
        "gen": ("Qwen/Qwen2.5-0.5B-Instruct", "Generator (CPU)"),
        "gen-gpu": ("Qwen/Qwen2.5-1.5B-Instruct", "Generator (GPU)"),
    }
    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    selected = [args.model] if args.model and args.model != "all" else list(models.keys())

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    table.add_column("Model", style="bold")
    table.add_column("Description")
    table.add_column("Status", justify="center")
    for key in selected:
        entry = models.get(key)
        if not entry:
            console.print(f"[red]Unknown model: {key}[/]")
            console.print(f"[dim]Choices: all, {', '.join(models.keys())}[/]")
            sys.exit(1)
        repo, desc = entry
        with console.status(f"[violet]Downloading {desc}...[/]", spinner="dots"):
            snapshot_download(repo_id=repo, cache_dir=cache)
        table.add_row(key, desc, "[green]OK[/]")

    console.print()
    console.print(table)
    console.print(f"\n[dim]Cache: {cache}[/]")


def cmd_list_models(args):
    _banner()
    table = Table(title="Available Models", box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Key", style="bold cyan")
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Use", style="dim")

    table.add_section()
    table.add_row("embed", "BAAI/bge-small-en-v1.5", "33M", "Embeddings")
    table.add_row("nli", "cross-encoder/nli-deberta-v3-xsmall", "110M", "NLI verification")
    table.add_row("gen", "Qwen/Qwen2.5-0.5B-Instruct", "0.5B", "Generator (CPU)")
    table.add_row("gen-gpu", "Qwen/Qwen2.5-1.5B-Instruct", "1.5B", "Generator (GPU)")
    table.add_row("llama-1b", "meta-llama/Llama-3.2-1B-Instruct", "1B", "Generator (GPU)")
    table.add_row("llama-3b", "meta-llama/Llama-3.2-3B-Instruct", "3B", "Generator (GPU)")
    table.add_row("gemma-2b", "google/gemma-2-2b-it", "2B", "Generator (GPU)")

    console.print(table)
    console.print("\n[dim]Set model via: RAG_GEN_MODEL=<model> or --model <key>[/]")


def cmd_status(args):
    _banner()
    from src.config import Settings

    config = Settings().validate()
    index_path = Path(config.index_dir)
    has_index = (index_path / "index.faiss").exists()

    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column(style="bold", width=18)
    table.add_column()
    table.add_row("Device", f"[cyan]{config.device}[/]")
    table.add_row("Embed model", config.embed_model)
    table.add_row("Gen model", config.gen_model)
    table.add_row("NLI model", config.nli_model)
    table.add_row("HRR dim", str(config.hrr_dim))
    table.add_row("Index dir", config.index_dir)
    table.add_row("Index status", "[green]Found[/]" if has_index else "[yellow]Not built[/]")
    table.add_row("Hybrid search", "Enabled" if config.use_hybrid else "Disabled")
    table.add_row("Structural", "On" if config.use_structural else "Off")
    table.add_row("Entailment", "On" if config.use_entailment else "Off")

    if has_index:
        chunks_file = index_path / "chunks.jsonl"
        if chunks_file.exists():
            n = sum(1 for _ in chunks_file.open())
            table.add_row("Indexed chunks", f"[cyan]{n}[/]")

    console.print(Panel(table, title="[bold violet]System Status[/]", border_style="violet", padding=(1, 2)))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser(
        prog="rag",
        description="[bold violet]Corrective RAG Toolkit[/] — HRR + NLI + Embedding verification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="[dim]Docs: https://github.com/yourname/corrective-rag[/]",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # query
    q = sub.add_parser("query", help="Ask a question")
    q.add_argument("question", help="Question to answer")
    q.add_argument("--verbose", "-v", action="store_true", help="Show trace steps")
    q.add_argument("--model", "-m", default="", help="Generator model override")
    q.add_argument("--hybrid", action="store_true", help="Enable BM25 hybrid search")
    q.add_argument("--no-quant", action="store_true", help="Disable quantization")

    # serve
    s = sub.add_parser("serve", help="Start API server with Web UI")
    s.add_argument("--port", "-p", type=int, default=8000, help="Port (default: 8000)")
    s.add_argument("--reload", action="store_true", help="Auto-reload on changes")

    # index
    idx = sub.add_parser("index", help="Index documents")
    idx.add_argument("source", help="File or directory path")
    idx.add_argument("--hybrid", action="store_true", help="Build BM25 index")

    # index-url
    url_cmd = sub.add_parser("index-url", help="Index a URL")
    url_cmd.add_argument("url", help="URL to index")
    url_cmd.add_argument("--hybrid", action="store_true", help="Build BM25 index")

    # demo
    d = sub.add_parser("demo", help="Run built-in demo")
    d.add_argument("--verbose", "-v", action="store_true", help="Show trace steps")
    d.add_argument("--model", "-m", default="", help="Generator model override")
    d.add_argument("--no-quant", action="store_true", help="Disable quantization")

    # benchmark
    b = sub.add_parser("benchmark", help="Run benchmark evaluation")
    b.add_argument("--dataset", "-d", default="hotpot_qa", choices=["hotpot_qa", "asqa", "popqa"])
    b.add_argument("--samples", "-n", type=int, default=50, help="Number of samples")
    b.add_argument("--ablation", action="store_true", help="Run ablation study (A0-A4)")
    b.add_argument("--model", "-m", default="", help="Generator model override")

    # download-models
    dl = sub.add_parser("download-models", help="Download ML models")
    dl.add_argument("model", nargs="?", default="all",
                    help="all, embed, nli, gen, gen-gpu")

    # list-models
    sub.add_parser("list-models", help="List available models")

    # status
    sub.add_parser("status", help="Show system configuration")

    args = parser.parse_args()
    if not args.command:
        _banner()
        console.print("[dim]Usage:[/] rag [bold]<command>[/] [dim][options][/]\n")
        cmd_table = Table(box=None, show_header=False, padding=(0, 2))
        cmd_table.add_column(style="bold cyan", width=16)
        cmd_table.add_column(style="dim")
        cmd_table.add_row("query", "Ask a question")
        cmd_table.add_row("serve", "Start API server + Web UI")
        cmd_table.add_row("index", "Index documents")
        cmd_table.add_row("index-url", "Index a URL")
        cmd_table.add_row("demo", "Run built-in demo")
        cmd_table.add_row("benchmark", "Run evaluation")
        cmd_table.add_row("download-models", "Download ML models")
        cmd_table.add_row("list-models", "List available models")
        cmd_table.add_row("status", "Show system status")
        console.print(cmd_table)
        sys.exit(0)

    cmd_name = args.command.replace("-", "_")
    globals()[f"cmd_{cmd_name}"](args)


if __name__ == "__main__":
    main()
