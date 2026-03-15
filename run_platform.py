from __future__ import annotations

import sys, os
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import asyncio
import subprocess
import time
from datetime import datetime, timezone
from typing import Optional

import typer
from dotenv import load_dotenv
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

load_dotenv(dotenv_path=_ROOT / ".env")

app = typer.Typer(help="CareClaimAI Platform Orchestrator")
console = Console()

BANNER = r"""
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Autonomous CMS Policy Monitoring · Real-Time Claim Adjudication
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def print_banner():
    console.print(BANNER, style="bold blue")
    console.print(Panel(
        "[bold cyan]CareClaimAI v2.0[/bold cyan] — "
        "Regulatory Entropy Management System\n"
        f"[dim]Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="blue",
    ))


def print_service_table(services: list[dict]):
    table = Table(
        title="Service Status",
        box=box.ROUNDED,
        border_style="blue",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Service",    style="white",   min_width=25)
    table.add_column("Status",     style="green",   min_width=12)
    table.add_column("Endpoint",   style="cyan",    min_width=30)
    table.add_column("Details",    style="dim",     min_width=30)

    for svc in services:
        status = svc.get("status", "STOPPED")
        color  = "green" if status == "RUNNING" else "red" if status == "ERROR" else "yellow"
        table.add_row(
            svc["name"],
            f"[{color}]{status}[/{color}]",
            svc.get("endpoint", "N/A"),
            svc.get("details", ""),
        )

    console.print(table)

# Environment Checks

def check_environment() -> bool:
    console.print("\n[bold]Checking environment...[/bold]")
    ok = True

    checks = [
        ("MONGO_URI",      os.getenv("MONGO_URI",  "mongodb://localhost:27017"), True),
        ("REDIS_URL",      os.getenv("REDIS_URL",  "redis://localhost:6379/0"),  False),
        ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""),                      False),
    ]

    for name, value, required in checks:
        if value:
            console.print(f"  ✅ {name}: [dim]{value[:40]}...[/dim]" if len(value) > 40
                          else f"  ✅ {name}: [dim]{value}[/dim]")
        elif required:
            console.print(f"  ❌ [red]{name}[/red]: Not set (required)")
            ok = False
        else:
            console.print(f"  ⚠️  [yellow]{name}[/yellow]: Not set (optional)")

    if not ok:
        console.print("\n[red]Critical environment variables missing![/red]")
        console.print("Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and fill in required values.")
    return ok


# Initial CMS Scan (real data only — no mocks)

async def run_initial_scan():

    console.print("\n[bold cyan]Running initial CMS policy scan...[/bold cyan]")
    console.print("[dim]Scraping live CMS sources — this takes 30-60 seconds[/dim]\n")

    from database.mongo import MongoManager
    from workers.policy_monitor import PolicyMonitorWorker

    await MongoManager.connect()
    worker = PolicyMonitorWorker()
    # Initialise Redis + pipeline without starting the full scheduler
    try:
        import redis.asyncio as aioredis
        from workers.policy_monitor import _NullRedis, PolicyProcessingPipeline
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            r = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            await r.ping()
        except Exception:
            r = _NullRedis()

        from scraper.cms_scraper import CMSScraper
        scraper = CMSScraper(concurrency=4)
        pipeline = PolicyProcessingPipeline(r)

        console.print("[cyan]Fetching CMS pages...[/cyan]")
        policies = await scraper.scrape_all()
        console.print(f"[green]Found {len(policies)} policy document(s)[/green]")

        import asyncio as _asyncio
        sem = _asyncio.Semaphore(3)

        async def _proc(p):
            async with sem:
                return await pipeline.process(p)

        if policies:
            results = await _asyncio.gather(*[_proc(p) for p in policies], return_exceptions=True)
            ok  = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "completed")
            err = sum(1 for r in results if isinstance(r, Exception))
            console.print(f"\n[bold green]✅ Initial scan complete: {ok} policies processed, {err} errors[/bold green]")
        else:
            console.print("[yellow]No new policies detected on initial scan (CMS pages may be unchanged or unreachable)[/yellow]")

    except Exception as e:
        console.print(f"[red]Initial scan error: {e}[/red]")
        logger.exception("Initial scan failed")
    finally:
        await MongoManager.disconnect()



# Service Launchers


def start_api_server() -> subprocess.Popen:
    log_dir = _ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = open(log_dir / "api.log", "a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.billing_api:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        stdout=log_file,
        stderr=log_file,
        cwd=str(_ROOT),
        env=env,
    )
    console.print("  🚀 [green]FastAPI[/green] started (PID {}) → http://localhost:8000".format(proc.pid))
    console.print("     API Docs → http://localhost:8000/docs")
    console.print("     Logs     → logs/api.log")
    return proc


def start_dashboard() -> subprocess.Popen:
    log_dir = _ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = open(log_dir / "dashboard.log", "a", encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT)
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         str(_ROOT / "dashboard" / "main_dashboard.py"),
         "--server.port", "8501",
         "--server.headless", "true",
         "--theme.base", "dark"],
        stdout=log_file,
        stderr=log_file,
        cwd=str(_ROOT),
        env=env,
    )
    console.print("  🚀 [green]Dashboard[/green] started (PID {}) → http://localhost:8501".format(proc.pid))
    console.print("     Logs → logs/dashboard.log")
    return proc


async def run_monitor_worker():
    import sys, os
    from pathlib import Path
    _r = Path(__file__).resolve().parent
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))
    from workers.policy_monitor import run_worker
    await run_worker()



# CLI Commands


@app.command()
def run(
    api_only:      bool = typer.Option(False, "--api-only",      help="Start only the billing API"),
    worker_only:   bool = typer.Option(False, "--worker-only",   help="Start only the monitor worker"),
    initial_scan:  bool = typer.Option(False, "--initial-scan",  help="Run a live CMS scan before starting services"),
    no_dashboard:  bool = typer.Option(False, "--no-dashboard",  help="Skip the Streamlit dashboard"),
    skip_env:      bool = typer.Option(False, "--skip-env",      help="Skip env checks"),
):
    """Start the CareClaimAI compliance platform."""
    print_banner()

    if not skip_env:
        if not check_environment():
            console.print("\n[yellow]Use --skip-env to proceed anyway[/yellow]")
            raise typer.Exit(1)

    if initial_scan:
        asyncio.run(run_initial_scan())

    services = []

    if worker_only:
        console.print("\n[bold]Starting policy monitor worker...[/bold]")
        asyncio.run(run_monitor_worker())
        return

    procs = []

    if not worker_only:
        console.print("\n[bold]Starting services...[/bold]\n")

        # API
        api_proc = start_api_server()
        procs.append(api_proc)
        services.append({
            "name":     "Billing Validation API",
            "status":   "RUNNING",
            "endpoint": "http://localhost:8000",
            "details":  "FastAPI + Uvicorn",
        })
        time.sleep(2)

        # Dashboard
        if not api_only and not no_dashboard:
            dash_proc = start_dashboard()
            procs.append(dash_proc)
            services.append({
                "name":     "Compliance Dashboard",
                "status":   "RUNNING",
                "endpoint": "http://localhost:8501",
                "details":  "Streamlit",
            })
            time.sleep(2)

        # Monitor worker (async, in background thread)
        if not api_only:
            import threading
            def _worker_target():
                import sys, os
                from pathlib import Path
                _r = Path(__file__).resolve().parent
                if str(_r) not in sys.path:
                    sys.path.insert(0, str(_r))
                os.chdir(_r)
                asyncio.run(run_monitor_worker())

            worker_thread = threading.Thread(
                target=_worker_target,
                daemon=True,
                name="PolicyMonitor",
            )
            worker_thread.start()
            services.append({
                "name":     "Policy Monitor Worker",
                "status":   "RUNNING",
                "endpoint": "Background",
                "details":  "APScheduler (30min scans)",
            })

    print_service_table(services)
    console.print("\n[dim]Press Ctrl+C to stop all services[/dim]\n")

    try:
        while True:
            time.sleep(5)
            for proc in procs:
                ret = proc.poll()
                if ret is not None:
                    name = "API" if "uvicorn" in str(proc.args) else "Dashboard"
                    console.print(f"[red]{name} process (PID {proc.pid}) exited with code {ret}[/red]")
                    log_hint = "logs/api.log" if name == "API" else "logs/dashboard.log"
                    console.print(f"[yellow]Check {log_hint} for the full error[/yellow]")
                    procs.remove(proc)
                    # Auto-restart
                    console.print(f"[cyan]Restarting {name}...[/cyan]")
                    if name == "API":
                        new_proc = start_api_server()
                    else:
                        new_proc = start_dashboard()
                    procs.append(new_proc)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down CareClaimAI...[/yellow]")
        for proc in procs:
            proc.terminate()
        console.print("[green]All services stopped. Goodbye![/green]")


@app.command()
def check():
    print_banner()
    check_environment()


if __name__ == "__main__":
    app()
