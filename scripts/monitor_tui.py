#!/usr/bin/env python3
"""
VulnLab TUI Monitor — reemplaza monitor_alertas.sh
Uso: python scripts/monitor_tui.py [container]
"""
import json
import subprocess
import sys
from collections import defaultdict

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Header, Footer, RichLog, Static
from textual import work


CONTAINER = sys.argv[1] if len(sys.argv) > 1 else "vulnlab"

SEV_STYLE = {
    "CRITICO": ("bold red",         "●"),
    "ALTO":    ("bold dark_orange", "●"),
    "MEDIO":   ("bold yellow",      "●"),
    "INFO":    ("bold blue",        "●"),
}

VULN_TYPES = [
    "SQL_INJECTION", "XSS", "CMD_INJECTION", "PATH_TRAVERSAL", "SSRF",
    "INFO_DISCLOSURE", "BRUTE_FORCE", "DESERIALIZATION", "BROKEN_AUTH_API",
    "CRYPTO_FAILURE", "INSECURE_DESIGN", "VULN_COMPONENTS", "IDOR",
]

OWASP_SEV = {
    "SQL_INJECTION":   "CRITICO", "CMD_INJECTION":  "CRITICO",
    "DESERIALIZATION": "CRITICO", "CRYPTO_FAILURE": "CRITICO",
    "INFO_DISCLOSURE": "CRITICO", "XSS":            "ALTO",
    "IDOR":            "ALTO",    "PATH_TRAVERSAL": "ALTO",
    "SSRF":            "ALTO",    "BRUTE_FORCE":    "ALTO",
    "BROKEN_AUTH_API": "ALTO",    "INSECURE_DESIGN":"ALTO",
    "VULN_COMPONENTS": "MEDIO",
}


class CountersPanel(Static):
    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self._counts: dict = defaultdict(int)
        self._sev: dict    = defaultdict(int)
        self._total: int   = 0

    def add(self, vuln_type: str, severity: str) -> None:
        self._counts[vuln_type] += 1
        self._sev[severity]     += 1
        self._total             += 1
        self._redraw()

    def _redraw(self) -> None:
        lines = ["[bold green]── Tipo ──────────────────────[/]\n"]

        for vt in VULN_TYPES:
            cnt   = self._counts.get(vt, 0)
            sev   = OWASP_SEV.get(vt, "INFO")
            style, dot = SEV_STYLE.get(sev, ("white", "●"))
            bar   = "[red]" + "█" * min(cnt, 8) + "[/]"
            dim   = "dim " if cnt == 0 else ""
            lines.append(
                f"[{dim}{style}]{dot} {vt:<22}[/] "
                f"[{'bold white' if cnt else 'dim'}]{cnt:>3}[/] "
                f"{bar}"
            )

        lines.append("\n[bold green]── Severidad ─────────────────[/]\n")
        for sev in ("CRITICO", "ALTO", "MEDIO", "INFO"):
            cnt   = self._sev.get(sev, 0)
            style, _ = SEV_STYLE.get(sev, ("white", "●"))
            lines.append(
                f"  [{'bold white' if cnt else 'dim'}]{sev:<10}[/] "
                f"[{style if cnt else 'dim'}]{cnt:>3}[/]"
            )

        lines.append("\n[bold green]── Total ─────────────────────[/]\n")
        lines.append(f"  [bold white]{self._total}[/] alertas detectadas")
        self.update("\n".join(lines))


class VulnMonitor(App):
    CSS = """
    Screen  { background: #0d0d0d; }
    Header  { background: #161616; color: #22c55e; }
    Footer  { background: #161616; }
    Horizontal { height: 1fr; }

    #counters {
        width: 40;
        border-right: solid #2a2a2a;
        padding: 1 1;
        background: #0f0f0f;
    }
    #feed { width: 1fr; padding: 0 1; }
    """

    TITLE = "VulnLab Monitor"
    BINDINGS = [
        ("q", "quit",         "Salir"),
        ("c", "action_clear", "Limpiar feed"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield CountersPanel(id="counters")
            yield RichLog(id="feed", markup=True, highlight=True, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"VulnLab Monitor  [{CONTAINER}]"
        feed = self.query_one("#feed", RichLog)
        feed.write(f"[dim]Conectando a [bold]{CONTAINER}[/bold]...[/dim]")
        self.stream_docker()

    @work(thread=True)
    def stream_docker(self) -> None:
        feed = self.query_one("#feed", RichLog)

        try:
            proc = subprocess.Popen(
                ["docker", "logs", "-f", "--tail=0", CONTAINER],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            self.call_from_thread(
                feed.write,
                "[bold red]Error:[/] docker no encontrado en PATH"
            )
            return

        if proc.poll() is not None:
            self.call_from_thread(
                feed.write,
                f"[bold red]Error:[/] contenedor [bold]{CONTAINER}[/bold] no está corriendo\n"
                "[dim]Ejecuta: make start[/dim]"
            )
            return

        self.call_from_thread(
            feed.write,
            f"[dim green]Conectado — esperando alertas en [bold]{CONTAINER}[/bold]...[/dim green]\n"
        )

        for raw_line in proc.stdout:
            line = raw_line.strip()
            if "[VULN_ALERT]" not in line:
                continue
            json_str = line.split("[VULN_ALERT]", 1)[-1].strip()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                continue
            self.call_from_thread(self._render_alert, data)

    def _render_alert(self, d: dict) -> None:
        feed  = self.query_one("#feed", RichLog)
        panel = self.query_one(CountersPanel)

        sev     = d.get("severity", "INFO")
        vtype   = d.get("type",     "UNKNOWN")
        ts      = d.get("ts",       "?")
        ip      = d.get("ip",       "?")
        method  = d.get("method",   "?")
        ep      = d.get("endpoint", "?")
        payload = d.get("payload",  "")[:80]
        detail  = d.get("detail",   "")
        owasp   = d.get("owasp",    "?")

        style, dot = SEV_STYLE.get(sev, ("white", "●"))

        feed.write(
            f"[{style}]{dot} [{sev}] {vtype}[/]  "
            f"[dim]{owasp}  {ts}[/dim]"
        )
        feed.write(f"  [cyan]ip[/cyan] {ip}  [dim]{method} {ep}[/dim]")
        if payload:
            feed.write(f"  [dim]payload[/dim] [{style}]{payload}[/]")
        if detail:
            feed.write(f"  [dim]{detail}[/dim]")
        feed.write("")

        panel.add(vtype, sev)

    def action_clear(self) -> None:
        self.query_one("#feed", RichLog).clear()


if __name__ == "__main__":
    VulnMonitor().run()
