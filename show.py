import shutil
import plotille
import random, time
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.live import Live

# Info Panel
def info_panel(step, loss):
    table = Table.grid(padding=1)
    table.add_row("[bold yellow]Epoch", f"{step // 20 + 1}/5")
    table.add_row("[bold yellow]Step", str(step))
    table.add_row("[bold yellow]Items", f"{step * 64}/64000")
    table.add_row("[bold yellow]Loss", f"{loss:.4f}")
    table.add_row("[bold yellow]CPU Temp", f"{random.randint(75, 98)} °C")
    return Panel(table, title="📊 Training Info")

# Graph Panel
def graph_panel(losses):
    # Get terminal size
    term_size = shutil.get_terminal_size((80, 24))
    term_width = term_size.columns
    term_height = term_size.lines

    # Conservative figure size
    fig_width = max(20, (term_width // 2)-25)
    fig_height = max(8, term_height -9)

    # Trim data
    x_vals = list(range(len(losses)))[:]
    y_vals = losses[:]

    if len(x_vals) < 2:
        x_vals = [0, 1]
        y_vals = [losses[-1]] * 2

    # Build plot
    fig = plotille.Figure()
    fig.width = fig_width
    fig.height = fig_height
    fig.color_mode = 'names'
    fig.set_x_limits(min_=0, max_=len(losses))

    y_min, y_max = 0, max(y_vals) + 0.1
    fig.set_y_limits(min_=y_min, max_=y_max)
    fig.plot(x_vals, y_vals, label="Loss")

    # Trim output to fit
    graph_str = fig.show(legend=False)
    # trimmed = "\n".join(graph_str.splitlines()[:24])

    return Panel(graph_str, title="📉 Loss (plotille)", padding=(0, 1))

# Progress bar
progress = Progress(
    TextColumn("Step {task.completed}/{task.total}"),
    BarColumn(bar_width=None),
    TimeRemainingColumn(),
    expand=True,
)
task_id = progress.add_task("Training", total=200)

# Layout setup
layout = Layout()
layout.split(
    Layout(name="main", ratio=2),
    Layout(name="progress", size=3),
)
layout["main"].split_row(
    Layout(name="info", size=25),
    Layout(name="plot", ratio=2),
    Layout(name="plot")
)

losses = []
lrs = []

# Live dashboard
with Live(layout, refresh_per_second=10, screen=False):
    for step in range(200):
        loss = max(0.01, 2.0 * (0.95 ** step) + random.uniform(-0.05, 0.05))
        losses.append(loss)
        lrs.append(0.1 * (0.95 ** step) + random.uniform(-0.01, 0.01))
        progress.update(task_id, advance=1)

        layout["info"].update(info_panel(step, loss))
        layout["plot"].update(graph_panel(losses))
        layout["progress"].update(Panel(progress, title="🚀 Progress"))

        time.sleep(0.1)
