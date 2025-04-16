import shutil
import plotille
import random
import time
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.live import Live


class TrainingDashboard:
    def __init__(self, total_steps=4, batch_size=64, total_items=64000, epochs=1000):
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.total_items = total_items
        self.epochs = epochs
        
        # Initialize data storage
        self.losses = []
        self.learning_rates = []
        
        # Create layout
        self.layout = self._create_layout()
        
        # Initialize live display
        self.live = Live(self.layout, refresh_per_second=10, screen=False)
        self.live.start()
    
    def _info_panel(self, step, loss, gflops=None, epoch=None):
        """Create the info panel with training statistics"""
        table = Table.grid(padding=1)
        table.add_row("[bold yellow]Step", f"{step}")
        table.add_row("[bold yellow]Epoch", f"{epoch}")
        table.add_row("[bold yellow]Loss", f"{loss:.4e}")
        table.add_row("[bold yellow]GFOPS", f"{gflops:.0f} GFLOPs")
        return Panel(table, title="📊 Training Info")
    
    def _graph_panel_plot_loss(self, losses):
        # Get terminal size
        term_size = shutil.get_terminal_size((80, 24))
        term_width = term_size.columns
        term_height = term_size.lines
        
        # Conservative figure size
        fig_width = max(20, (term_width // 2) - 30)
        fig_height = max(8, term_height - 11)
        
        x_vals = list(range(len(losses)))
        y_vals = losses
        
        if len(x_vals) < 2:
            x_vals = [0, 1]
            y_vals = [losses[-1]] * 2
        
        # Build plot
        fig = plotille.Figure()
        fig.width = fig_width
        fig.height = fig_height
        fig.color_mode = 'names'
        fig.set_x_limits(min_=0, max_=len(losses))
        
        if(len(losses) < 2):
            y_min, y_max = 0, 1
        else:
            y_min, y_max = 0, max(losses[-min(len(losses)-1, 100):])*2
        
        fig.set_y_limits(min_=y_min, max_=y_max)
        fig.plot(x_vals, y_vals, label="Loss")
        
        # Trim output to fit
        graph_str = fig.show(legend=False)
        
        return Panel(graph_str, title="📉 Loss", padding=(0, 1))
    
    def _graph_panel_plot_lr(self, learning_rates):
        # Get terminal size
        term_size = shutil.get_terminal_size((80, 24))
        term_width = term_size.columns
        term_height = term_size.lines
        
        # Conservative figure size
        fig_width = max(20, (term_width // 2) - 30)
        fig_height = max(8, term_height - 11)
        
        x_vals = list(range(len(learning_rates)))
        y_vals = learning_rates
        
        if len(x_vals) < 2:
            x_vals = [0, 1]
            y_vals = [learning_rates[-1]] * 2
        
        # Build plot
        fig = plotille.Figure()
        fig.width = fig_width
        fig.height = fig_height
        fig.color_mode = 'names'
        fig.set_x_limits(min_=0, max_=len(learning_rates))
        
        if(len(learning_rates) < 2):
            y_min, y_max = 0, 1
        else:
            y_min, y_max = 0, max(learning_rates[-min(len(learning_rates)-1, 100):])*2
        
        fig.set_y_limits(min_=y_min, max_=y_max)
        fig.plot(x_vals, y_vals, label="Loss")
        
        # Trim output to fit
        graph_str = fig.show(legend=False)
        
        return Panel(graph_str, title="📉 Learning rate", padding=(0, 1))
    
    def _create_layout(self):
        # Progress bars
        self.step_progress = Progress(
            TextColumn("Step {task.completed}/{task.total}"),
            BarColumn(bar_width=None),
            TimeRemainingColumn(),
            expand=True,
        )
        self.step_task_id = self.step_progress.add_task("Training Steps", total=self.total_steps)
        
        self.epoch_progress = Progress(
            TextColumn("Epoch {task.completed}/{task.total}"),
            BarColumn(bar_width=None),
            TimeRemainingColumn(),
            expand=True,
        )
        self.epoch_task_id = self.epoch_progress.add_task("Training Epochs", total=self.epochs)
        
        # Layout setup
        layout = Layout()
        layout.split(
            Layout(name="main", ratio=2),
            Layout(name="progress", size=6),  # Increased size to accommodate two progress bars
        )
        layout["main"].split_row(
            Layout(name="info", size=25),
            Layout(name="plot_loss", ratio=2),
            Layout(name="plot_lr", ratio=2),
        )
        
        # Split progress section to show both progress bars
        layout["progress"].split(
            Layout(name="step_progress", size=3),
            Layout(name="epoch_progress", size=3),
        )
        
        return layout
    
    def update(self, step, loss, learning_rate=None, epoch=None, gflops=None):
        """Update the dashboard with new training data"""
        # Store data
        self.losses.append(loss)
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        # Update step progress (batches within an epoch)
        self.step_progress.update(self.step_task_id, completed=step)
        
        # Update epoch progress if epoch is provided
        if epoch is not None:
            self.epoch_progress.update(self.epoch_task_id, advance=1)
        
        # Update layout components
        self.layout["info"].update(self._info_panel(step, loss, gflops, epoch))
        self.layout["plot_loss"].update(self._graph_panel_plot_loss(self.losses))
        self.layout["plot_lr"].update(self._graph_panel_plot_lr(self.learning_rates))
        self.layout["step_progress"].update(Panel(self.step_progress, title="🚀 Batch Progress"))
        self.layout["epoch_progress"].update(Panel(self.epoch_progress, title="🌍 Epoch Progress"))
    
    def reset_step_progress(self):
        """Reset the step progress bar for a new epoch"""
        self.step_progress.reset(self.step_task_id)
    
    def close(self):
        """Stop the live display"""
        self.live.stop()

# # Example usage:
# if __name__ == "__main__":
#     # Create dashboard
#     dashboard = TrainingDashboard()
    
#     try:
#         # Simulate training loop
#         for step in range(200):
#             # Simulate loss and learning rate
#             loss = max(0.01, 2.0 * (0.95 ** step) + random.uniform(-0.05, 0.05))
#             lr = 0.1 * (0.95 ** step) + random.uniform(-0.01, 0.01)
            
#             # Update dashboard
#             dashboard.update(step, loss, lr)
            
#             # Simulate training time
#             time.sleep(0.1)
#     finally:
#         # Always close the dashboard when done
#         dashboard.close()
