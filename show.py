import time
import random
import curses
import math
import threading
from datetime import datetime

class TrainingVisualizer:
    def __init__(self, window):
        self.window = window
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_YELLOW, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_CYAN, -1)
        
        self.yellow = curses.color_pair(1)
        self.red = curses.color_pair(2)
        self.cyan = curses.color_pair(3)
        
        # Hide cursor
        curses.curs_set(0)
        
        # Training parameters
        self.epochs = 10
        self.current_epoch = 1
        self.iterations_per_epoch = 1000
        self.current_iteration = 0
        self.total_items = 60000
        self.processed_items = 0
        
        # Metrics history
        self.loss_history = []
        self.running = True
        self.active_plot = "Loss"  # Default plot
        
        # Mock data
        self.metrics = {
            'accuracy': {'train': 0.7, 'valid': 0.68},
            'cpu_usage': {'train': 15.0, 'valid': 22.0},
            'memory': {'train': 8.5, 'valid': 8.4},
            'temperature': {'train': 65.0, 'valid': 65.0},
            'loss': {'train': 0.45, 'valid': 0.48}
        }

    def simulate_training(self):
        """Simulate the training process with randomly improving metrics"""
        while self.running and self.current_epoch <= self.epochs:
            # Update iteration counters
            self.current_iteration += 1
            self.processed_items += random.randint(50, 150)
            
            # Simulate improving metrics
            self.metrics['accuracy']['train'] = min(0.99, self.metrics['accuracy']['train'] + random.uniform(0.0001, 0.001))
            self.metrics['accuracy']['valid'] = min(0.99, self.metrics['accuracy']['train'] - random.uniform(0, 0.02))
            
            self.metrics['loss']['train'] = max(0.01, self.metrics['loss']['train'] * 0.998)
            self.metrics['loss']['valid'] = max(0.01, self.metrics['loss']['train'] * (1 + random.uniform(-0.1, 0.1)))
            
            self.metrics['cpu_usage']['train'] = min(100, max(5, self.metrics['cpu_usage']['train'] + random.uniform(-1, 1)))
            self.metrics['cpu_usage']['valid'] = min(100, max(5, self.metrics['cpu_usage']['valid'] + random.uniform(-1, 1)))
            
            self.metrics['temperature']['train'] = min(100, max(50, self.metrics['temperature']['train'] + random.uniform(-0.5, 0.5)))
            self.metrics['temperature']['valid'] = self.metrics['temperature']['train']
            
            # Record loss for plotting
            self.loss_history.append(self.metrics['loss']['train'])
            if len(self.loss_history) > 100:
                self.loss_history = self.loss_history[-100:]
            
            # Check if epoch is complete
            if self.current_iteration % self.iterations_per_epoch == 0:
                self.current_epoch += 1
                self.current_iteration = 0
            
            time.sleep(0.1)

    
    def draw_frame(self):
        self.window.clear()
        max_y, max_x = self.window.getmaxyx()
        
        # Ensure minimum size
        if max_y < 24 or max_x < 80:
            self.window.addstr(0, 0, "Terminal too small! Please resize to at least 80x24")
            self.window.refresh()
            return
        
        # Draw borders
        self.draw_box(0, 0, max_y//2, max_x//2, "Controls")
        self.draw_box(0, max_x//2, max_y//2, max_x, "Plots")
        self.draw_box(max_y//2, 0, max_y-3, max_x//2, "Metrics")
        self.draw_box(max_y-3, 0, max_y, max_x//2, "Status")
        self.draw_box(max_y-10, 0, max_y, max_x//2, "Progress")
        
        # Draw controls
        self.window.addstr(2, 3, "Quit", self.yellow)
        self.window.addstr(2, 20, ": q", self.yellow)
        self.window.addstr(2, 30, "Stop the training.")
        
        self.window.addstr(3, 3, "Plots Metrics", self.yellow)
        self.window.addstr(3, 20, ": ← →", self.yellow)
        self.window.addstr(3, 30, "Switch between metrics.")
        
        self.window.addstr(4, 3, "Plots Type", self.yellow)
        self.window.addstr(4, 20, ": ↑ ↓", self.yellow)
        self.window.addstr(4, 30, "Switch between types.")
        
        # Draw metrics
        y_pos = max_y//2 + 2
        
        # Accuracy
        self.window.addstr(y_pos, 3, "Accuracy", self.yellow)
        self.window.addstr(y_pos+1, 3, "  Train")
        self.window.addstr(y_pos+1, 12, f"epoch {self.current_epoch}.{self.current_iteration:02d} % - batch {self.metrics['accuracy']['train']*100:.2f} %")
        self.window.addstr(y_pos+2, 3, "  Valid")
        self.window.addstr(y_pos+2, 12, f"epoch {self.current_epoch}.{self.current_iteration:02d} % - batch {self.metrics['accuracy']['valid']*100:.2f} %")
        
        # CPU Usage
        y_pos += 4
        self.window.addstr(y_pos, 3, "CPU Usage", self.yellow)
        self.window.addstr(y_pos+1, 3, "  Train")
        self.window.addstr(y_pos+1, 12, f"CPU Usage: {self.metrics['cpu_usage']['train']:.2f} %")
        self.window.addstr(y_pos+2, 3, "  Valid")
        self.window.addstr(y_pos+2, 12, f"CPU Usage: {self.metrics['cpu_usage']['valid']:.2f} %")
        
        # Memory
        y_pos += 4
        self.window.addstr(y_pos, 3, "CPU Memory", self.yellow)
        self.window.addstr(y_pos+1, 3, "  Train")
        self.window.addstr(y_pos+1, 12, f"RAM Used: {self.metrics['memory']['train']:.2f} / 33.33 Gb")
        self.window.addstr(y_pos+2, 3, "  Valid")
        self.window.addstr(y_pos+2, 12, f"RAM Used: {self.metrics['memory']['valid']:.2f} / 33.33 Gb")
        
        # Temperature
        y_pos += 4
        self.window.addstr(y_pos, 3, "CPU Temperature", self.yellow)
        self.window.addstr(y_pos+1, 3, "  Train")
        self.window.addstr(y_pos+1, 12, f"CPU Temperature: {self.metrics['temperature']['train']:.2f} °C")
        self.window.addstr(y_pos+2, 3, "  Valid")
        self.window.addstr(y_pos+2, 12, f"CPU Temperature: {self.metrics['temperature']['valid']:.2f} °C")
        
        # Loss
        y_pos += 4
        self.window.addstr(y_pos, 3, "Loss", self.yellow)
        self.window.addstr(y_pos+1, 3, "  Train")
        self.window.addstr(y_pos+1, 12, f"epoch {self.metrics['loss']['train']:.2e} - batch {self.metrics['loss']['train']/1.5:.2e}")
        self.window.addstr(y_pos+2, 3, "  Valid")
        self.window.addstr(y_pos+2, 12, f"epoch {self.metrics['loss']['valid']:.2e} - batch {self.metrics['loss']['valid']/1.8:.2e}")
        
        # Draw status
        self.window.addstr(max_y-2, 3, "Mode", self.yellow)
        self.window.addstr(max_y-2, 20, ": Training")
        
        self.window.addstr(max_y-3, 3, "Epoch", self.yellow)
        self.window.addstr(max_y-3, 20, f": {self.current_epoch}/{self.epochs}")
        
        self.window.addstr(max_y-4, 3, "Iteration", self.yellow)
        self.window.addstr(max_y-4, 20, f": {self.current_iteration}")
        
        self.window.addstr(max_y-5, 3, "Items", self.yellow)
        self.window.addstr(max_y-5, 20, f": {self.processed_items}/{self.total_items}")
        
        # Draw progress bar
        progress = self.processed_items / self.total_items
        bar_width = max_x//2 - 4
        filled_width = int(bar_width * progress)
        
        self.window.addstr(max_y-8, 2, "█" * filled_width, curses.A_BOLD)
        self.window.addstr(max_y-8, max_x//2 - 15, f"{progress*100:.0f}%", self.yellow)
        
        seconds_elapsed = int(time.time() - self.start_time)
        self.window.addstr(max_y-8, max_x//2 - 4, f"({seconds_elapsed} secs)")
        
        # Draw plot header
        plot_x = max_x//2 + 2
        self.window.addstr(1, plot_x, "Accuracy", self.yellow)
        self.window.addstr(1, plot_x + 10, "| CPU Usage", self.yellow)
        self.window.addstr(1, plot_x + 22, "| CPU Memory", self.yellow)
        self.window.addstr(1, plot_x + 35, "| CPU Temperature", self.yellow)
        self.window.addstr(1, plot_x + 52, "| Loss", self.yellow)
        
        # Draw loss plot
        self.draw_loss_plot(2, max_x//2 + 1, max_y-3, max_x-2)
        
        # Draw plot legends
        plot_y = max_y-3
        legend_x = max_x - 50
        self.window.addstr(plot_y+1, legend_x, "Train", self.red)
        self.window.addstr(plot_y+1, legend_x + 12, "Valid", self.cyan)
        
        self.window.refresh()

    def draw_loss_plot(self, y1, x1, y2, x2):
        if not self.loss_history:
            return
        
        plot_height = y2 - y1 - 1
        plot_width = x2 - x1 - 1
        
        # Calculate scaling
        max_val = max(self.loss_history)
        min_val = min(self.loss_history)
        value_range = max_val - min_val
        
        if value_range == 0:
            value_range = 1
        
        # Draw y-axis labels
        self.window.addstr(y1+1, x1, f"{max_val:.2e}")
        self.window.addstr(y2-1, x1, f"{min_val:.2e}")
        
        # Draw x-axis labels
        self.window.addstr(y2-1, x1, "0")
        self.window.addstr(y2-1, x2-4, str(len(self.loss_history)))
        
        # Draw the plot points
        for i, val in enumerate(self.loss_history):
            plot_x = x1 + int((i / len(self.loss_history)) * plot_width)
            plot_y = y2 - 1 - int(((val - min_val) / value_range) * plot_height)
            
            if 0 <= plot_y < y2 and 0 <= plot_x < x2:  # Check bounds
                self.window.addstr(plot_y, plot_x, ".", self.red)

    def draw_box(self, y1, x1, y2, x2, title=None):
        """Draw a box with an optional title"""
        self.window.attron(curses.A_BOLD)
        # Draw corners
        self.window.addch(y1, x1, curses.ACS_ULCORNER)
        self.window.addch(y1, x2-1, curses.ACS_URCORNER)
        self.window.addch(y2-1, x1, curses.ACS_LLCORNER)
        self.window.addch(y2-1, x2-1, curses.ACS_LRCORNER)
        
        # Draw edges
        for x in range(x1+1, x2-1):
            self.window.addch(y1, x, curses.ACS_HLINE)
            self.window.addch(y2-1, x, curses.ACS_HLINE)
        
        for y in range(y1+1, y2-1):
            self.window.addch(y, x1, curses.ACS_VLINE)
            self.window.addch(y, x2-1, curses.ACS_VLINE)
        
        self.window.attroff(curses.A_BOLD)
        
        # Add title if provided
        if title:
            self.window.addstr(y1, x1+2, f"-{title}-", self.yellow)

    def run(self):
        self.start_time = time.time()
        
        # Start the training simulation in a separate thread
        training_thread = threading.Thread(target=self.simulate_training)
        training_thread.daemon = True
        training_thread.start()
        
        try:
            while self.running:
                self.draw_frame()
                
                # Check for user input (non-blocking)
                self.window.timeout(100)
                key = self.window.getch()
                
                if key == ord('q'):
                    self.running = False
                elif key == curses.KEY_LEFT:
                    # Switch plot metric to the left
                    pass
                elif key == curses.KEY_RIGHT:
                    # Switch plot metric to the right
                    pass
                
        except KeyboardInterrupt:
            self.running = False

def main(stdscr):
    visualizer = TrainingVisualizer(stdscr)
    visualizer.run()

if __name__ == "__main__":
    curses.wrapper(main)