import sys
import time
import tkinter as tk
from tkinter import scrolledtext, Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

from sfm_lite import load_calibration_data, SFM


class StdoutRedirector(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)  # Scroll to the bottom

    def flush(self):
        pass


class SFMApplication:
    def __init__(self, master):
        self.master = master
        self.thread = None
        self.thread_running = False
        self.plot_window = None
        self.canvas = None
        self.view_init = None  # 添加属性存储视角信息

        master.title("SFM Reconstruction GUI")

        # Set up the GUI
        tk.Label(master, text="Image Directory:").grid(row=0, column=0, sticky='ew')
        self.image_dir_entry = tk.Entry(master, width=50)
        self.image_dir_entry.grid(row=0, column=1, sticky='ew')
        self.image_dir_entry.insert(0, "./ImageDataset_SceauxCastle/images")

        tk.Label(master, text="Calibration File:").grid(row=1, column=0, sticky='ew')
        self.cal_file_entry = tk.Entry(master, width=50)
        self.cal_file_entry.grid(row=1, column=1, sticky='ew')
        self.cal_file_entry.insert(0, "./ImageDataset_SceauxCastle/images/K.txt")

        tk.Label(master, text="Use Bundle Adjustment:").grid(row=2, column=0, sticky='ew')
        self.use_ba_var = tk.BooleanVar(value=False)
        tk.Checkbutton(master, text="Enable", variable=self.use_ba_var).grid(row=2, column=1, sticky='ew')

        tk.Label(master, text="BA tol").grid(row=3, column=0, sticky='ew')
        self.ba_tol_entry = tk.Entry(master, width=50)
        self.ba_tol_entry.grid(row=3, column=1, sticky='ew')
        self.ba_tol_entry.insert(0, "1e-10")

        # Add radiobuttons for color selection
        self.color_mode_var = tk.StringVar(value="increment")
        tk.Label(master, text="Color Mode:").grid(row=4, column=0, sticky='ew')
        tk.Radiobutton(master, text="Increment Colors", variable=self.color_mode_var, value="increment").grid(row=4, column=1, sticky='w')
        tk.Radiobutton(master, text="Average Colors", variable=self.color_mode_var, value="average").grid(row=4, column=1, sticky='e')

        tk.Button(master, text="Run Reconstruction", command=self.start_thread).grid(row=5, columnspan=2, sticky='ew')

        self.output_console = scrolledtext.ScrolledText(master, height=15)
        self.output_console.grid(row=6, columnspan=2, sticky='nsew')
        sys.stdout = StdoutRedirector(self.output_console)

        self.fig = plt.Figure(figsize=(5, 4))

    def start_thread(self):
        if self.thread_running:
            return  # Ignore button clicks if a thread is already running
        self.thread_running = True
        self.thread = threading.Thread(target=self.run_reconstruction)
        self.thread.start()

    def run_reconstruction(self):
        image_dir = self.image_dir_entry.get()
        cal_file = self.cal_file_entry.get()
        use_ba = self.use_ba_var.get()
        ba_tol = float(self.ba_tol_entry.get())

        try:
            print("Loading calibration data...")
            K = load_calibration_data(cal_file)
            print("Initializing SFM...")
            sfm = SFM(image_dir, K)
            print("Running reconstruction...")

            def plot():
                color_mode = self.color_mode_var.get()
                if color_mode == "increment":
                    colors = sfm.graph.increment_colors
                else:
                    colors = sfm.graph.colors
                self.master.after(0, self.plot_results, sfm.graph.X3d, colors)

            sfm.construct(use_ba=use_ba, ba_tol=ba_tol, verbose=0, callback=plot, interval=0.1)
            print("Reconstruction completed successfully.")
            plot()
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            self.thread_running = False

    def stop_thread(self):
        if self.thread_running and self.thread.is_alive():
            # 这里可以设置一个标志，通知线程停止
            self.thread_running = False
            self.thread.join()  # 等待线程结束

    def plot_results(self, X3d, colors):
        if self.plot_window is None or not self.plot_window.winfo_exists():
            self.plot_window = Toplevel(self.master)
            self.plot_window.title("Reconstruction Results")
            self.canvas = FigureCanvasTkAgg(self.fig, self.plot_window)
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            self.fig.clear()

        ax = self.fig.add_subplot(111, projection='3d')

        # 检查之前是否有视角信息
        if self.view_init is not None:
            ax.view_init(elev=self.view_init[0], azim=self.view_init[1])

        ax.scatter(X3d[:, 0], X3d[:, 1], X3d[:, 2], c=colors / 255.0, s=5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 保存当前的视角信息
        self.view_init = (ax.elev, ax.azim)

        self.canvas.draw()

        # 保存用户交互后的视角
        def update_view(event):
            self.view_init = (ax.elev, ax.azim)

        self.canvas.mpl_connect('motion_notify_event', update_view)


if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(False, False)
    app = SFMApplication(root)
    root.mainloop()
