import sys
import tkinter as tk
from tkinter import scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sfm import load_calibration_data, SFM
from sfm.visualize import visualize_points3d


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
        master.title("SFM Reconstruction GUI")

        # Redirect stdout
        self.output_console = scrolledtext.ScrolledText(master, height=10, width=70)
        self.output_console.grid(row=5, columnspan=2)
        sys.stdout = StdoutRedirector(self.output_console)

        # Set up the GUI
        tk.Label(master, text="Image Directory:").grid(row=0)
        self.image_dir_entry = tk.Entry(master, width=50)
        self.image_dir_entry.grid(row=0, column=1)
        self.image_dir_entry.insert(0, "./ImageDataset_SceauxCastle/images")

        tk.Label(master, text="Calibration File:").grid(row=1)
        self.cal_file_entry = tk.Entry(master, width=50)
        self.cal_file_entry.grid(row=1, column=1)
        self.cal_file_entry.insert(0, "./ImageDataset_SceauxCastle/images/K.txt")

        tk.Label(master, text="Use Bundle Adjustment:").grid(row=2)
        self.use_ba_var = tk.BooleanVar(value=False)
        tk.Checkbutton(master, text="Enable", variable=self.use_ba_var).grid(row=2, column=1)

        tk.Label(master, text="BA tol").grid(row=3)
        self.ba_tol_entry = tk.Entry(master, width=50)
        self.ba_tol_entry.grid(row=3, column=1)
        self.ba_tol_entry.insert(0, "1e-10")

        tk.Button(master, text="Run Reconstruction", command=self.run_reconstruction).grid(row=4, columnspan=2)

        self.fig = plt.Figure(figsize=(5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master)
        self.canvas.get_tk_widget().grid(row=6, columnspan=2)

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
            X3d, colors = sfm.reconstruct(use_ba=use_ba, ba_tol=ba_tol)
            self.plot_results(X3d, colors)
            print("Reconstruction completed successfully.")
        except Exception as e:
            print(f"Error: {str(e)}")

    def plot_results(self, X3d, colors):
        ax = self.fig.add_subplot(111, projection='3d')
        ax.clear()
        ax.scatter(X3d[:, 0], X3d[:, 1], X3d[:, 2], c=colors/255.0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        self.canvas.draw()


if __name__ == '__main__':
    root = tk.Tk()
    app = SFMApplication(root)
    root.mainloop()