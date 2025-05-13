import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


class HouseAIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HouseAI Model Trainer & Classifier")
        self.geometry("700x520")
        self.minsize(600, 400)

        tab_control = ttk.Notebook(self)
        self.train_tab = ttk.Frame(tab_control)
        self.classify_tab = ttk.Frame(tab_control)
        tab_control.add(self.train_tab, text='Train Model')
        tab_control.add(self.classify_tab, text='Classify Image')
        tab_control.pack(expand=1, fill='both')

        self.create_train_tab()
        self.create_classify_tab()

        self.sample_imgs = []
        self.annotation_content = None

    def create_train_tab(self):
        # Main frame for padding and layout
        main_frame = ttk.Frame(self.train_tab, padding=15)
        main_frame.pack(fill='both', expand=True)

        # Folder selection
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=0, column=0, sticky='ew', pady=5)
        ttk.Label(folder_frame, text="Training Images Folder:").grid(row=0, column=0, sticky='w')
        self.train_folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.train_folder_var, width=35).grid(row=0, column=1, padx=5, sticky='ew')
        ttk.Button(folder_frame, text="Browse", command=self.browse_train_folder).grid(row=0, column=2, padx=5)
        folder_frame.columnconfigure(1, weight=1)

        # Training parameters
        param_frame = ttk.Frame(main_frame)
        param_frame.grid(row=1, column=0, sticky='ew', pady=5)
        ttk.Label(param_frame, text="Epochs:").grid(row=0, column=0, sticky='e')
        self.epochs_var = tk.IntVar(value=10)
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=6).grid(row=0, column=1, padx=5)
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=2, sticky='e')
        self.batch_var = tk.IntVar(value=32)
        ttk.Entry(param_frame, textvariable=self.batch_var, width=6).grid(row=0, column=3, padx=5)
        param_frame.columnconfigure(4, weight=1)

        # Annotation file selection
        annot_frame = ttk.Frame(main_frame)
        annot_frame.grid(row=2, column=0, sticky='ew', pady=5)
        ttk.Label(annot_frame, text="Annotation File (.txt or .csv):").grid(row=0, column=0, sticky='w')
        self.annotation_file_var = tk.StringVar()
        ttk.Entry(annot_frame, textvariable=self.annotation_file_var, width=35).grid(row=0, column=1, padx=5, sticky='ew')
        ttk.Button(annot_frame, text="Browse", command=self.browse_annotation_file).grid(row=0, column=2, padx=5)
        annot_frame.columnconfigure(1, weight=1)

        # Train button
        ttk.Button(main_frame, text="Train Model", command=self.train_model).grid(row=3, column=0, pady=15, sticky='ew')

        # Frame for sample images
        self.img_frame = ttk.LabelFrame(main_frame, text="Sample Images", padding=10)
        self.img_frame.grid(row=4, column=0, pady=10, sticky='ew')

        # Console output viewer
        console_frame = ttk.LabelFrame(main_frame, text="Training Console Output", padding=5)
        console_frame.grid(row=5, column=0, sticky='nsew', pady=5)
        self.console_text = tk.Text(console_frame, height=8, wrap='word', state='disabled', bg="#222", fg="#eee")
        self.console_text.pack(fill='both', expand=True)
        main_frame.rowconfigure(5, weight=1)
        main_frame.columnconfigure(0, weight=1)

    def create_classify_tab(self):
        main_frame = ttk.Frame(self.classify_tab, padding=15)
        main_frame.pack(fill='both', expand=True)

        # Model selection
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=0, column=0, sticky='w', pady=5)
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky='w')
        self.classify_model_var = tk.StringVar(value="own")
        ttk.Radiobutton(model_frame, text="Own Model", variable=self.classify_model_var, value="own").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(model_frame, text="Pretrained", variable=self.classify_model_var, value="pretrained").grid(row=0, column=2, padx=5)

        # Image selection
        imgsel_frame = ttk.Frame(main_frame)
        imgsel_frame.grid(row=1, column=0, sticky='ew', pady=5)
        ttk.Label(imgsel_frame, text="Image to Classify:").grid(row=0, column=0, sticky='w')
        self.image_path_var = tk.StringVar()
        ttk.Entry(imgsel_frame, textvariable=self.image_path_var, width=35).grid(row=0, column=1, padx=5, sticky='ew')
        ttk.Button(imgsel_frame, text="Browse", command=self.browse_image).grid(row=0, column=2, padx=5)
        imgsel_frame.columnconfigure(1, weight=1)

        # Model file selection for own model
        self.model_path_var = tk.StringVar()
        self.model_path_entry = ttk.Entry(main_frame, textvariable=self.model_path_var, width=35)
        self.model_browse_btn = ttk.Button(main_frame, text="Browse Model", command=self.browse_model)
        self.classify_model_var.trace_add("write", self.toggle_model_path_widgets)
        self.toggle_model_path_widgets()

        # Classify button
        ttk.Button(main_frame, text="Classify", command=self.classify_image).grid(row=4, column=0, pady=15, sticky='ew')

        # Result
        self.result_var = tk.StringVar()
        ttk.Label(main_frame, textvariable=self.result_var, foreground="blue", font=("Arial", 12, "bold")).grid(row=5, column=0, pady=10, sticky='ew')

        main_frame.columnconfigure(0, weight=1)

    def toggle_model_path_widgets(self, *args):
        # Place model path widgets only if "own" model is selected
        if self.classify_model_var.get() == "own":
            self.model_path_entry.grid(row=2, column=0, sticky='ew', pady=2)
            self.model_browse_btn.grid(row=3, column=0, sticky='ew', pady=2)
        else:
            self.model_path_entry.grid_forget()
            self.model_browse_btn.grid_forget()

    def browse_train_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.train_folder_var.set(folder)

    def browse_image(self):
        file = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file:
            self.image_path_var.set(file)

    def browse_model(self):
        file = filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras")])
        if file:
            self.model_path_var.set(file)

    def browse_annotation_file(self):
        file = filedialog.askopenfilename(filetypes=[("Annotation files", "*.txt *.csv")])
        if file:
            self.annotation_file_var.set(file)
            with open(file, "r", encoding="utf-8") as f:
                self.annotation_content = f.read()

    def write_console(self, text):
        self.console_text.config(state='normal')
        self.console_text.insert('end', text + '\n')
        self.console_text.see('end')
        self.console_text.config(state='disabled')

    def train_model(self):
        folder = self.train_folder_var.get()
        epochs = self.epochs_var.get()
        batch_size = self.batch_var.get()
        annotation_file = self.annotation_file_var.get()
        if annotation_file and self.annotation_content is None:
            with open(annotation_file, "r", encoding="utf-8") as f:
                self.annotation_content = f.read()
        # Example: Simulate training output
        self.write_console("Starting training...")
        self.after(500, lambda: self.write_console("Epoch 1/{}: loss=0.5 acc=0.8".format(epochs)))
        self.after(1000, lambda: self.write_console("Epoch 2/{}: loss=0.3 acc=0.9".format(epochs)))
        self.after(1500, lambda: self.write_console("Training finished."))
        self.after(1700, lambda: self.training_complete(folder))

    def training_complete(self, folder):
        # Save model if own model
        model_save_path = os.path.join(folder, "my_model.keras")
        # Placeholder: Save your model here
        # model.save(model_save_path)
        pass
        messagebox.showinfo("Training", "Training complete!")
        self.display_sample_images(folder)

    def display_sample_images(self, folder):
        # Clear previous images
        for widget in self.img_frame.winfo_children():
            widget.destroy()
        self.sample_imgs.clear()
        img_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for i, img_name in enumerate(img_files[:3]):
            img_path = os.path.join(folder, img_name)
            img = Image.open(img_path)
            img.thumbnail((120, 120))
            img_tk = ImageTk.PhotoImage(img)
            self.sample_imgs.append(img_tk)
            lbl = ttk.Label(self.img_frame, image=img_tk)
            lbl.grid(row=0, column=i, padx=10, pady=5)

    def classify_image(self):
        image_path = self.image_path_var.get()
        model_type = self.classify_model_var.get()
        if model_type == "own":
            model_path = self.model_path_var.get()
            if not model_path.endswith(".keras"):
                self.result_var.set("Please select a valid .keras model file.")
                return
            # Placeholder: Load your keras model and classify
            # model = keras.models.load_model(model_path)
            # result = model.predict(...)
            result = "Classified as: ExampleClass (Own Model)"
        else:
            # Placeholder: Use pretrained model
            result = "Classified as: ExampleClass (Pretrained)"
        self.result_var.set(result)


if __name__ == "__main__":
    app = HouseAIApp()
    app.mainloop()
