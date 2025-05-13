import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


class HouseAIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HouseAI Model Trainer & Classifier")
        self.geometry("600x500")

        # Tabs for Train and Classify
        tab_control = ttk.Notebook(self)
        self.train_tab = ttk.Frame(tab_control)
        self.classify_tab = ttk.Frame(tab_control)
        tab_control.add(self.train_tab, text='Train Model')
        tab_control.add(self.classify_tab, text='Classify Image')
        tab_control.pack(expand=1, fill='both')

        self.create_train_tab()
        self.create_classify_tab()

        self.sample_imgs = []  # For displaying sample images

    def create_train_tab(self):
        # Folder selection
        ttk.Label(self.train_tab, text="Training Images Folder:").pack(pady=5)
        self.train_folder_var = tk.StringVar()
        ttk.Entry(self.train_tab, textvariable=self.train_folder_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.train_tab, text="Browse", command=self.browse_train_folder).pack(side=tk.LEFT)

        # Training parameters
        param_frame = ttk.Frame(self.train_tab)
        param_frame.pack(pady=10)
        ttk.Label(param_frame, text="Epochs:").grid(row=0, column=0)
        self.epochs_var = tk.IntVar(value=10)
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=5).grid(row=0, column=1)
        ttk.Label(param_frame, text="Batch Size:").grid(row=0, column=2)
        self.batch_var = tk.IntVar(value=32)
        ttk.Entry(param_frame, textvariable=self.batch_var, width=5).grid(row=0, column=3)

        # Train button
        ttk.Button(self.train_tab, text="Train Model", command=self.train_model).pack(pady=15)

        # Frame for sample images
        self.img_frame = ttk.Frame(self.train_tab)
        self.img_frame.pack(pady=10)

    def create_classify_tab(self):
        # Model selection
        ttk.Label(self.classify_tab, text="Model:").pack(pady=5)
        self.classify_model_var = tk.StringVar(value="own")
        ttk.Radiobutton(self.classify_tab, text="Own Model", variable=self.classify_model_var, value="own").pack()
        ttk.Radiobutton(self.classify_tab, text="Pretrained", variable=self.classify_model_var, value="pretrained").pack()

        # Image selection
        ttk.Label(self.classify_tab, text="Image to Classify:").pack(pady=5)
        self.image_path_var = tk.StringVar()
        ttk.Entry(self.classify_tab, textvariable=self.image_path_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.classify_tab, text="Browse", command=self.browse_image).pack(side=tk.LEFT)

        # Model file selection for own model
        self.model_path_var = tk.StringVar()
        self.model_path_entry = ttk.Entry(self.classify_tab, textvariable=self.model_path_var, width=40)
        self.model_browse_btn = ttk.Button(self.classify_tab, text="Browse Model", command=self.browse_model)
        # Show/hide model path widgets based on model selection
        self.classify_model_var.trace_add("write", self.toggle_model_path_widgets)
        self.toggle_model_path_widgets()

        # Classify button
        ttk.Button(self.classify_tab, text="Classify", command=self.classify_image).pack(pady=15)

        # Result
        self.result_var = tk.StringVar()
        ttk.Label(self.classify_tab, textvariable=self.result_var, foreground="blue").pack(pady=10)

    def toggle_model_path_widgets(self, *args):
        if self.classify_model_var.get() == "own":
            self.model_path_entry.pack(side=tk.TOP, padx=5, pady=2)
            self.model_browse_btn.pack(side=tk.TOP, padx=5, pady=2)
        else:
            self.model_path_entry.pack_forget()
            self.model_browse_btn.pack_forget()

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

    def train_model(self):
        folder = self.train_folder_var.get()
        epochs = self.epochs_var.get()
        batch_size = self.batch_var.get()
        # Placeholder: Insert your model training logic here
        # Simulate training delay
        self.after(1000, lambda: self.training_complete(folder))

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
        # Get up to 3 image files from the folder
        img_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for i, img_name in enumerate(img_files[:3]):
            img_path = os.path.join(folder, img_name)
            img = Image.open(img_path)
            img.thumbnail((100, 100))
            img_tk = ImageTk.PhotoImage(img)
            self.sample_imgs.append(img_tk)  # Keep reference
            lbl = ttk.Label(self.img_frame, image=img_tk)
            lbl.grid(row=0, column=i, padx=5)

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
