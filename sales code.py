import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SalesPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Predictor")
        
        self.load_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_button.pack()

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack()

        self.date_label = tk.Label(root, text="Select Date:")
        self.date_label.pack()

        self.date_entry = tk.Entry(root)
        self.date_entry.pack()

        self.product_label = tk.Label(root, text="Select Product:")
        self.product_label.pack()

        self.product_combobox = ttk.Combobox(root)
        self.product_combobox.pack()

        self.predict_button = tk.Button(root, text="Predict Sales", command=self.predict_sales)
        self.predict_button.pack()

        self.result_label = tk.Label(root, text="Predicted Sales:")
        self.result_label.pack()

        self.result_text = tk.Text(root, height=2, width=30)
        self.result_text.pack()
        
        self.graph_button = tk.Button(root, text="Show in Bar Graph", command=self.show_bar_graph)
        self.graph_button.pack()

    def load_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_csv(file_path)
            products = self.data['Product'].unique().tolist()
            self.product_combobox['values'] = products
            messagebox.showinfo("Data Loaded", "Data loaded successfully!")

    def train_model(self):
        self.data['Date'] = pd.to_datetime(self.data['Date']).map(pd.Timestamp.toordinal)
        X = self.data[['Date', 'Product']]
        X = pd.get_dummies(X, columns=['Product'], drop_first=True)
        y = self.data['Sales']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)
        
        train_pred = self.model.predict(self.X_train)
        mse = mean_squared_error(self.y_train, train_pred)
        messagebox.showinfo("Model Trained", f"Model trained with MSE: {mse:.2f}")

    def predict_sales(self):
        date = pd.to_datetime(self.date_entry.get()).toordinal()
        product = self.product_combobox.get()
        
        input_data = {'Date': [date]}
        for col in self.X_train.columns[1:]:
            input_data[col] = [1 if product in col else 0]
        
        input_df = pd.DataFrame(input_data)
        prediction = self.model.predict(input_df)[0]
        
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"{prediction:.2f}")

    def show_bar_graph(self):
        if not hasattr(self, 'data'):
            messagebox.showerror("Error", "No data loaded!")
            return
        
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        self.data.groupby('Product')['Sales'].sum().plot(kind='bar', ax=ax)
        
        ax.set_xlabel('Product')
        ax.set_ylabel('Total Sales')
        ax.set_title('Total Sales by Product')
        
        plt.tight_layout()

        # Embed the plot into Tkinter
        self.canvas = FigureCanvasTkAgg(plt.gcf(), master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = SalesPredictorApp(root)
    root.mainloop()
