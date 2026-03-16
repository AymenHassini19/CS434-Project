import sys
import os
import io
import traceback
import importlib.util
from contextlib import redirect_stdout

import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit,
    QTabWidget, QTableView, QHeaderView
)

# ================= LOAD initialproject =================
spec = importlib.util.spec_from_file_location(
    "initialproject",
    "initialproject.py"
)
initialproject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initialproject)


# ================= TABLE MODEL =================
class PandasModel(QAbstractTableModel):

    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self.df = df

    def update(self, df):
        self.beginResetModel()
        self.df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            return str(self.df.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.df.columns[section]
            return str(section)
        return None


# ================= ML THREAD =================
class MLThread(QThread):

    finished_signal = pyqtSignal(object, str)

    def __init__(self, hours):
        super().__init__()
        self.hours = hours

    def run(self):

        buffer = io.StringIO()

        try:
            with redirect_stdout(buffer):

                df = initialproject.load_nvda_data(
                    "NVDA_hourly_last_2_years.csv"
                )

                df = initialproject.add_moving_averages(df)

                model, forecast = initialproject.run_ml_pipeline_and_forecast(
                    df,
                    hours_to_predict=self.hours
                )

                print("\nForecast:")
                print(forecast)

            text = buffer.getvalue()
            self.finished_signal.emit(forecast, text)

        except Exception:
            text = buffer.getvalue() + "\n" + traceback.format_exc()
            self.finished_signal.emit(None, text)


# ================= MAIN WINDOW =================
class Window(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("NVDA Forecast Dashboard")
        self.resize(1300, 850)

        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout()
        root.setLayout(layout)

        # ===== LEFT PANEL =====
        left = QVBoxLayout()

        title = QLabel("NVDA Forecast Dashboard")
        f = QFont()
        f.setPointSize(24)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignCenter)

        logo = QLabel()
        if os.path.exists("nvidia_logo.png"):
            pix = QPixmap("nvidia_logo.png").scaledToHeight(90)
            logo.setPixmap(pix)
        else:
            logo.setText("NVIDIA")
        logo.setAlignment(Qt.AlignCenter)

        self.chart_btn = QPushButton("Open Candlestick Chart")
        self.chart_btn.clicked.connect(self.open_candle_chart)

        row = QHBoxLayout()
        self.hours = QLineEdit()
        self.hours.setPlaceholderText("Hours to predict")

        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.start_ml)

        self.status = QLabel("Idle")

        row.addWidget(self.hours)
        row.addWidget(self.predict_btn)
        row.addWidget(self.status)

        left.addWidget(title)
        left.addWidget(logo)
        left.addWidget(self.chart_btn)
        left.addLayout(row)

        layout.addLayout(left, 2)

        # ===== RIGHT PANEL =====
        right = QVBoxLayout()

        self.tabs = QTabWidget()

        self.table = QTableView()
        self.model = PandasModel()
        self.table.setModel(self.model)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.table, "Predictions")

        self.console = QTextEdit()
        self.tabs.addTab(self.console, "ML Output")

        self.feature_table = QTableView()
        self.feature_model = PandasModel()
        self.feature_table.setModel(self.feature_model)
        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.feature_table, "Top Features")

        right.addWidget(self.tabs)
        layout.addLayout(right, 3)

    # ================= OPEN CANDLE CHART =================
    def open_candle_chart(self):

        try:
            self.status.setText("Generating chart...")

            df = initialproject.load_nvda_data(
                "NVDA_hourly_last_2_years.csv"
            )
            df = initialproject.add_moving_averages(df)

            initialproject.plot_and_save_chart(df)

            self.status.setText("Chart opened")

        except Exception as e:
            self.console.setText(str(e))
            self.status.setText("Chart error")

    # ================= START ML =================
    def start_ml(self):

        try:
            h = int(self.hours.text())
        except:
            h = 24

        self.status.setText("Running ML...")

        self.thread = MLThread(h)
        self.thread.finished_signal.connect(self.ml_finished)
        self.thread.start()

    # ================= ML FINISHED =================
    def ml_finished(self, forecast, text):

        self.status.setText("Done")
        self.console.setText(text)

        if forecast is None:
            return

        df = forecast.reset_index()
        self.model.update(df)

        # ⭐ USE YOUR ORIGINAL PLOTTING FUNCTION
        forecast_plot_df = forecast.reset_index()

        if "PredictedClose" in forecast_plot_df.columns:
            forecast_plot_df.rename(
                columns={"PredictedClose": "Close"},
                inplace=True
            )

        initialproject.plot_forecast_close(forecast_plot_df)

        self.parse_features(text)

    # ================= FEATURE IMPORTANCE =================
    def parse_features(self, text):

        lines = text.splitlines()
        data = []

        for l in lines:
            if "->" in l:
                try:
                    name, val = l.split("->")
                    data.append((name.strip(), float(val)))
                except:
                    pass

        if data:
            df = pd.DataFrame(data, columns=["Feature", "Importance"])
            self.feature_model.update(df)


# ================= RUN =================
app = QApplication(sys.argv)
w = Window()
w.show()
sys.exit(app.exec_())