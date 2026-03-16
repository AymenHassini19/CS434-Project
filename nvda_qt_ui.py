# nvda_qt_ui.py
import sys
import os
import io
import re
import traceback
import webbrowser
import importlib.util
from html import escape as html_escape
from contextlib import redirect_stdout

import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QLineEdit, QTextEdit,
    QTabWidget, QTableView, QHeaderView
)

# ------------------ load user's initialproject.py ------------------
SPEC_PATH = "initialproject.py"
spec = importlib.util.spec_from_file_location("initialproject", SPEC_PATH)
initialproject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initialproject)

# ------------------ simple pandas -> Qt model ------------------
class PandasModel(QAbstractTableModel):
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self.df = df.copy()

    def update(self, df: pd.DataFrame):
        self.beginResetModel()
        self.df = df.copy()
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.df.index)

    def columnCount(self, parent=QModelIndex()):
        return len(self.df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            val = self.df.iloc[index.row(), index.column()]
            return str(val)
        return None

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self.df.columns[section])
        return str(self.df.index[section])

# ------------------ ML worker thread ------------------
class MLThread(QThread):
    finished_signal = pyqtSignal(object, str)

    def __init__(self, hours: int):
        super().__init__()
        self.hours = hours

    def run(self):
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                # Load and prepare data using user's functions
                df = initialproject.load_nvda_data("NVDA_hourly_last_2_years.csv")
                df = initialproject.add_moving_averages(df)

                # Run the pipeline (this should print pipeline + evaluation + feature importance)
                model, forecast = initialproject.run_ml_pipeline_and_forecast(
                    df,
                    hours_to_predict=self.hours
                )

                # Some scripts print the forecast; replicate that to be safe
                print("\nForecast:")
                print(forecast)

            txt = buf.getvalue()
            self.finished_signal.emit(forecast, txt)

        except Exception:
            txt = buf.getvalue() + "\n\n" + traceback.format_exc()
            self.finished_signal.emit(None, txt)

# ------------------ helper: split ML output into pipeline / evaluation ------------------
def split_pipeline_and_evaluation(full_text: str):
    """
    Improved heuristic split:
    - Find the line that contains the evaluation header (case-insensitive, e.g. 'EVALUATION')
    - Evaluation section = from that header line up to (but not including) the next big section header
      (e.g. lines containing 'TOP FEATURE', 'FEATURE IMPORTANCE', 'FORECAST', or a line of '=' chars).
    - Everything before the evaluation header is ML Pipeline.
    - If no evaluation header is found, return (full_text, "").
    """
    lines = full_text.splitlines(True)
    eval_idx = None
    for i, ln in enumerate(lines):
        if re.search(r'\bevaluation\b', ln, flags=re.I):
            eval_idx = i
            break

    if eval_idx is None:
        # try looser matches
        for i, ln in enumerate(lines):
            if re.search(r'\bmodel eval|evaluation metrics|eval\b', ln, flags=re.I):
                eval_idx = i
                break

    if eval_idx is None:
        return full_text, ""

    # find end of evaluation section
    end_idx = None
    for j in range(eval_idx + 1, len(lines)):
        ln = lines[j]
        # strong separators or next known section headers
        if re.search(r'={3,}', ln):  # lines with many '='
            end_idx = j
            break
        if re.search(r'\bTOP FEATURE IMPORTANCE\b', ln, flags=re.I):
            end_idx = j
            break
        if re.search(r'\bFEATURE IMPORTANCE\b', ln, flags=re.I):
            end_idx = j
            break
        if re.search(r'\bFORECAST(ING)?\b', ln, flags=re.I):
            end_idx = j
            break
        # detect an uppercase header line (short and mostly uppercase words)
        stripped = ln.strip()
        if 1 < len(stripped) <= 80 and re.match(r'^[A-Z0-9 \-_/]{3,}$', stripped) and stripped.isupper():
            end_idx = j
            break

    if end_idx is None:
        end_idx = len(lines)

    pipeline_text = "".join(lines[:eval_idx])
    evaluation_text = "".join(lines[eval_idx:end_idx])
    return pipeline_text, evaluation_text

# ------------------ helper: format text as beautiful HTML for QTextEdit ------------------
def make_html_block(title: str, body_text: str):

    import re
    from html import escape as html_escape

    # ⭐ remove the ===== HEADERS ===== lines
    body_text = re.sub(r'=+\s*ML\s*PIPELINE\s*=+', '', body_text, flags=re.I)
    body_text = re.sub(r'=+\s*EVALUATION\s*=+', '', body_text, flags=re.I)

    safe_body = html_escape(body_text.strip())

    html = f"""
    <html>
      <body style="
            background-color:white;
            color:#111;
            font-family:Segoe UI, Arial;
            ">

        <div style="padding:20px;">

          <h1 style="
                font-size:28px;
                color:#76b900;
                margin-bottom:15px;
                ">
                {html_escape(title)}
          </h1>

          <div style="
                background:#f4f6f8;
                border-radius:10px;
                padding:20px;
                box-shadow:0 3px 10px rgba(0,0,0,0.08);
                ">

            <pre style="
                    font-size:18px;
                    line-height:1.6;
                    font-family:Consolas, monospace;
                    white-space:pre-wrap;
                    ">
{safe_body}
            </pre>

          </div>

        </div>

      </body>
    </html>
    """

    return html

# ------------------ Main UI ------------------
class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NVDA Forecast Dashboard")
        self.resize(1300, 840)

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        # LEFT: title, logo, controls
        left_col = QVBoxLayout()
        title = QLabel("NVDA Forecast Dashboard")
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)

        logo = QLabel()
        logo.setAlignment(Qt.AlignCenter)
        if os.path.exists("nvidia_logo.png"):
            pix = QPixmap("nvidia_logo.png").scaledToHeight(90, Qt.SmoothTransformation)
            logo.setPixmap(pix)
        else:
            logo.setText("NVIDIA")
            smallf = QFont()
            smallf.setPointSize(16)
            smallf.setBold(True)
            logo.setFont(smallf)

        # Buttons & input
        self.open_chart_btn = QPushButton("Open Candlestick Chart (browser)")
        self.open_chart_btn.clicked.connect(self.open_candle_chart)

        controls = QHBoxLayout()
        self.hours_input = QLineEdit()
        self.hours_input.setPlaceholderText("Hours to predict (e.g. 24)")
        self.predict_btn = QPushButton("Predict")
        self.predict_btn.clicked.connect(self.start_prediction)
        self.status_lbl = QLabel("Idle")
        controls.addWidget(self.hours_input)
        controls.addWidget(self.predict_btn)
        controls.addWidget(self.status_lbl)

        left_col.addWidget(title)
        left_col.addWidget(logo)
        left_col.addWidget(self.open_chart_btn)
        left_col.addLayout(controls)
        left_col.addStretch()
        main_layout.addLayout(left_col, 2)

        # RIGHT: tabbed area
        right_col = QVBoxLayout()
        self.tabs = QTabWidget()

        # Predictions table tab
        self.pred_table = QTableView()
        self.pred_model = PandasModel(pd.DataFrame())
        self.pred_table.setModel(self.pred_model)
        self.pred_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.pred_table, "Predictions")

        # ML Pipeline tab (rich formatted)
        self.pipeline_text = QTextEdit()
        self.pipeline_text.setReadOnly(True)
        self.tabs.addTab(self.pipeline_text, "ML Pipeline")

        # Evaluation tab (rich formatted)
        self.eval_text = QTextEdit()
        self.eval_text.setReadOnly(True)
        self.tabs.addTab(self.eval_text, "Evaluation")

        # Top Features tab (table)
        self.feat_table = QTableView()
        self.feat_model = PandasModel(pd.DataFrame())
        self.feat_table.setModel(self.feat_model)
        self.feat_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.feat_table, "Top Features")

        right_col.addWidget(self.tabs)
        main_layout.addLayout(right_col, 3)

    # ----------------- open candle chart in browser (uses user's function) -----------------
    def open_candle_chart(self):
        try:
            self.status_lbl.setText("Generating candle chart...")
            df = initialproject.load_nvda_data("NVDA_hourly_last_2_years.csv")
            df = initialproject.add_moving_averages(df)

            html_name = "chart.html"
            # call user's function (allowing it to write the HTML and show)
            try:
                # If their function accepts save_path keyword, prefer to pass it
                initialproject.plot_and_save_chart(df, save_path=html_name)
            except TypeError:
                # fallback - call with single argument
                initialproject.plot_and_save_chart(df)

            # open saved html if exists, otherwise let user's fig.show() handle it
            if os.path.exists(html_name):
                webbrowser.open(os.path.abspath(html_name))
                self.status_lbl.setText("Opened chart in browser")
            else:
                # some implementations call fig.show() which opens a browser window; we reflect that status
                self.status_lbl.setText("Chart generation requested (check browser/window)")

        except Exception as e:
            self.status_lbl.setText("Chart error")
            tb = traceback.format_exc()
            self.pipeline_text.setHtml(make_html_block("Chart generation error", tb))

    # ----------------- start ML in background -----------------
    def start_prediction(self):
        try:
            hours = int(self.hours_input.text().strip())
            if hours <= 0:
                raise ValueError
        except Exception:
            hours = 24

        self.status_lbl.setText("Running ML pipeline...")
        self.predict_btn.setEnabled(False)
        self.thread = MLThread(hours)
        self.thread.finished_signal.connect(self.ml_finished)
        self.thread.start()

    # ----------------- when ML finishes -----------------
    def ml_finished(self, forecast_df, full_text):
        # restore UI
        self.status_lbl.setText("Done")
        self.predict_btn.setEnabled(True)

        # Split into pipeline vs evaluation (improved)
        pipeline_part, evaluation_part = split_pipeline_and_evaluation(full_text)

        # Format and set HTML in the two tabs
        self.pipeline_text.setHtml(make_html_block("ML Pipeline", pipeline_part if pipeline_part.strip() else "(no pipeline output found)"))
        self.eval_text.setHtml(make_html_block("Evaluation", evaluation_part if evaluation_part.strip() else "(no evaluation output found)"))

        # Update predictions table (if forecast_df is a DataFrame)
        if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
            display_df = forecast_df.reset_index()
            # ensure datetime columns are strings for table display
            for col in display_df.columns:
                if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                    display_df[col] = display_df[col].astype(str)
            self.pred_model.update(display_df)
        else:
            self.pred_model.update(pd.DataFrame())

        # Parse and show top feature importance (heuristic)
        feat_df = self._parse_feature_importance(full_text)
        if not feat_df.empty:
            self.feat_model.update(feat_df)
        else:
            self.feat_model.update(pd.DataFrame())

        # Call user's original plotting function to show Actual vs Predicted (matplotlib)
        try:
            if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
                fig_df = forecast_df.reset_index()
                # If the user's function expects 'Datetime' and 'Close' columns, ensure they exist
                if "PredictedClose" in fig_df.columns and "Close" not in fig_df.columns:
                    fig_df = fig_df.rename(columns={"PredictedClose": "Close"})
                # call their plotting routine (name from earlier conversation: plot_forecast_close)
                try:
                    initialproject.plot_forecast_close(fig_df)
                except TypeError:
                    # fallback: call with single dataframe if signature differs
                    initialproject.plot_forecast_close(fig_df)
        except Exception:
            # If plotting fails, append error to pipeline tab
            tb = traceback.format_exc()
            prev = pipeline_part + "\n\n--- Plotting exception ---\n" + tb
            self.pipeline_text.setHtml(make_html_block("ML Pipeline (with plotting error)", prev))

    # ----------------- parse top feature importance -----------------
    def _parse_feature_importance(self, text: str) -> pd.DataFrame:
        """
        Look for lines like:
            feature_name   -> 0.1234
        or 'Feature importance:' blocks and parse numeric values.
        """
        lines = text.splitlines()
        data = []
        # Try to find a header indicating feature importance, but also parse any '->' lines globally.
        for ln in lines:
            if '->' in ln:
                parts = ln.split('->')
                if len(parts) >= 2:
                    name = parts[0].strip().strip(':')
                    right = parts[1].strip()
                    # try to extract a float from right
                    m = re.search(r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?\d+)?', right)
                    if m:
                        try:
                            val = float(m.group(0))
                            data.append((name, val))
                        except Exception:
                            pass
        # If no '->' lines found, look for common "feature: value" patterns
        if not data:
            for ln in lines:
                m = re.match(r'\s*([\w_\-\. ]{2,50})\s*[:=]\s*([-+]?[0-9]*\.?[0-9]+([eE][-+]?\d+)?)', ln)
                if m:
                    name = m.group(1).strip()
                    val = float(m.group(2))
                    data.append((name, val))

        if not data:
            return pd.DataFrame()
        # Keep top 30 if too many
        df = pd.DataFrame(data, columns=["Feature", "Importance"])
        df = df.drop_duplicates(subset=["Feature"]).sort_values("Importance", ascending=False).reset_index(drop=True)
        return df.head(30)

# ------------------ run app ------------------
def main():
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()