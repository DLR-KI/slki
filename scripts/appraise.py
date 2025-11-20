#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#
# type: ignore
# flake8: noqa
#
# Requirements:
#   pip install PyQt6
#
from dlr.ki.logging import load_default
from logging import getLogger
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import os
import pickle
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QToolButton, QButtonGroup, QMessageBox
from slki.config import TNorm
from slki.data.data_import import load_all_typed
from slki.utils.debug import ensure_deterministic
import sys
from typing import Dict


# ### USER SETTINGS ###################################################################################################

raw_data: bool = False  # raw data vs. processed data
resample_size: int | None = 1000  # resample the data to N points per signal
normalize: TNorm | None = "mone_one_zero_fix"  # normalize the data to [-1, 1], but keep the zero point fixed at 0.0
# Hint: raw and not resampled data requires a lot of memory!

dataset_file_directory: str = "/home/lab/slki/Dataset/DataFromMarkus"
dataset_type: str = "points-kionix-sh-x-" + ("raw" if raw_data else "dt")
datasets_filename: str = f"GueterOct24-{dataset_type}%(chunk?).pkl"  # f"gueterzuege-{dataset_type}%(chunk?).pkl"
datasets_label: str = "Güterzüge"

# #####################################################################################################################

# initialize
load_default("../logs/notebooks.log")
logger = getLogger("slki.appraise")
ensure_deterministic(42)

# load dataset
dataset = load_all_typed(
    os.path.join(dataset_file_directory, datasets_filename),
    resample_size=resample_size,
    normalize=normalize,
)
dataset_len = len(dataset)
logger.info(f"Loaded dataset '{datasets_label}' with {dataset_len} samples.")

# prepare apprasing
mask_file_name = datasets_filename.replace("%(chunk?)", "")[:-4] + "-mask.pkl"
mask_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", mask_file_name)
os.makedirs(os.path.dirname(mask_path), exist_ok=True)
mask: dict[int, int] = {}
if os.path.exists(mask_path):
    with open(mask_path, "rb") as f:
        mask = pickle.load(f)
    logger.info(f"{len(mask)}/{dataset_len} already appraised samples found.")
    if len(mask) >= dataset_len:
        logger.warning("Nothing to appraised anymore.")
        exit()

# ### GUI #############################################################################################################

idx = len(mask)
data_items = list(iter(dataset))
fig, _ = plt.subplots(figsize=(10, 6))


class CustomToolbar(NavigationToolbar):
    def __init__(self, canvas, parent=None, coordinates=True):
        super().__init__(canvas, parent)
        self.btn_group = QButtonGroup(self)
        self.btn_group.buttonClicked.connect(self._on_click)
        self.btn_good = QToolButton(self)
        self.btn_good.setStyleSheet("color: green;")
        self.btn_good.setCheckable(True)
        self.btn_good.setText("✔")
        self.addWidget(self.btn_good)
        self.btn_group.addButton(self.btn_good)
        self.btn_bad = QToolButton(self)
        self.btn_bad.setStyleSheet("color: red;")
        self.btn_bad.setCheckable(True)
        self.btn_bad.setText("✘")
        self.addWidget(self.btn_bad)
        self.btn_group.addButton(self.btn_bad)
        self.btn_skip = QToolButton(self)
        self.btn_skip.setCheckable(True)
        self.btn_skip.setText("skip")
        self.addWidget(self.btn_skip)
        self.btn_group.addButton(self.btn_skip)

    def _on_click(self, btn):
        text = btn.text()
        if text == "✔":
            mask[idx] = 1
        if text == "✘":
            mask[idx] = 0
        if text == "skip":
            mask[idx] = -1
        logger.info(f"{datasets_label}: {idx + 1}/{dataset_len} set to {text}")
        self.forward()

    def save_figure(self, *args):
        with open(mask_path, "wb") as f:
            pickle.dump(mask, f)
        logger.info("Mask saved to {mask_path}")

    def home(self, *args):
        global idx
        idx = len(mask)
        return super().home(*args)

    def back(self, *args):
        global idx
        idx -= 1
        return super().back(*args)

    def forward(self, *args):
        global idx
        idx += 1
        return super().forward(*args)

    def set_history_buttons(self):
        can_backward = idx > 0
        can_forward = idx < dataset_len - 1
        if "back" in self._actions:
            self._actions["back"].setEnabled(can_backward)
        if "forward" in self._actions:
            self._actions["forward"].setEnabled(can_forward)

    def _update_view(self):
        fig.canvas.figure.clear(True)
        ax = self.canvas.figure.add_subplot(1, 1, 1)
        ax.plot(data_items[idx].data)
        ax.set_title(f"{datasets_label}: {idx + 1}/{dataset_len}")
        self.canvas.figure.canvas.draw()
        self._update_buttons()

    def _update_buttons(self):
        n = mask.get(idx, -2)
        self.btn_group.setExclusive(False)
        self.btn_good.setChecked(False)
        self.btn_bad.setChecked(False)
        self.btn_skip.setChecked(False)
        if n == 1:
            self.btn_good.setChecked(True)
        elif n == 0:
            self.btn_bad.setChecked(True)
        elif n == -1:
            self.btn_skip.setChecked(True)
        self.btn_group.setExclusive(True)


# Create a QApplication and QMainWindow for the custom toolbar
apps = QApplication(sys.argv)
main_window = QMainWindow()
main_window.setWindowTitle("SLKI Appraise")
main_window.setGeometry(100, 100, 800, 600)  # Set window size
# Create the central widget and layout
central_widget = QWidget(main_window)
layout = QVBoxLayout(central_widget)
# Create a custom toolbar with the "Graph Type" drop-down menu
canvas = FigureCanvas(fig)
custom_toolbar = CustomToolbar(canvas, main_window)
custom_toolbar._update_view()
layout.addWidget(canvas)
layout.addWidget(custom_toolbar)


# Ask user to save on close
def closeEvent(event):
    confirmation = QMessageBox.question(
        main_window,
        "Confirmation",
        "Do you want to save before closing the application?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
    )
    if confirmation == QMessageBox.StandardButton.Yes:
        custom_toolbar.save_figure()
        event.accept()
    if confirmation == QMessageBox.StandardButton.No:
        event.accept()
    if confirmation == QMessageBox.StandardButton.Cancel:
        event.ignore()  # Don't close the app


main_window.closeEvent = closeEvent
# Set the central widget and show the main window
main_window.setCentralWidget(central_widget)
main_window.show()
# Run the Qt event loop
sys.exit(apps.exec())
