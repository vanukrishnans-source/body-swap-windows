"""Dark theme for Ally X: 7″ 1080p @ ~150% scaling. Tap targets ≥ 56 logical-px.

Explicit Windows font stack (Segoe UI Variable → Segoe UI → Tahoma) avoids tofu /
missing-glyph boxes seen when Qt falls back to a CJK-only or empty font on
headless CI and some handheld images.
"""

DARK_QSS = """
* {
  font-family: "Segoe UI Variable", "Segoe UI", "Segoe UI Semibold", Tahoma, "Noto Sans", sans-serif;
}
QWidget { background:#0e1117; color:#e8eaed; font-size:16px; }
QMainWindow, QDialog { background:#0e1117; }
QLabel { background:transparent; color:#e8eaed; font-size:16px; }
QLabel#title { font-size:26px; font-weight:700; color:#ffffff; letter-spacing:0.2px; }
QLabel#subtitle { font-size:15px; color:#b0b6bf; }
QLabel#hint { font-size:14px; color:#8b929c; }
QLabel#section { font-size:18px; font-weight:600; color:#ffffff; }
QLabel#stepnum {
  font-size:18px; font-weight:700; color:#041014; background:#00b8a9;
  border-radius:18px; min-width:36px; max-width:36px; min-height:36px; max-height:36px;
  qproperty-alignment: AlignCenter;
}
QLabel#steplabel { font-size:18px; font-weight:600; color:#ffffff; }
QPushButton {
  background:#232833; color:#e8eaed; border:2px solid #3a4252; border-radius:14px;
  padding:16px 22px; min-height:52px; font-size:17px; font-weight:600;
}
QPushButton:hover { background:#2c3340; border-color:#4a5568; }
QPushButton:pressed { background:#1a1e27; }
QPushButton:disabled { color:#5f6368; background:#161a22; border-color:#2a2f3a; }
QPushButton#primary {
  background:#00b8a9; color:#041014; border:none; font-weight:700; font-size:18px; min-height:56px;
}
QPushButton#primary:hover { background:#1ad1c1; }
QPushButton#primary:disabled { background:#1a4a45; color:#6a8a86; }
QPushButton#danger { background:#c5221f; color:#fff; border:none; min-height:52px; }
QPushButton#danger:hover { background:#e33b38; }
QPushButton#mode {
  background:#1a1e27; border:2px solid #3a4252; border-radius:14px;
  font-size:18px; font-weight:700; min-height:64px;
}
QPushButton#mode:checked {
  background:#00b8a9; color:#041014; border:none;
}
QPushButton#ghost {
  background:transparent; border:2px dashed #3a4254; color:#9aa0a6; min-height:72px; font-size:16px;
}
QFrame#card { background:#161a22; border:1px solid #2a3140; border-radius:16px; }
QFrame#stepcard { background:#161a22; border:2px solid #2a3140; border-radius:18px; }
QFrame#stepcard[active="true"] { border:2px solid #00b8a9; }
QProgressBar {
  background:#161a22; border:1px solid #2a3140; border-radius:10px; text-align:center;
  min-height:28px; color:#e8eaed; font-size:14px;
}
QProgressBar::chunk { background:#00b8a9; border-radius:9px; }
QSlider::groove:horizontal { height:10px; background:#2a3140; border-radius:5px; }
QSlider::handle:horizontal {
  width:32px; height:32px; margin:-11px 0; background:#00b8a9; border-radius:16px;
}
QCheckBox { spacing:14px; font-size:16px; min-height:44px; }
QCheckBox::indicator { width:28px; height:28px; }
QScrollArea { border:none; background:transparent; }
QMessageBox { background:#161a22; }
QMessageBox QLabel { font-size:16px; }
"""
