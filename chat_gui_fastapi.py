# # # import sys
# # # import time
# # # from pathlib import Path
# # # import requests
# # #
# # # from PyQt5.QtWidgets import (
# # #     QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
# # #     QScrollArea, QLabel, QFrame, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy
# # # )
# # # from PyQt5.QtCore import Qt, QPropertyAnimation, QUrl
# # # from PyQt5.QtGui import QFont, QPixmap
# # #
# # # # -----------------------------
# # # # FastAPI Backend URL
# # # # -----------------------------
# # # FASTAPI_URL = "http://127.0.0.1:8000/predict-smote"  # Change if deployed remotely
# # #
# # # # -----------------------------
# # # # Spam Classification via API
# # # # -----------------------------
# # # def is_spam(text: str, sender_is_user: bool):
# # #     """
# # #     Calls FastAPI backend to classify text as spam or ham.
# # #     Returns: (is_spam: bool, score: float, reasons: list[str])
# # #     """
# # #     if not text.strip():
# # #         return False, 0.0, []
# # #
# # #     try:
# # #         response = requests.post(FASTAPI_URL, json={"message": text}, timeout=5)
# # #         data = response.json()
# # #         label = data.get("prediction", "ham").lower()
# # #         confidence = data.get("confidence", 0.0)
# # #         # Optional: extract “reasons” locally (top spammy words)
# # #         spammy_terms = ["win", "free", "offer", "click", "link"]
# # #         tokens = [t.lower() for t in text.split()]
# # #         reasons = [t for t in tokens if t in spammy_terms][:5]
# # #
# # #         return label == "spam", float(confidence), reasons
# # #
# # #     except Exception as e:
# # #         print("Error calling FastAPI:", e)
# # #         return False, 0.0, []
# # #
# # # # -----------------------------
# # # # UI Components
# # # # -----------------------------
# # # class WarningBanner(QFrame):
# # #     def __init__(self, text: str, strong: bool = False):
# # #         super().__init__()
# # #         self.setFrameShape(QFrame.NoFrame)
# # #         self.setStyleSheet(
# # #             "QFrame {"
# # #             f"background-color: {'#FEE2E2' if strong else '#FEF3C7'};"
# # #             "border-radius: 8px;"
# # #             "border: 1px solid #FCA5A5; }"
# # #         )
# # #         lay = QHBoxLayout(self)
# # #         lay.setContentsMargins(10, 6, 10, 6)
# # #         lay.setSpacing(8)
# # #         icon = QLabel("⚠️")
# # #         icon.setFont(QFont("Arial", 11))
# # #         lbl = QLabel(text)
# # #         lbl.setWordWrap(True)
# # #         lbl.setFont(QFont("Arial", 10))
# # #         lay.addWidget(icon)
# # #         lay.addWidget(lbl)
# # #         lay.addStretch(1)
# # #
# # # class ChatBubble(QFrame):
# # #     def __init__(self, sender: str, text: str = None, timestamp: str = None,
# # #                  is_user: bool = False, is_spam: bool = False,
# # #                  attachment_path: str = None):
# # #         super().__init__()
# # #         self.setFrameShape(QFrame.NoFrame)
# # #         base_bg = "#DCF8C6" if is_user else "#ECECEC"
# # #         border = "#FCA5A5" if is_spam else "transparent"
# # #         self.setStyleSheet(
# # #             "QFrame {"
# # #             f"background-color: {base_bg};"
# # #             "border-radius: 12px;"
# # #             f"border: 1px solid {border}; }}"
# # #         )
# # #         outer = QVBoxLayout(self)
# # #         outer.setContentsMargins(10, 6, 10, 6)
# # #         outer.setSpacing(4)
# # #
# # #         sender_lbl = QLabel(sender)
# # #         sender_lbl.setFont(QFont("Arial", 9, QFont.Bold))
# # #         sender_lbl.setStyleSheet("color: #4B5563;")
# # #         if is_user:
# # #             sender_lbl.hide()
# # #         outer.addWidget(sender_lbl)
# # #
# # #         if attachment_path:
# # #             ext = Path(attachment_path).suffix.lower()
# # #             if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
# # #                 img_lbl = QLabel()
# # #                 pix = QPixmap(attachment_path)
# # #                 if not pix.isNull():
# # #                     pix = pix.scaledToWidth(220, Qt.SmoothTransformation)
# # #                     img_lbl.setPixmap(pix)
# # #                 else:
# # #                     img_lbl.setText("(image failed to load)")
# # #                 outer.addWidget(img_lbl)
# # #                 if text:
# # #                     msg_lbl = QLabel(text)
# # #                     msg_lbl.setWordWrap(True)
# # #                     msg_lbl.setFont(QFont("Arial", 11))
# # #                     msg_lbl.setStyleSheet("QLabel { border: none; }")
# # #                     outer.addWidget(msg_lbl)
# # #             else:
# # #                 link_lbl = QLabel(f'<a href="{QUrl.fromLocalFile(attachment_path).toString()}">{Path(attachment_path).name}</a>')
# # #                 link_lbl.setOpenExternalLinks(True)
# # #                 link_lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
# # #                 link_lbl.setFont(QFont("Arial", 11))
# # #                 outer.addWidget(link_lbl)
# # #                 if text:
# # #                     msg_lbl = QLabel(text)
# # #                     msg_lbl.setWordWrap(True)
# # #                     msg_lbl.setFont(QFont("Arial", 11))
# # #                     outer.addWidget(msg_lbl)
# # #         else:
# # #             msg_lbl = QLabel(text or "")
# # #             msg_lbl.setWordWrap(True)
# # #             msg_lbl.setFont(QFont("Arial", 11))
# # #             msg_lbl.setStyleSheet("QLabel { border: none; }")
# # #             outer.addWidget(msg_lbl)
# # #
# # #         ts = timestamp or time.strftime("%H:%M")
# # #         time_lbl = QLabel(ts)
# # #         time_lbl.setFont(QFont("Arial", 8))
# # #         time_lbl.setStyleSheet("color: grey;")
# # #         time_lbl.setAlignment(Qt.AlignRight)
# # #         outer.addWidget(time_lbl)
# # #
# # # class ChatWindow(QWidget):
# # #     def __init__(self):
# # #         super().__init__()
# # #         self.setWindowTitle("📩 SMS Chat — FastAPI Backend")
# # #         self.resize(460, 680)
# # #         self.setStyleSheet("background-color: white;")
# # #         self.known_contacts = {"Alice", "Bob"}
# # #         self.current_contact = "Unknown +91-9284-706-XXX"
# # #         self.has_seen_message_from_contact = set()
# # #         root = QVBoxLayout(self)
# # #         root.setSpacing(0)
# # #         root.setContentsMargins(0, 0, 0, 0)
# # #
# # #         top_bar = QHBoxLayout()
# # #         top = QFrame()
# # #         top.setStyleSheet("QFrame { background-color: #075E54; }")
# # #         top.setLayout(top_bar)
# # #         title = QLabel(f"  {self.current_contact}")
# # #         title.setStyleSheet("color: white;")
# # #         title.setFont(QFont("Arial", 12, QFont.Bold))
# # #         attach_btn = QPushButton("📎")
# # #         attach_btn.setToolTip("Send attachment")
# # #         attach_btn.setFixedWidth(36)
# # #         attach_btn.setStyleSheet("QPushButton { color: white; background: transparent; }")
# # #         attach_btn.clicked.connect(self.choose_attachment)
# # #
# # #         simulate_btn = QPushButton("⇦ Simulate Incoming")
# # #         simulate_btn.setToolTip("Add an incoming message (for testing alerts)")
# # #         simulate_btn.setStyleSheet(
# # #             "QPushButton { color: white; background: rgba(255,255,255,0.15); border-radius: 6px; padding: 4px 8px; }"
# # #             "QPushButton:hover { background: rgba(255,255,255,0.25); }"
# # #         )
# # #         simulate_btn.clicked.connect(self.simulate_incoming)
# # #
# # #         top_bar.addWidget(title)
# # #         top_bar.addStretch(1)
# # #         top_bar.addWidget(simulate_btn)
# # #         top_bar.addWidget(attach_btn)
# # #         root.addWidget(top)
# # #
# # #         self.scroll = QScrollArea()
# # #         self.scroll.setWidgetResizable(True)
# # #         self.scroll.setStyleSheet("QScrollArea { border: none; }")
# # #         self.chat_content = QWidget()
# # #         self.chat_layout = QVBoxLayout(self.chat_content)
# # #         self.chat_layout.setContentsMargins(12, 12, 12, 12)
# # #         self.chat_layout.setSpacing(10)
# # #         self.bottom_spacer = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
# # #         self.chat_layout.addItem(self.bottom_spacer)
# # #         self.scroll.setWidget(self.chat_content)
# # #         root.addWidget(self.scroll)
# # #
# # #         input_bar = QHBoxLayout()
# # #         input_bar.setContentsMargins(10, 10, 10, 10)
# # #         self.input_box = QLineEdit()
# # #         self.input_box.setPlaceholderText("Type a message…")
# # #         self.input_box.setFont(QFont("Arial", 11))
# # #         self.input_box.returnPressed.connect(self.handle_send)
# # #
# # #         self.send_btn = QPushButton("Send")
# # #         self.send_btn.setFont(QFont("Arial", 11))
# # #         self.send_btn.setStyleSheet(
# # #             "QPushButton { background-color: #25D366; color: white; border-radius: 6px; padding: 6px 14px; }"
# # #             "QPushButton:hover { background-color: #1EBE5B; }"
# # #         )
# # #         self.send_btn.clicked.connect(self.handle_send)
# # #
# # #         input_bar.addWidget(self.input_box)
# # #         input_bar.addWidget(self.send_btn)
# # #         root.addLayout(input_bar)
# # #
# # #         self.pending_attachment = None
# # #
# # #     def handle_send(self):
# # #         text = self.input_box.text().strip()
# # #         if not text and not self.pending_attachment:
# # #             return
# # #         if text:
# # #             flag, score, reasons = is_spam(text, sender_is_user=True)
# # #         else:
# # #             flag, score, reasons = (False, 0.0, [])
# # #         if flag:
# # #             msg = (
# # #                 "⚠️ This message looks suspicious.\n\n"
# # #                 f"Confidence: {int(score*100)}%\n"
# # #                 f"Reasons: {', '.join(reasons) if reasons else 'N/A'}\n\n"
# # #                 "Do you still want to send it?"
# # #             )
# # #             res = QMessageBox.warning(self, "Spam Warning", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
# # #             if res == QMessageBox.No:
# # #                 self.input_box.setFocus()
# # #                 return
# # #         self.add_message_bubble(text, is_user=True, attachment_path=self.pending_attachment)
# # #         self.input_box.clear()
# # #         self.pending_attachment = None
# # #         self.receive_message(f"Echo: {text}" if text else "(received your attachment)")
# # #
# # #     def receive_message(self, text: str, attachment_path: str = None):
# # #         is_unknown_contact = self.current_contact not in self.known_contacts
# # #         first_time_from_contact = self.current_contact not in self.has_seen_message_from_contact
# # #         flag, score, reasons = is_spam(text or "", sender_is_user=False)
# # #         if flag:
# # #             if is_unknown_contact and first_time_from_contact:
# # #                 self.add_banner("First message from this sender looks like spam. Do not click links or share OTPs.", strong=True)
# # #             else:
# # #                 self.add_banner("This message is suspected to be spam. Be cautious with links and requests.", strong=False)
# # #         self.add_message_bubble(text, is_user=False, is_spam=flag, attachment_path=attachment_path)
# # #         self.has_seen_message_from_contact.add(self.current_contact)
# # #
# # #     def add_banner(self, text: str, strong: bool = False):
# # #         banner = WarningBanner(text, strong=strong)
# # #         idx = self.chat_layout.count() - 1
# # #         self.chat_layout.insertWidget(idx, banner, 0, Qt.AlignHCenter)
# # #         self.smooth_scroll_to_bottom()
# # #
# # #     def add_message_bubble(self, text: str = None, is_user: bool = False,
# # #                            is_spam: bool = False, attachment_path: str = None):
# # #         bubble = ChatBubble(
# # #             sender="You" if is_user else self.current_contact,
# # #             text=text,
# # #             is_user=is_user,
# # #             is_spam=is_spam,
# # #             attachment_path=attachment_path
# # #         )
# # #         idx = self.chat_layout.count() - 1
# # #         align = Qt.AlignRight if is_user else Qt.AlignLeft
# # #         self.chat_layout.insertWidget(idx, bubble, 0, align)
# # #         spacer = QWidget()
# # #         spacer.setFixedHeight(2)
# # #         self.chat_layout.insertWidget(idx + 1, spacer)
# # #         self.smooth_scroll_to_bottom()
# # #
# # #     def smooth_scroll_to_bottom(self):
# # #         sb = self.scroll.verticalScrollBar()
# # #         anim = QPropertyAnimation(sb, b"value")
# # #         anim.setDuration(350)
# # #         anim.setStartValue(sb.value())
# # #         anim.setEndValue(sb.maximum())
# # #         anim.start()
# # #         self._anim = anim
# # #
# # #     def choose_attachment(self):
# # #         path, _ = QFileDialog.getOpenFileName(self, "Choose attachment", "", "All Files (*.*)")
# # #         if not path:
# # #             return
# # #         self.pending_attachment = path
# # #         base = Path(path).name
# # #         self.add_banner(f"Attachment ready: {base}. It will be sent with your next message.", strong=False)
# # #
# # #     def simulate_incoming(self):
# # #         path, _ = QFileDialog.getOpenFileName(self, "Simulate incoming attachment (optional)", "", "All Files (*.*)")
# # #         if path:
# # #             self.receive_message("(incoming attachment)", attachment_path=path)
# # #         else:
# # #             self.receive_message("Congratulations! You won a FREE reward. Click http://fake.link to claim now.")
# # #
# # # # -----------------------------
# # # # Launch App
# # # # -----------------------------
# # # if __name__ == "__main__":
# # #     app = QApplication(sys.argv)
# # #     win = ChatWindow()
# # #     win.show()
# # #     sys.exit(app.exec_())
# #
# #
# # import sys
# # import time
# # import random
# # from pathlib import Path
# # import requests
# #
# # from PyQt5.QtWidgets import (
# #     QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
# #     QScrollArea, QLabel, QFrame, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy
# # )
# # from PyQt5.QtCore import Qt, QPropertyAnimation, QUrl
# # from PyQt5.QtGui import QFont, QPixmap
# #
# # # -----------------------------
# # # FastAPI Backend URL
# # # -----------------------------
# # FASTAPI_URL = "http://127.0.0.1:8000/predict-smote"  # Change if deployed remotely
# #
# # # -----------------------------
# # # Spam Classification via API
# # # -----------------------------
# # def is_spam(text: str, sender_is_user: bool):
# #     """Calls FastAPI backend to classify text as spam or ham."""
# #     if not text.strip():
# #         return False, 0.0, []
# #
# #     try:
# #         response = requests.post(FASTAPI_URL, json={"message": text}, timeout=5)
# #         data = response.json()
# #         label = data.get("prediction", "ham").lower()
# #         confidence = data.get("confidence", 0.0)
# #
# #         # Optional: extract top spammy words locally
# #         spammy_terms = ["win", "free", "offer", "click", "link"]
# #         tokens = [t.lower() for t in text.split()]
# #         reasons = [t for t in tokens if t in spammy_terms][:5]
# #
# #         return label == "spam", float(confidence), reasons
# #
# #     except Exception as e:
# #         print("Error calling FastAPI:", e)
# #         return False, 0.0, []
# #
# # # -----------------------------
# # # UI Components
# # # -----------------------------
# # class WarningBanner(QFrame):
# #     def __init__(self, text: str, strong: bool = False):
# #         super().__init__()
# #         self.setFrameShape(QFrame.NoFrame)
# #         self.setStyleSheet(
# #             "QFrame {"
# #             f"background-color: {'#FEE2E2' if strong else '#FEF3C7'};"
# #             "border-radius: 8px;"
# #             "border: 1px solid #FCA5A5; }}"
# #         )
# #         lay = QHBoxLayout(self)
# #         lay.setContentsMargins(10, 6, 10, 6)
# #         lay.setSpacing(8)
# #         icon = QLabel("⚠️")
# #         icon.setFont(QFont("Arial", 11))
# #         lbl = QLabel(text)
# #         lbl.setWordWrap(True)
# #         lbl.setFont(QFont("Arial", 10))
# #         lay.addWidget(icon)
# #         lay.addWidget(lbl)
# #         lay.addStretch(1)
# #
# # class ChatBubble(QFrame):
# #     def __init__(self, sender: str, text: str = None, timestamp: str = None,
# #                  is_user: bool = False, is_spam: bool = False,
# #                  attachment_path: str = None):
# #         super().__init__()
# #         self.setFrameShape(QFrame.NoFrame)
# #         base_bg = "#DCF8C6" if is_user else "#ECECEC"
# #         border = "#FCA5A5" if is_spam else "transparent"
# #         self.setStyleSheet(
# #             "QFrame {"
# #             f"background-color: {base_bg};"
# #             "border-radius: 12px;"
# #             f"border: 1px solid {border}; }}"
# #         )
# #         outer = QVBoxLayout(self)
# #         outer.setContentsMargins(10, 6, 10, 6)
# #         outer.setSpacing(4)
# #
# #         sender_lbl = QLabel(sender)
# #         sender_lbl.setFont(QFont("Arial", 9, QFont.Bold))
# #         sender_lbl.setStyleSheet("color: #4B5563;")
# #         if is_user:
# #             sender_lbl.hide()
# #         outer.addWidget(sender_lbl)
# #
# #         if attachment_path:
# #             ext = Path(attachment_path).suffix.lower()
# #             if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
# #                 img_lbl = QLabel()
# #                 pix = QPixmap(attachment_path)
# #                 if not pix.isNull():
# #                     pix = pix.scaledToWidth(220, Qt.SmoothTransformation)
# #                     img_lbl.setPixmap(pix)
# #                 else:
# #                     img_lbl.setText("(image failed to load)")
# #                 outer.addWidget(img_lbl)
# #                 if text:
# #                     msg_lbl = QLabel(text)
# #                     msg_lbl.setWordWrap(True)
# #                     msg_lbl.setFont(QFont("Arial", 11))
# #                     msg_lbl.setStyleSheet("QLabel { border: none; }")
# #                     outer.addWidget(msg_lbl)
# #             else:
# #                 link_lbl = QLabel(f'<a href="{QUrl.fromLocalFile(attachment_path).toString()}">{Path(attachment_path).name}</a>')
# #                 link_lbl.setOpenExternalLinks(True)
# #                 link_lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
# #                 link_lbl.setFont(QFont("Arial", 11))
# #                 outer.addWidget(link_lbl)
# #                 if text:
# #                     msg_lbl = QLabel(text)
# #                     msg_lbl.setWordWrap(True)
# #                     msg_lbl.setFont(QFont("Arial", 11))
# #                     outer.addWidget(msg_lbl)
# #         else:
# #             msg_lbl = QLabel(text or "")
# #             msg_lbl.setWordWrap(True)
# #             msg_lbl.setFont(QFont("Arial", 11))
# #             msg_lbl.setStyleSheet("QLabel { border: none; }")
# #             outer.addWidget(msg_lbl)
# #
# #         ts = timestamp or time.strftime("%H:%M")
# #         time_lbl = QLabel(ts)
# #         time_lbl.setFont(QFont("Arial", 8))
# #         time_lbl.setStyleSheet("color: grey;")
# #         time_lbl.setAlignment(Qt.AlignRight)
# #         outer.addWidget(time_lbl)
# #
# # # -----------------------------
# # # Chat Window
# # # -----------------------------
# # class ChatWindow(QWidget):
# #     def __init__(self):
# #         super().__init__()
# #         self.setWindowTitle("📩 SMS Chat — FastAPI Backend")
# #         self.resize(460, 680)
# #         self.setStyleSheet("background-color: white;")
# #         self.known_contacts = {"Alice", "Bob"}
# #         self.current_contact = "Unknown +91-9284-706-XXX"
# #         self.has_seen_message_from_contact = set()
# #         root = QVBoxLayout(self)
# #         root.setSpacing(0)
# #         root.setContentsMargins(0, 0, 0, 0)
# #
# #         # Top bar
# #         top_bar = QHBoxLayout()
# #         top = QFrame()
# #         top.setStyleSheet("QFrame { background-color: #075E54; }")
# #         top.setLayout(top_bar)
# #         title = QLabel(f"  {self.current_contact}")
# #         title.setStyleSheet("color: white;")
# #         title.setFont(QFont("Arial", 12, QFont.Bold))
# #         attach_btn = QPushButton("📎")
# #         attach_btn.setToolTip("Send attachment")
# #         attach_btn.setFixedWidth(36)
# #         attach_btn.setStyleSheet("QPushButton { color: white; background: transparent; }")
# #         attach_btn.clicked.connect(self.choose_attachment)
# #
# #         simulate_btn = QPushButton("⇦ Simulate Incoming")
# #         simulate_btn.setToolTip("Add an incoming message (for testing alerts)")
# #         simulate_btn.setStyleSheet(
# #             "QPushButton { color: white; background: rgba(255,255,255,0.15); border-radius: 6px; padding: 4px 8px; }"
# #             "QPushButton:hover { background: rgba(255,255,255,0.25); }"
# #         )
# #         simulate_btn.clicked.connect(self.simulate_incoming)
# #
# #         top_bar.addWidget(title)
# #         top_bar.addStretch(1)
# #         top_bar.addWidget(simulate_btn)
# #         top_bar.addWidget(attach_btn)
# #         root.addWidget(top)
# #
# #         # Chat area
# #         self.scroll = QScrollArea()
# #         self.scroll.setWidgetResizable(True)
# #         self.scroll.setStyleSheet("QScrollArea { border: none; }")
# #         self.chat_content = QWidget()
# #         self.chat_layout = QVBoxLayout(self.chat_content)
# #         self.chat_layout.setContentsMargins(12, 12, 12, 12)
# #         self.chat_layout.setSpacing(10)
# #         self.bottom_spacer = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
# #         self.chat_layout.addItem(self.bottom_spacer)
# #         self.scroll.setWidget(self.chat_content)
# #         root.addWidget(self.scroll)
# #
# #         # Input area
# #         input_bar = QHBoxLayout()
# #         input_bar.setContentsMargins(10, 10, 10, 10)
# #         self.input_box = QLineEdit()
# #         self.input_box.setPlaceholderText("Type a message…")
# #         self.input_box.setFont(QFont("Arial", 11))
# #         self.input_box.returnPressed.connect(self.handle_send)
# #
# #         self.send_btn = QPushButton("Send")
# #         self.send_btn.setFont(QFont("Arial", 11))
# #         self.send_btn.setStyleSheet(
# #             "QPushButton { background-color: #25D366; color: white; border-radius: 6px; padding: 6px 14px; }"
# #             "QPushButton:hover { background-color: #1EBE5B; }"
# #         )
# #         self.send_btn.clicked.connect(self.handle_send)
# #
# #         input_bar.addWidget(self.input_box)
# #         input_bar.addWidget(self.send_btn)
# #         root.addLayout(input_bar)
# #
# #         self.pending_attachment = None
# #
# #     # -----------------------------
# #     # Handle sending user messages
# #     # -----------------------------
# #     def handle_send(self):
# #         text = self.input_box.text().strip()
# #         if not text and not self.pending_attachment:
# #             return
# #         if text:
# #             flag, score, reasons = is_spam(text, sender_is_user=True)
# #         else:
# #             flag, score, reasons = (False, 0.0, [])
# #
# #         if flag:
# #             msg = (
# #                 "⚠️ This message looks suspicious.\n\n"
# #                 f"Confidence: {int(score*100)}%\n"
# #                 f"Reasons: {', '.join(reasons) if reasons else 'N/A'}\n\n"
# #                 "Do you still want to send it?"
# #             )
# #             res = QMessageBox.warning(self, "Spam Warning", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
# #             if res == QMessageBox.No:
# #                 self.input_box.setFocus()
# #                 return
# #
# #         self.add_message_bubble(text, is_user=True, attachment_path=self.pending_attachment)
# #         self.input_box.clear()
# #         self.pending_attachment = None
# #         # Optional echo response
# #         self.receive_message(f"Echo: {text}" if text else "(received your attachment)")
# #
# #     # -----------------------------
# #     # Receive incoming messages
# #     # -----------------------------
# #     def receive_message(self, text: str, attachment_path: str = None):
# #         is_unknown_contact = self.current_contact not in self.known_contacts
# #         first_time_from_contact = self.current_contact not in self.has_seen_message_from_contact
# #         flag, score, reasons = is_spam(text or "", sender_is_user=False)
# #
# #         if flag:
# #             if is_unknown_contact and first_time_from_contact:
# #                 self.add_banner("First message from this sender looks like spam. Do not click links or share OTPs.", strong=True)
# #             else:
# #                 self.add_banner("This message is suspected to be spam. Be cautious with links and requests.", strong=False)
# #
# #         self.add_message_bubble(text, is_user=False, is_spam=flag, attachment_path=attachment_path)
# #         self.has_seen_message_from_contact.add(self.current_contact)
# #
# #     # -----------------------------
# #     # Add banner
# #     # -----------------------------
# #     def add_banner(self, text: str, strong: bool = False):
# #         banner = WarningBanner(text, strong=strong)
# #         idx = self.chat_layout.count() - 1
# #         self.chat_layout.insertWidget(idx, banner, 0, Qt.AlignHCenter)
# #         self.smooth_scroll_to_bottom()
# #
# #     # -----------------------------
# #     # Add chat bubble
# #     # -----------------------------
# #     def add_message_bubble(self, text: str = None, is_user: bool = False,
# #                            is_spam: bool = False, attachment_path: str = None):
# #         bubble = ChatBubble(
# #             sender="You" if is_user else self.current_contact,
# #             text=text,
# #             is_user=is_user,
# #             is_spam=is_spam,
# #             attachment_path=attachment_path
# #         )
# #         idx = self.chat_layout.count() - 1
# #         align = Qt.AlignRight if is_user else Qt.AlignLeft
# #         self.chat_layout.insertWidget(idx, bubble, 0, align)
# #         spacer = QWidget()
# #         spacer.setFixedHeight(2)
# #         self.chat_layout.insertWidget(idx + 1, spacer)
# #         self.smooth_scroll_to_bottom()
# #
# #     # -----------------------------
# #     # Smooth scroll
# #     # -----------------------------
# #     def smooth_scroll_to_bottom(self):
# #         sb = self.scroll.verticalScrollBar()
# #         anim = QPropertyAnimation(sb, b"value")
# #         anim.setDuration(350)
# #         anim.setStartValue(sb.value())
# #         anim.setEndValue(sb.maximum())
# #         anim.start()
# #         self._anim = anim
# #
# #     # -----------------------------
# #     # Choose attachment
# #     # -----------------------------
# #     def choose_attachment(self):
# #         path, _ = QFileDialog.getOpenFileName(self, "Choose attachment", "", "All Files (*.*)")
# #         if not path:
# #             return
# #         self.pending_attachment = path
# #         base = Path(path).name
# #         self.add_banner(f"Attachment ready: {base}. It will be sent with your next message.", strong=False)
# #
# #     # -----------------------------
# #     # Simulate incoming message
# #     # -----------------------------
# #     def simulate_incoming(self):
# #         # Random text messages; model decides spam/ham
# #         sample_messages = [
# #             "Hey! How are you doing today?",
# #             "Are we still meeting at 5 PM?",
# #             "Congratulations! You won a FREE reward. Click http://fake.link to claim now.",
# #             "Win a brand new iPhone! Visit http://fake.link now!",
# #             "Don't forget to submit your assignment by tonight.",
# #             "Your OTP is 123456. Do not share it with anyone.",
# #             "Limited time offer! Claim your free voucher now."
# #         ]
# #         msg = random.choice(sample_messages)
# #
# #         # Optional attachment only occasionally
# #         attach_path = None
# #         if random.random() < 0.1:  # 10% chance
# #             path, _ = QFileDialog.getOpenFileName(self, "Optional attachment", "", "All Files (*.*)")
# #             if path:
# #                 attach_path = path
# #
# #         # Let the backend classify
# #         self.receive_message(msg, attachment_path=attach_path)
# #
# # # -----------------------------
# # # Launch Application
# # # -----------------------------
# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     win = ChatWindow()
# #     win.show()
# #     sys.exit(app.exec_())
#
#
# import sys
# import time
# import random
# import threading
# from pathlib import Path
# import requests
# from PyQt5.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
#     QScrollArea, QLabel, QFrame, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy
# )
# from PyQt5.QtCore import Qt, QPropertyAnimation, QUrl
# from PyQt5.QtGui import QFont, QPixmap
#
# # -----------------------------
# # FastAPI backend
# # -----------------------------
# from fastapi import FastAPI
# from pydantic import BaseModel
# import uvicorn
#
# app = FastAPI(title="SMS Spam Classifier API")
#
# class MessageRequest(BaseModel):
#     message: str
#
# @app.post("/predict-smote")
# def predict_smote(req: MessageRequest):
#     text = req.message.lower()
#     # Example simple rules; replace with your actual ML model prediction
#     if any(word in text for word in ["win", "free", "offer", "click", "link"]):
#         return {"prediction": "spam", "confidence": 0.95}
#     return {"prediction": "ham", "confidence": 0.85}
#
# # Start FastAPI in a background thread
# def start_fastapi():
#     uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
#
# threading.Thread(target=start_fastapi, daemon=True).start()
#
# # -----------------------------
# # Spam Classification via API
# # -----------------------------
# FASTAPI_URL = "http://127.0.0.1:8000/predict-smote"
#
# def is_spam(text: str, sender_is_user: bool):
#     """Call FastAPI backend to classify text."""
#     if not text.strip():
#         return False, 0.0, []
#     try:
#         response = requests.post(FASTAPI_URL, json={"message": text}, timeout=5)
#         data = response.json()
#         label = data.get("prediction", "ham").lower()
#         confidence = data.get("confidence", 0.0)
#
#         # Optional: extract top spammy words locally
#         spammy_terms = ["win", "free", "offer", "click", "link"]
#         tokens = [t.lower() for t in text.split()]
#         reasons = [t for t in tokens if t in spammy_terms][:5]
#
#         return label == "spam", float(confidence), reasons
#
#     except Exception as e:
#         print("Error calling FastAPI:", e)
#         return False, 0.0, []
#
# # -----------------------------
# # UI Components
# # -----------------------------
# class WarningBanner(QFrame):
#     def __init__(self, text: str, strong: bool = False):
#         super().__init__()
#         self.setFrameShape(QFrame.NoFrame)
#         self.setStyleSheet(
#             f"QFrame {{background-color: {'#FEE2E2' if strong else '#FEF3C7'}; "
#             "border-radius: 8px; border: 1px solid #FCA5A5;}}"
#         )
#         lay = QHBoxLayout(self)
#         lay.setContentsMargins(10, 6, 10, 6)
#         lay.setSpacing(8)
#         icon = QLabel("⚠️")
#         icon.setFont(QFont("Arial", 11))
#         lbl = QLabel(text)
#         lbl.setWordWrap(True)
#         lbl.setFont(QFont("Arial", 10))
#         lay.addWidget(icon)
#         lay.addWidget(lbl)
#         lay.addStretch(1)
#
# class ChatBubble(QFrame):
#     def __init__(self, sender: str, text: str = None, timestamp: str = None,
#                  is_user: bool = False, is_spam: bool = False,
#                  attachment_path: str = None):
#         super().__init__()
#         self.setFrameShape(QFrame.NoFrame)
#         base_bg = "#DCF8C6" if is_user else "#ECECEC"
#         border = "#FCA5A5" if is_spam else "transparent"
#         self.setStyleSheet(
#             f"QFrame {{background-color: {base_bg}; border-radius: 12px; border: 1px solid {border};}}"
#         )
#         outer = QVBoxLayout(self)
#         outer.setContentsMargins(10, 6, 10, 6)
#         outer.setSpacing(4)
#
#         sender_lbl = QLabel(sender)
#         sender_lbl.setFont(QFont("Arial", 9, QFont.Bold))
#         sender_lbl.setStyleSheet("color: #4B5563;")
#         if is_user:
#             sender_lbl.hide()
#         outer.addWidget(sender_lbl)
#
#         if attachment_path:
#             ext = Path(attachment_path).suffix.lower()
#             if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
#                 img_lbl = QLabel()
#                 pix = QPixmap(attachment_path)
#                 if not pix.isNull():
#                     pix = pix.scaledToWidth(220, Qt.SmoothTransformation)
#                     img_lbl.setPixmap(pix)
#                 else:
#                     img_lbl.setText("(image failed to load)")
#                 outer.addWidget(img_lbl)
#                 if text:
#                     msg_lbl = QLabel(text)
#                     msg_lbl.setWordWrap(True)
#                     msg_lbl.setFont(QFont("Arial", 11))
#                     outer.addWidget(msg_lbl)
#             else:
#                 link_lbl = QLabel(f'<a href="{QUrl.fromLocalFile(attachment_path).toString()}">{Path(attachment_path).name}</a>')
#                 link_lbl.setOpenExternalLinks(True)
#                 link_lbl.setTextInteractionFlags(Qt.TextBrowserInteraction)
#                 link_lbl.setFont(QFont("Arial", 11))
#                 outer.addWidget(link_lbl)
#                 if text:
#                     msg_lbl = QLabel(text)
#                     msg_lbl.setWordWrap(True)
#                     msg_lbl.setFont(QFont("Arial", 11))
#                     outer.addWidget(msg_lbl)
#         else:
#             msg_lbl = QLabel(text or "")
#             msg_lbl.setWordWrap(True)
#             msg_lbl.setFont(QFont("Arial", 11))
#             outer.addWidget(msg_lbl)
#
#         ts = timestamp or time.strftime("%H:%M")
#         time_lbl = QLabel(ts)
#         time_lbl.setFont(QFont("Arial", 8))
#         time_lbl.setStyleSheet("color: grey;")
#         time_lbl.setAlignment(Qt.AlignRight)
#         outer.addWidget(time_lbl)
#
# # -----------------------------
# # Chat Window
# # -----------------------------
# class ChatWindow(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("📩 SMS Chat — FastAPI Backend")
#         self.resize(460, 680)
#         self.setStyleSheet("background-color: white;")
#         self.known_contacts = {"Alice", "Bob"}
#         self.current_contact = "Unknown +91-9284-706-XXX"
#         self.has_seen_message_from_contact = set()
#         root = QVBoxLayout(self)
#         root.setSpacing(0)
#         root.setContentsMargins(0, 0, 0, 0)
#
#         # Top bar
#         top_bar = QHBoxLayout()
#         top = QFrame()
#         top.setStyleSheet("QFrame { background-color: #075E54; }")
#         top.setLayout(top_bar)
#         title = QLabel(f"  {self.current_contact}")
#         title.setStyleSheet("color: white;")
#         title.setFont(QFont("Arial", 12, QFont.Bold))
#         attach_btn = QPushButton("📎")
#         attach_btn.setToolTip("Send attachment")
#         attach_btn.setFixedWidth(36)
#         attach_btn.setStyleSheet("QPushButton { color: white; background: transparent; }")
#         attach_btn.clicked.connect(self.choose_attachment)
#
#         simulate_btn = QPushButton("⇦ Simulate Incoming")
#         simulate_btn.setToolTip("Add an incoming message (for testing alerts)")
#         simulate_btn.setStyleSheet(
#             "QPushButton { color: white; background: rgba(255,255,255,0.15); border-radius: 6px; padding: 4px 8px; }"
#             "QPushButton:hover { background: rgba(255,255,255,0.25); }"
#         )
#         simulate_btn.clicked.connect(self.simulate_incoming)
#
#         top_bar.addWidget(title)
#         top_bar.addStretch(1)
#         top_bar.addWidget(simulate_btn)
#         top_bar.addWidget(attach_btn)
#         root.addWidget(top)
#
#         # Chat area
#         self.scroll = QScrollArea()
#         self.scroll.setWidgetResizable(True)
#         self.scroll.setStyleSheet("QScrollArea { border: none; }")
#         self.chat_content = QWidget()
#         self.chat_layout = QVBoxLayout(self.chat_content)
#         self.chat_layout.setContentsMargins(12, 12, 12, 12)
#         self.chat_layout.setSpacing(10)
#         self.bottom_spacer = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
#         self.chat_layout.addItem(self.bottom_spacer)
#         self.scroll.setWidget(self.chat_content)
#         root.addWidget(self.scroll)
#
#         # Input area
#         input_bar = QHBoxLayout()
#         input_bar.setContentsMargins(10, 10, 10, 10)
#         self.input_box = QLineEdit()
#         self.input_box.setPlaceholderText("Type a message…")
#         self.input_box.setFont(QFont("Arial", 11))
#         self.input_box.returnPressed.connect(self.handle_send)
#
#         self.send_btn = QPushButton("Send")
#         self.send_btn.setFont(QFont("Arial", 11))
#         self.send_btn.setStyleSheet(
#             "QPushButton { background-color: #25D366; color: white; border-radius: 6px; padding: 6px 14px; }"
#             "QPushButton:hover { background-color: #1EBE5B; }"
#         )
#         self.send_btn.clicked.connect(self.handle_send)
#
#         input_bar.addWidget(self.input_box)
#         input_bar.addWidget(self.send_btn)
#         root.addLayout(input_bar)
#
#         self.pending_attachment = None
#
#     def handle_send(self):
#         text = self.input_box.text().strip()
#         if not text and not self.pending_attachment:
#             return
#         if text:
#             flag, score, reasons = is_spam(text, sender_is_user=True)
#         else:
#             flag, score, reasons = (False, 0.0, [])
#         if flag:
#             msg = (
#                 "⚠️ This message looks suspicious.\n\n"
#                 f"Confidence: {int(score*100)}%\n"
#                 f"Reasons: {', '.join(reasons) if reasons else 'N/A'}\n\n"
#                 "Do you still want to send it?"
#             )
#             res = QMessageBox.warning(self, "Spam Warning", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
#             if res == QMessageBox.No:
#                 self.input_box.setFocus()
#                 return
#         self.add_message_bubble(text, is_user=True, attachment_path=self.pending_attachment)
#         self.input_box.clear()
#         self.pending_attachment = None
#
#     def receive_message(self, text: str, attachment_path: str = None):
#         is_unknown_contact = self.current_contact not in self.known_contacts
#         first_time_from_contact = self.current_contact not in self.has_seen_message_from_contact
#         flag, score, reasons = is_spam(text or "", sender_is_user=False)
#         if flag:
#             if is_unknown_contact and first_time_from_contact:
#                 self.add_banner("First message from this sender looks like spam. Do not click links or share OTPs.", strong=True)
#             else:
#                 self.add_banner("This message is suspected to be spam. Be cautious with links and requests.", strong=False)
#         self.add_message_bubble(text, is_user=False, is_spam=flag, attachment_path=attachment_path)
#         self.has_seen_message_from_contact.add(self.current_contact)
#
#     def add_banner(self, text: str, strong: bool = False):
#         banner = WarningBanner(text, strong=strong)
#         idx = self.chat_layout.count() - 1
#         self.chat_layout.insertWidget(idx, banner, 0, Qt.AlignHCenter)
#         self.smooth_scroll_to_bottom()
#
#     def add_message_bubble(self, text: str = None, is_user: bool = False,
#                            is_spam: bool = False, attachment_path: str = None):
#         bubble = ChatBubble(
#             sender="You" if is_user else self.current_contact,
#             text=text,
#             is_user=is_user,
#             is_spam=is_spam,
#             attachment_path=attachment_path
#         )
#         idx = self.chat_layout.count() - 1
#         align = Qt.AlignRight if is_user else Qt.AlignLeft
#         self.chat_layout.insertWidget(idx, bubble, 0, align)
#         spacer = QWidget()
#         spacer.setFixedHeight(2)
#         self.chat_layout.insertWidget(idx + 1, spacer)
#         self.smooth_scroll_to_bottom()
#
#     def smooth_scroll_to_bottom(self):
#         sb = self.scroll.verticalScrollBar()
#         anim = QPropertyAnimation(sb, b"value")
#         anim.setDuration(350)
#         anim.setStartValue(sb.value())
#         anim.setEndValue(sb.maximum())
#         anim.start()
#         self._anim = anim
#
#     def choose_attachment(self):
#         path, _ = QFileDialog.getOpenFileName(self, "Choose attachment", "", "All Files (*.*)")
#         if not path:
#             return
#         self.pending_attachment = path
#         base = Path(path).name
#         self.add_banner(f"Attachment ready: {base}. It will be sent with your next message.", strong=False)
#
#     def simulate_incoming(self):
#         # Random text; backend will classify
#         sample_messages = [
#             "Hey! How are you doing today?",
#             "Are we still meeting at 5 PM?",
#             "Congratulations! You won a FREE reward. Click http://fake.link to claim now.",
#             "Win a brand new iPhone! Visit http://fake.link now!",
#             "Don't forget to submit your assignment by tonight.",
#             "Your OTP is 123456. Do not share it with anyone.",
#             "Limited time offer! Claim your free voucher now."
#         ]
#         msg = random.choice(sample_messages)
#
#         # Optional attachment only occasionally
#         attach_path = None
#         if random.random() < 0.1:
#             path, _ = QFileDialog.getOpenFileName(self, "Optional attachment", "", "All Files (*.*)")
#             if path:
#                 attach_path = path
#
#         # Let the backend classify
#         self.receive_message(msg, attachment_path=attach_path)
#
# # -----------------------------
# # Launch PyQt GUI
# # -----------------------------
# if __name__ == "__main__":
#     app_qt = QApplication(sys.argv)
#     win = ChatWindow()
#     win.show()
#     sys.exit(app_qt.exec_())


import sys
import time
import random
import threading
from pathlib import Path
import requests

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton,
    QScrollArea, QLabel, QFrame, QFileDialog, QMessageBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QPropertyAnimation, QUrl
from PyQt5.QtGui import QFont, QPixmap

# -----------------------------
# FastAPI imports
# -----------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# -----------------------------
# Sample model simulation
# -----------------------------
# Replace this with your trained RandomForest or any ML model
def predict_spam(message: str):
    """Simulate model prediction"""
    spam_keywords = ["win", "free", "offer", "click", "link", "reward", "voucher", "iphone"]
    score = min(1.0, sum(word in message.lower() for word in spam_keywords) / 2)
    label = "spam" if score >= 0.5 else "ham"
    reasons = [word for word in message.lower().split() if word in spam_keywords][:5]
    return label, score, reasons

# -----------------------------
# FastAPI Backend
# -----------------------------
app = FastAPI(title="SMS Spam Classifier API")

class MessageRequest(BaseModel):
    message: str

@app.post("/predict-smote")
def predict(request: MessageRequest):
    label, score, reasons = predict_spam(request.message)
    return {"prediction": label, "confidence": score, "reasons": reasons}

# -----------------------------
# FastAPI Runner Thread
# -----------------------------
def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

# -----------------------------
# Spam Classification via API
# -----------------------------
FASTAPI_URL = "http://127.0.0.1:8000/predict-smote"

def is_spam(text: str, sender_is_user: bool):
    """Call FastAPI backend to classify text as spam or ham"""
    if not text.strip():
        return False, 0.0, []

    try:
        response = requests.post(FASTAPI_URL, json={"message": text}, timeout=5)
        data = response.json()
        label = data.get("prediction", "ham").lower()
        confidence = data.get("confidence", 0.0)
        reasons = data.get("reasons", [])

        return label == "spam", float(confidence), reasons
    except Exception as e:
        print("Error calling FastAPI:", e)
        return False, 0.0, []

# -----------------------------
# PyQt GUI Components
# -----------------------------
class WarningBanner(QFrame):
    def __init__(self, text: str, strong: bool = False):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet(
            "QFrame {"
            f"background-color: {'#FEE2E2' if strong else '#FEF3C7'};"
            "border-radius: 8px;"
            "border: 1px solid #FCA5A5; }"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(8)
        icon = QLabel("⚠️")
        icon.setFont(QFont("Arial", 11))
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setFont(QFont("Arial", 10))
        lay.addWidget(icon)
        lay.addWidget(lbl)
        lay.addStretch(1)

class ChatBubble(QFrame):
    def __init__(self, sender: str, text: str = None, timestamp: str = None,
                 is_user: bool = False, is_spam: bool = False,
                 attachment_path: str = None):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame)
        base_bg = "#DCF8C6" if is_user else "#ECECEC"
        border = "#FCA5A5" if is_spam else "transparent"
        self.setStyleSheet(
            "QFrame {"
            f"background-color: {base_bg};"
            "border-radius: 12px;"
            f"border: 1px solid {border}; }}"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        sender_lbl = QLabel(sender)
        sender_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        sender_lbl.setStyleSheet("color: #4B5563;")
        if is_user:
            sender_lbl.hide()
        outer.addWidget(sender_lbl)

        if attachment_path:
            ext = Path(attachment_path).suffix.lower()
            if ext in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
                img_lbl = QLabel()
                pix = QPixmap(attachment_path)
                if not pix.isNull():
                    pix = pix.scaledToWidth(220, Qt.SmoothTransformation)
                    img_lbl.setPixmap(pix)
                else:
                    img_lbl.setText("(image failed to load)")
                outer.addWidget(img_lbl)
                if text:
                    msg_lbl = QLabel(text)
                    msg_lbl.setWordWrap(True)
                    msg_lbl.setFont(QFont("Arial", 11))
                    outer.addWidget(msg_lbl)
            else:
                link_lbl = QLabel(f'<a href="{QUrl.fromLocalFile(attachment_path).toString()}">{Path(attachment_path).name}</a>')
                link_lbl.setOpenExternalLinks(True)
                link_lbl.setFont(QFont("Arial", 11))
                outer.addWidget(link_lbl)
                if text:
                    msg_lbl = QLabel(text)
                    msg_lbl.setWordWrap(True)
                    msg_lbl.setFont(QFont("Arial", 11))
                    outer.addWidget(msg_lbl)
        else:
            msg_lbl = QLabel(text or "")
            msg_lbl.setWordWrap(True)
            msg_lbl.setFont(QFont("Arial", 11))
            outer.addWidget(msg_lbl)

        ts = timestamp or time.strftime("%H:%M")
        time_lbl = QLabel(ts)
        time_lbl.setFont(QFont("Arial", 8))
        time_lbl.setStyleSheet("color: grey;")
        time_lbl.setAlignment(Qt.AlignRight)
        outer.addWidget(time_lbl)

# -----------------------------
# Chat Window
# -----------------------------
class ChatWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📩 SMS Chat — FastAPI Backend")
        self.resize(460, 680)
        self.setStyleSheet("background-color: white;")
        self.known_contacts = {"Alice", "Bob"}
        self.current_contact = "Unknown +91-9284-706-XXX"
        self.has_seen_message_from_contact = set()
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Top bar
        top_bar = QHBoxLayout()
        top = QFrame()
        top.setStyleSheet("QFrame { background-color: #075E54; }")
        top.setLayout(top_bar)
        title = QLabel(f"  {self.current_contact}")
        title.setStyleSheet("color: white;")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        attach_btn = QPushButton("📎")
        attach_btn.setToolTip("Send attachment")
        attach_btn.setFixedWidth(36)
        attach_btn.setStyleSheet("QPushButton { color: white; background: transparent; }")
        attach_btn.clicked.connect(self.choose_attachment)

        simulate_btn = QPushButton("⇦ Simulate Incoming")
        simulate_btn.setToolTip("Simulate incoming message")
        simulate_btn.setStyleSheet(
            "QPushButton { color: white; background: rgba(255,255,255,0.15); border-radius: 6px; padding: 4px 8px; }"
            "QPushButton:hover { background: rgba(255,255,255,0.25); }"
        )
        simulate_btn.clicked.connect(self.simulate_incoming)

        top_bar.addWidget(title)
        top_bar.addStretch(1)
        top_bar.addWidget(simulate_btn)
        top_bar.addWidget(attach_btn)
        root.addWidget(top)

        # Chat area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        self.chat_content = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_content)
        self.chat_layout.setContentsMargins(12, 12, 12, 12)
        self.chat_layout.setSpacing(10)
        self.bottom_spacer = QSpacerItem(1, 1, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.chat_layout.addItem(self.bottom_spacer)
        self.scroll.setWidget(self.chat_content)
        root.addWidget(self.scroll)

        # Input area
        input_bar = QHBoxLayout()
        input_bar.setContentsMargins(10, 10, 10, 10)
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type a message…")
        self.input_box.setFont(QFont("Arial", 11))
        self.input_box.returnPressed.connect(self.handle_send)

        self.send_btn = QPushButton("Send")
        self.send_btn.setFont(QFont("Arial", 11))
        self.send_btn.setStyleSheet(
            "QPushButton { background-color: #25D366; color: white; border-radius: 6px; padding: 6px 14px; }"
            "QPushButton:hover { background-color: #1EBE5B; }"
        )
        self.send_btn.clicked.connect(self.handle_send)

        input_bar.addWidget(self.input_box)
        input_bar.addWidget(self.send_btn)
        root.addLayout(input_bar)

        self.pending_attachment = None

    # -----------------------------
    # Send user message
    # -----------------------------
    def handle_send(self):
        text = self.input_box.text().strip()
        if not text and not self.pending_attachment:
            return
        if text:
            flag, score, reasons = is_spam(text, sender_is_user=True)
        else:
            flag, score, reasons = (False, 0.0, [])

        if flag:
            msg = (
                "⚠️ This message looks suspicious.\n\n"
                f"Confidence: {int(score*100)}%\n"
                f"Reasons: {', '.join(reasons) if reasons else 'N/A'}\n\n"
                "Do you still want to send it?"
            )
            res = QMessageBox.warning(self, "Spam Warning", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if res == QMessageBox.No:
                self.input_box.setFocus()
                return

        self.add_message_bubble(text, is_user=True, attachment_path=self.pending_attachment)
        self.input_box.clear()
        self.pending_attachment = None

    # -----------------------------
    # Receive incoming message
    # -----------------------------
    def receive_message(self, text: str, attachment_path: str = None):
        is_unknown_contact = self.current_contact not in self.known_contacts
        first_time_from_contact = self.current_contact not in self.has_seen_message_from_contact
        flag, score, reasons = is_spam(text or "", sender_is_user=False)

        # Include info for spam messages
        info_text = ""
        if flag:
            info_text = f"⚠️ Suspected spam!\nConfidence: {int(score*100)}%\nReasons: {', '.join(reasons) if reasons else 'N/A'}"
        full_text = f"{text}\n\n{info_text}" if info_text else text

        self.add_message_bubble(full_text, is_user=False, is_spam=flag, attachment_path=attachment_path)

        if flag:
            if is_unknown_contact and first_time_from_contact:
                self.add_banner("First message from this sender looks like spam. Do not click links or share OTPs.", strong=True)
            else:
                self.add_banner("This message is suspected to be spam. Be cautious with links and requests.", strong=False)

        self.has_seen_message_from_contact.add(self.current_contact)

    # -----------------------------
    # Add banner
    # -----------------------------
    def add_banner(self, text: str, strong: bool = False):
        banner = WarningBanner(text, strong=strong)
        idx = self.chat_layout.count() - 1
        self.chat_layout.insertWidget(idx, banner, 0, Qt.AlignHCenter)
        self.smooth_scroll_to_bottom()

    # -----------------------------
    # Add chat bubble
    # -----------------------------
    def add_message_bubble(self, text: str = None, is_user: bool = False,
                           is_spam: bool = False, attachment_path: str = None):
        bubble = ChatBubble(
            sender="You" if is_user else self.current_contact,
            text=text,
            is_user=is_user,
            is_spam=is_spam,
            attachment_path=attachment_path
        )
        idx = self.chat_layout.count() - 1
        align = Qt.AlignRight if is_user else Qt.AlignLeft
        self.chat_layout.insertWidget(idx, bubble, 0, align)
        spacer = QWidget()
        spacer.setFixedHeight(2)
        self.chat_layout.insertWidget(idx + 1, spacer)
        self.smooth_scroll_to_bottom()

    # -----------------------------
    # Smooth scroll
    # -----------------------------
    def smooth_scroll_to_bottom(self):
        sb = self.scroll.verticalScrollBar()
        anim = QPropertyAnimation(sb, b"value")
        anim.setDuration(350)
        anim.setStartValue(sb.value())
        anim.setEndValue(sb.maximum())
        anim.start()
        self._anim = anim

    # -----------------------------
    # Choose attachment
    # -----------------------------
    def choose_attachment(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose attachment", "", "All Files (*.*)")
        if not path:
            return
        self.pending_attachment = path
        base = Path(path).name
        self.add_banner(f"Attachment ready: {base}. It will be sent with your next message.", strong=False)

    # -----------------------------
    # Simulate incoming message
    # -----------------------------
    def simulate_incoming(self):
        # Random text messages
        sample_messages = [
            "Hey! How are you doing today?",
            "Are we still meeting at 5 PM?",
            "Congratulations! You won a FREE reward. Click http://fake.link to claim now.",
            "Win a brand new iPhone! Visit http://fake.link now!",
            "Don't forget to submit your assignment by tonight.",
            "Your OTP is 123456. Do not share it with anyone.",
            "Limited time offer! Claim your free voucher now."
        ]
        msg = random.choice(sample_messages)
        attach_path = None

        # Let backend classify
        self.receive_message(msg, attachment_path=attach_path)

# -----------------------------
# Launch Application
# -----------------------------
if __name__ == "__main__":
    # Start FastAPI in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    # Launch GUI
    app = QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec_())
