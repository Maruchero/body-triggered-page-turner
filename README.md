# Body-Triggered Page Turner

A hands-free background utility for musicians. This daemon runs quietly in the background, allowing you to turn pages in your favorite PDF viewer (like Okular, Acrobat, or Evince) using simple body gestures. Designed for convertible laptops and tablets running Windows or Linux.

## 🎵 Features

- **Background Daemon**: Runs unobtrusively in the background without a distracting GUI.
- **Universal Compatibility**: Works with **any** application that accepts keyboard input (optimized for PDF viewers like Okular).
- **Gesture Control**: Simulates key presses (e.g., `Page Down`, `Arrow Keys`) triggered by facial expressions like a smile or head tilt.
- **Cross-Platform**: Built to run on Linux and Windows.
- **Low Latency**: Immediate response to ensure seamless performance.

## 🛠️ Planned Tech Stack

- **Language**: Python 3.x
- **Computer Vision**: OpenCV & MediaPipe (for robust face and gesture tracking)
- **Input Simulation**: `pynput` or `pyautogui` (to simulate keyboard events)
- **System Integration**: System tray icon for easy toggling/calibration (optional)

## 🚀 Getting Started (Draft)

### Prerequisites

- A laptop with a webcam.
- Python 3.10+ installed.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/body-triggered-page-turner.git
   cd body-triggered-page-turner
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Start the daemon:
   ```bash
   python main.py --daemon
   ```
2. Open your music sheet in your preferred viewer (e.g., **Okular**).
3. Focus the PDF viewer window.
4. Perform the configured gesture (e.g., smile) to trigger a page turn.

## 🤝 Contributing

Contributions are welcome! Whether it's adding new gestures, improving tracking accuracy, or fixing bugs, feel free to open an issue or submit a pull request.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.