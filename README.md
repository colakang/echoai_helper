# 🎙️ EchoAI Helper - Your Real-time Conversation Assistant

[![GitHub Stars](https://img.shields.io/github/stars/colakang/echoai_helper?style=social)](https://github.com/colakang/echoai_helper/stargazers)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-≥3.8-blue.svg)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com)

EchoAI Helper is a powerful real-time conversation assistant that provides instant transcription and intelligent responses. It captures both microphone input and speaker output, making it perfect for meetings, interviews, or any scenario where you need real-time conversation analysis.

<p>
<a href="https://www.producthunt.com/posts/echoai-interview-copilot?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-echoai&#0045;interview&#0045;copilot" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=601490&theme=light" alt="EchoAI&#0032;Interview&#0032;Copilot&#0032; - Real&#0045;time&#0032;conversation&#0032;with&#0032;LLM&#0032;responses | Product Hunt" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</p>

<p align="center">
<img width="800" alt="EchoAI Helper Interface" src="https://github.com/colakang/echoai_helper/raw/main/images/ui.png">
</p>

## ✨ Features

- **Local ASR Support** - CPU-based speech recognition without cloud dependency
- **Flexible ASR Options** 
  - Local ASR: FunASR for offline, privacy-focused transcription
  - Cloud ASR: Optional OpenAI Whisper API integration for enhanced accuracy
- **Real-time Transcription** - Simultaneous transcription of both microphone input and speaker output
- **Intelligent Response Generation** - Contextually aware responses powered by OpenAI GPT models
- **Dual Audio Source Support** - Captures both microphone and system audio simultaneously
- **Conversation History** - Complete transcript storage and export capabilities
- **Customizable Response Templates** - Flexible system roles and knowledge base configuration
- **Buffer Management** - Advanced audio buffering for optimal transcription quality
- **Export Functionality** - Save conversations in JSON format for future reference
- **User-friendly Interface** - Clean and intuitive UI built with CustomTkinter

## 💡 Use Cases

- Real-time meeting transcription and assistance
- Interview transcription and analysis
- Live presentation with AI support
- Customer service conversation enhancement

## 🎬 Demo Video

https://github.com/user-attachments/assets/0d627e4a-960b-4628-8bbc-8d892f02cfd1


## 🆕 What's NEW

- Added local CPU-based ASR using FunASR - no cloud service required
- Added FunASR for improved multilingual support
- Enhanced response generation with context awareness
- Introduced customizable templates system
- Added conversation export functionality
- Improved audio buffer management
- Enhanced UI responsiveness

## 📝 TODO

We're actively working on new features to make EchoAI Helper even better:

### Coming Soon 🚀
- [ ] Smart sentence completion detection
 - Auto-detect sentence completeness
 - Improve transcription accuracy
 - Optimize response timing

- [ ] Enhanced Software Integration
 - Transparent overlay support
 - Easy attachment to any meeting software

- [ ] Installation & Platform Support
 - [ ] One-click Windows installer
 - [ ] macOS support (Intel)
 - [ ] macOS support (Apple Silicon)
 - [ ] Streamlined setup process

### Future Plans 🔮
- Cross-platform compatibility optimization
- Enhanced integration capabilities
- Performance improvements for various hardware

Want to contribute? Check out our [contribution guidelines](CONTRIBUTING.md)!

## 🔧 Prerequisites

### Required
- Python ≥ 3.8.0
- FFmpeg
- Windows OS (Other platforms not fully tested)

### Accounts & API Keys
- OpenAI API key (paid account required)

## ⚡ Quick Start

```bash
# Create conda environment
conda create -n echoai python=3.10.13
conda activate echoai

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env file and add your OpenAI API key
```

## 📦 Detailed Installation

### 1. Clone the repository
```bash
git clone https://github.com/colakang/echoai_helper.git
cd echoai_helper
```

### 2. Set up Python environment
```bash
# Using conda (recommended)
conda create -n echoai python=3.10.13
conda activate echoai

# Install dependencies
pip install -r requirements.txt
pip install -U funasr
pip install torch
pip install -U modelscope huggingface_hub
pip install "numpy<2.0"
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3. Install FFmpeg (Windows)
Using Chocolatey (Run PowerShell as Administrator):
```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install ffmpeg
```

### 4. Configure Environment Variables
1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit the `.env` file and configure your settings:
```plaintext
# OpenAI Configuration
OPENAI_API_KEY=your-api-key-here
```

### 5. Verify Installation
```bash
# Start the application
python main.py

```

> 📝 **Note:** Make sure to keep your `.env` file secure and never commit it to version control. The `.gitignore` file is already configured to exclude it.

## 🎯 Usage

1. Start the application:
```bash
python main.py
```

2. The interface will show two main sections:
   - Left panel: Real-time transcription
   - Right panel: AI-generated responses

3. Customize settings using the control panel:
   - Adjust phrase timeout
   - Configure buffer chunks
   - Select templates
   - Export conversations
   - Manual popup current sentence

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📞 Contact

- Website: [EchoAI](https://www.echo365.ai)
- Issues: [GitHub Issues](https://github.com/colakang/echoai_helper/issues)

Project Link: [https://github.com/colakang/echoai_helper](https://github.com/colakang/echoai_helper)

## 🙌 Credits & Inspiration

This project wouldn't be possible without these amazing projects and tools:

### 🛠️ Core Technologies
- [FunASR](https://github.com/modelscope/FunASR) - For state-of-the-art speech recognition
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) - For modern UI components
- [FFmpeg](https://ffmpeg.org/) - For audio processing capabilities

### 🌟 Related Projects
We've drawn inspiration and learned from these excellent projects:

- [Ecoute](https://github.com/SevaSk/ecoute)

### 🤝 Special Thanks
- [@zixing0131](https://github.com/zixing0131) - For implementing core audio processing components

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

