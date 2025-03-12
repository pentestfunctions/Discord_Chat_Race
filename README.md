# Discord Chat Race 📊

<div align="center">

![Discord Chat Race Banner](https://raw.githubusercontent.com/pentestfunctions/Discord_Chat_Race/main/example_output.gif)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![Made with ❤️](https://img.shields.io/badge/made%20with-%E2%9D%A4%EF%B8%8F-red.svg)](https://github.com/pentestfunctions)

*Generate stunning animated bar chart races that visualize Discord chat activity over time*

</div>

<p align="center">
  <img src="https://github.com/pentestfunctions/Discord_Chat_Race/blob/main/example.gif">
</p>

## 🌟 Features

- **Complete Data Pipeline**: From message extraction to beautiful visualizations
- **Multiple Themes**: Choose from Discord, Neon, Pastel, or Midnight visual styles
- **Flexible Time Grouping**: Analyze activity by day, week, or month
- **Top User Focus**: Automatically highlight the most active community members
- **High-Quality Export**: Save as MP4 videos perfect for sharing
- **User-Friendly CLI**: Simple command-line interface with extensive options

## 📋 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/pentestfunctions/Discord_Chat_Race.git
    cd Discord_Chat_Race
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Quick Start

### Option 1: Command-line arguments

```bash
python discord_chat_race.py --token "YOUR_DISCORD_TOKEN" --server "SERVER_ID"
```

### Option 2: Environment variables (.env file)

Create a `.env` file in the project directory:
```
DISCORD_USER_TOKEN=your_token_here
SERVER_ID=your_server_id_here
```

Then run:
```bash
python discord_chat_race.py
```

## 🔧 Advanced Usage

### Extract messages only

```bash
python discord_chat_race.py
```


## 🎨 Available Themes

| Theme | Description |
|-------|-------------|
| `discord` | Discord's official dark theme with blurple accents |
| `neon` | Sleek dark theme with bright neon green highlights |
| `pastel` | Light, modern theme with soft color palette |
| `midnight` | Deep blue gradient theme with light blue accents |

## 📝 Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Operation mode: `extract`, `visualize`, or `both` | `both` |
| `--token` | Discord user token | From `.env` |
| `--server` | Discord server ID | From `.env` |
| `--channel` | Discord channel ID | From `.env` |
| `--max-messages` | Maximum messages to extract | All messages |
| `--csv` | Path to CSV file | Auto-generated |
| `--time-grouping` | Time grouping: `D` (daily), `W` (weekly), `M` (monthly) | `D` |
| `--theme` | Visualization theme | `discord` |
| `--max-users` | Maximum users to show | `20` |
| `--output` | Output video file path | Auto-generated |
| `--fps` | Frames per second for video | `30` |
| `--dpi` | DPI for saved video | `200` |

## 🔑 How to Get Discord Credentials

### Obtaining Your Discord User Token

1. Open Discord in your web browser
2. Press `F12` to open Developer Tools
3. Go to the "Application" tab
4. Under "Storage" > "Local Storage" > "https://discord.com"
5. Find the "token" key and copy its value

> ⚠️ **Important**: Never share your Discord token publicly. It grants full access to your account.

### Finding Server and Channel IDs

1. Enable Developer Mode in Discord (User Settings → Advanced → Developer Mode)
2. Right-click on a server icon and select "Copy ID" to get the Server ID


<div align="center">
  Made with 💬 by <a href="https://github.com/pentestfunctions">pentestfunctions</a>
</div>
