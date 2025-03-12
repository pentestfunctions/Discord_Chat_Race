import requests
import json
import csv
import datetime
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.patheffects as path_effects
import random
import colorsys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PART 1: DISCORD MESSAGE EXTRACTION
# -----------------------------------

def extract_messages(channel_id, user_token, limit=100, before=None, max_messages=None):
    """
    Extract messages from a channel using Discord API directly.
    
    Args:
        channel_id: The Discord channel ID to extract messages from
        user_token: Discord user token for authentication
        limit: Number of messages to request per API call (max 100)
        before: Message ID to fetch messages before
        max_messages: Maximum number of messages to extract (None for all)
    
    Returns:
        List of message data (author, timestamp)
    """
    messages = []
    base_url = f'https://discord.com/api/v9/channels/{channel_id}/messages'
    
    # Headers for API requests
    headers = {
        'Authorization': user_token,
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    print(f"Starting to extract messages from channel {channel_id}...")
    
    # Continue fetching messages until we get all or hit an error
    total_messages = 0
    has_more = True
    
    while has_more and (max_messages is None or total_messages < max_messages):
        # Determine how many messages to request in this batch
        batch_limit = limit
        if max_messages is not None:
            batch_limit = min(limit, max_messages - total_messages)
            
        # Prepare query parameters
        params = {'limit': batch_limit}
        if before:
            params['before'] = before
            
        # Make API request
        response = requests.get(base_url, headers=headers, params=params)
        
        # Check if request was successful
        if response.status_code == 200:
            batch = response.json()
            
            # If we got no messages, we're done
            if not batch:
                has_more = False
                continue
                
            # Process messages in this batch
            for msg in batch:
                # Convert timestamp to a readable format
                created_at = datetime.datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
                timestamp = created_at.strftime("%Y-%m-%d %H:%M:%S")
                
                # Extract author information (account for new username system)
                if 'discriminator' in msg['author'] and msg['author']['discriminator'] != '0':
                    author = f"{msg['author']['username']}#{msg['author']['discriminator']}"
                else:
                    author = msg['author']['username']
                
                # Append only username and timestamp to our list
                messages.append({
                    'author': author,
                    'timestamp': timestamp,
                    'message_id': msg['id']
                })
            
            # Update the 'before' parameter to get the next batch
            before = batch[-1]['id']
            
            # Update total and print progress
            total_messages += len(batch)
            print(f"Extracted {total_messages} messages so far...")
            
            # If we got fewer messages than we asked for, we're done
            if len(batch) < batch_limit:
                has_more = False
        else:
            print(f"Error: {response.status_code} - {response.text}")
            has_more = False
    
    print(f"Finished extracting {len(messages)} messages.")
    return messages

def save_to_csv(messages, filename="discord_messages.csv"):
    """Save extracted messages to a CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Define CSV columns - only username and timestamp
        fieldnames = ['message_id', 'timestamp', 'author']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header and data
        writer.writeheader()
        for message in messages:
            writer.writerow(message)
    
    print(f"Messages saved to {filename}")


# PART 2: DATA VISUALIZATION
# --------------------------

def prepare_discord_data(csv_file, time_grouping='D'):
    """
    Prepare Discord message data for visualization.
    
    Args:
        csv_file: Path to CSV file with Discord messages
        time_grouping: How to group timestamps ('D' for daily, 'W' for weekly, 'M' for monthly)
    
    Returns:
        DataFrame ready for animation
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract date part based on grouping
    if time_grouping == 'D':
        df['date'] = df['timestamp'].dt.date
    elif time_grouping == 'W':
        df['date'] = df['timestamp'].dt.to_period('W').dt.start_time.dt.date
    elif time_grouping == 'M':
        df['date'] = df['timestamp'].dt.to_period('M').dt.start_time.dt.date
    
    # Count messages by author and date
    message_counts = df.groupby(['date', 'author']).size().reset_index(name='count')
    
    # Pivot the data to get authors as columns
    pivot_df = message_counts.pivot(index='date', columns='author', values='count').fillna(0)
    
    # Create cumulative sums for each author
    cumulative_df = pivot_df.cumsum()
    
    # Ensure the index is in datetime format
    cumulative_df.index = pd.to_datetime(cumulative_df.index)
    
    # Sort by date
    cumulative_df = cumulative_df.sort_index()
    
    return cumulative_df

# Theme options
THEMES = {
    'discord': {
        'background': '#36393F',
        'text_color': '#FFFFFF',
        'grid_color': '#40444B',
        'accent_color': '#7289DA',
        'bar_effect': 'gradient',
        'title_color': '#FFFFFF',
        'progress_color': '#7289DA'
    },
    'neon': {
        'background': '#121212',
        'text_color': '#FFFFFF',
        'grid_color': '#222222',
        'accent_color': '#00FF41',
        'bar_effect': 'glow',
        'title_color': '#00FF41',
        'progress_color': '#00FF41'
    },
    'pastel': {
        'background': '#F5F7FA',
        'text_color': '#333333',
        'grid_color': '#E4E7EB',
        'accent_color': '#74B9FF',
        'bar_effect': 'simple',
        'title_color': '#5F6A87',
        'progress_color': '#A3BFFA'
    },
    'midnight': {
        'background': '#0F2027',
        'text_color': '#E0E0E0',
        'grid_color': '#203A43',
        'accent_color': '#6DD5FA',
        'bar_effect': 'gradient',
        'title_color': '#FFFFFF',
        'progress_color': '#2980B9'
    }
}

# Generate improved colors based on theme
def generate_theme_colors(n, theme_name):
    """Generate n visually distinct colors that work with the selected theme"""
    
    base_color = THEMES[theme_name]['accent_color']
    
    # Convert hex to HSV
    base_h, base_s, base_v = colorsys.rgb_to_hsv(
        *[int(base_color[i:i+2], 16)/255 for i in (1, 3, 5)]
    )
    
    colors = []
    for i in range(n):
        # Rotate hue around color wheel with slight variations
        hue = (base_h + (i / n)) % 1.0
        saturation = 0.65 + (random.random() * 0.3)
        value = 0.7 + (random.random() * 0.3)
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert to hex
        colors.append(f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}')
    
    # Shuffle to ensure that similar colors aren't adjacent
    random.shuffle(colors)
    return colors

def create_animation(csv_file, time_grouping='D', theme_name='discord', max_users=20, 
                     output_file=None, fps=30, dpi=200):
    """
    Create a bar chart race animation from Discord message data.
    
    Args:
        csv_file: Path to CSV file with Discord messages
        time_grouping: How to group timestamps ('D' for daily, 'W' for weekly, 'M' for monthly)
        theme_name: Theme to use for the animation
        max_users: Maximum number of users to show
        output_file: Path to save the animation (None to just display)
        fps: Frames per second for the saved animation
        dpi: DPI for the saved animation
    """
    # Get theme
    theme = THEMES[theme_name]
    
    # Load the Discord message data
    full_df = prepare_discord_data(csv_file, time_grouping)
    
    # Get the final message counts to identify top users
    final_counts = full_df.iloc[-1].sort_values(ascending=False)
    
    # Get top N users
    top_users = final_counts.head(max_users).index.tolist()
    
    # Filter dataframe to only include top users
    raw_df = full_df[top_users]
    
    # Interpolate to create more data points for smoother animation
    steps_per_period = 5
    smooth_index = pd.date_range(
        start=raw_df.index.min(),
        end=raw_df.index.max(),
        periods=(len(raw_df.index)-1) * steps_per_period + 1
    )
    
    # Create an empty dataframe for interpolation
    smooth_df = pd.DataFrame(index=smooth_index, columns=raw_df.columns)
    
    # Perform linear interpolation between existing data points
    for col in raw_df.columns:
        smooth_df[col] = np.interp(
            x=range(len(smooth_index)),
            xp=[i * steps_per_period for i in range(len(raw_df.index))],
            fp=raw_df[col].values
        )
    
    # Use the interpolated dataframe for animation
    df = smooth_df
    
    # Set up colors with a theme-appropriate palette
    colors = generate_theme_colors(len(df.columns), theme_name)
    color_map = {user: color for user, color in zip(df.columns, colors)}
    
    # Function to draw a single frame
    def draw_barchart(frame):
        current_date = df.index[frame]
        # Get data for this frame and sort by value
        df_frame = df.loc[current_date].sort_values(ascending=True)
        
        # Only keep users with at least 1 message
        df_frame = df_frame[df_frame > 0]
        
        # Clear the current figure
        plt.clf()
        
        # Create horizontal bars
        ax = plt.gca()
        bars = ax.barh(
            df_frame.index, 
            df_frame.values, 
            color=[color_map[user] for user in df_frame.index], 
            height=0.8,
            alpha=0.9,
            edgecolor=theme['background'],
            linewidth=0.5
        )
        
        # Apply theme-specific bar effects
        if theme['bar_effect'] == 'gradient':
            for bar in bars:
                bar_width = bar.get_width()
                bar_height = bar.get_height()
                # Add subtle highlight/gradient
                ax.barh(
                    bar.get_y() + bar_height/2, 
                    bar_width * 0.9, 
                    height=bar_height * 0.7, 
                    left=bar.get_x() + bar_width * 0.05,
                    color='white', 
                    alpha=0.1
                )
        elif theme['bar_effect'] == 'glow':
            # Add subtle glow around bars
            for bar in bars:
                # Add multiple layers with decreasing alpha for glow effect
                for i in range(3):
                    width = bar.get_width()
                    height = bar.get_height() * (1 + i*0.1)
                    alpha = 0.04 * (3-i)
                    ax.barh(
                        bar.get_y() + bar.get_height()/2,
                        width,
                        height=height,
                        left=bar.get_x(),
                        color=theme['accent_color'],
                        alpha=alpha,
                        edgecolor=None
                    )
        
        # Remove axes spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Add value labels on the bars
        for i, (value, user) in enumerate(zip(df_frame.values, df_frame.index)):
            # Truncate long usernames to fit better
            display_name = user.split('#')[0] if '#' in user else user
            if len(display_name) > 15:
                display_name = display_name[:12] + '...'
                
            # Add count label at end of bar
            text_effect = [path_effects.withStroke(linewidth=2, foreground=theme['background'], alpha=0.3)]
            ax.text(value + 0.5, i, f'{int(value)}', ha='left', va='center', fontsize=10, 
                    color=theme['text_color'], path_effects=text_effect)
            
            # Add username inside bar
            ax.text(0.2, i, display_name, ha='left', va='center', fontsize=10, color='white', 
                    fontweight='bold', path_effects=text_effect)
        
        # Add date as chart title with cleaner formatting
        title = f'Discord Messages Over Time (as of {current_date.strftime("%Y-%m-%d")})'
        plt.title(title, fontsize=16, pad=20, fontweight='bold', loc='center', color=theme['title_color'])
        
        # Add subtitle if desired
        if time_grouping == 'D':
            grouping_text = "Daily Grouping"
        elif time_grouping == 'W':
            grouping_text = "Weekly Grouping"
        else:
            grouping_text = "Monthly Grouping"
        
        plt.suptitle(grouping_text, fontsize=10, color=theme['text_color'], alpha=0.7, y=0.91)
        
        # Customize x-axis
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.label.set_color(theme['text_color'])
        ax.tick_params(axis='x', colors=theme['text_color'], which='both')
        
        # Set axis limits - add small buffer for smooth appearance
        max_value = df.max().max() * 1.1
        ax.set_xlim(0, max_value)
        
        # Add subtle grid lines
        ax.grid(axis='x', linestyle='--', alpha=0.2, color=theme['grid_color'])
        
        # Remove y-axis ticks and labels
        ax.yaxis.set_ticks([])
        
        # Add animation progress bar
        progress_position = frame / (len(df.index) - 1)
        progress_height = -0.6
        
        # Background for progress bar (track)
        ax.axhline(y=progress_height, xmin=0, xmax=1, color=theme['grid_color'], alpha=0.3, linewidth=4)
        
        # Actual progress bar
        ax.axhline(y=progress_height, xmin=0, xmax=progress_position, color=theme['progress_color'], alpha=0.7, linewidth=4)
        
        # Add small label near progress bar with the current date in simpler format
        date_label = current_date.strftime("%b %d, %Y")
        ax.text(max_value * 0.01, progress_height - 0.5, date_label, color=theme['text_color'], 
                fontsize=9, ha='left', va='center', alpha=0.7)
        
        # Display note about top users
        if max_users:
            plt.figtext(0.5, 0.01, f"Showing only the top {max_users} most active users", 
                       ha="center", fontsize=9, color=theme['text_color'], alpha=0.7)
        
        plt.box(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make room for progress bar
    
    # Create the animation with theme
    plt.rcParams.update({'font.size': 12, 'text.color': theme['text_color'], 'axes.labelcolor': theme['text_color']})
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=theme['background'])
    fig.patch.set_facecolor(theme['background'])
    
    # Animation settings
    animator = animation.FuncAnimation(
        fig, draw_barchart, frames=len(df.index), interval=50,
        blit=False
    )
    
    # Save animation if output file is specified
    if output_file:
        print(f"Saving animation to {output_file}...")
        animator.save(output_file, 
                    writer=animation.FFMpegWriter(fps=fps, bitrate=2000),
                    dpi=dpi)
        print(f"Animation saved to {output_file}")
    
    # Display the animation (this will block until the window is closed)
    plt.show()
    
    return animator


# PART 3: MAIN FUNCTION
# ---------------------

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Discord Message Analyzer and Visualizer')
    
    # Main operation mode
    parser.add_argument('--mode', choices=['extract', 'visualize', 'both'], default='both',
                      help='Operation mode: extract messages, visualize existing data, or both')
    
    # Discord API parameters
    parser.add_argument('--token', help='Discord user token (or set DISCORD_USER_TOKEN env var)')
    parser.add_argument('--server', help='Discord server ID (or set SERVER_ID env var)')
    parser.add_argument('--channel', help='Discord channel ID (or set CHANNEL_ID env var)')
    parser.add_argument('--max-messages', type=int, default=None, 
                      help='Maximum number of messages to extract (default: all)')
    
    # Visualization parameters
    parser.add_argument('--csv', help='Path to CSV file (default: auto-generated)',
                      default=None)
    parser.add_argument('--time-grouping', choices=['D', 'W', 'M'], default='D',
                     help='Time grouping: D=daily, W=weekly, M=monthly (default: D)')
    parser.add_argument('--theme', choices=list(THEMES.keys()), default='discord',
                      help=f'Theme for visualization (default: discord)')
    parser.add_argument('--max-users', type=int, default=20,
                      help='Maximum number of users to show in visualization (default: 20)')
    parser.add_argument('--output', help='Output video file path (default: no save)',
                      default=None)
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second for saved video (default: 30)')
    parser.add_argument('--dpi', type=int, default=200,
                      help='DPI for saved video (default: 200)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get token and channel from arguments or environment variables
    user_token = args.token or os.getenv('DISCORD_USER_TOKEN')
    server_id = args.server or os.getenv('SERVER_ID')
    channel_id = args.channel or os.getenv('CHANNEL_ID')
    
    # Generate or use provided CSV filename
    csv_file = args.csv
    if not csv_file and (args.mode == 'extract' or args.mode == 'both'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"discord_messages_{server_id}_{channel_id}_{timestamp}.csv"
        # Remove characters that are not allowed in filenames
        csv_file = "".join(c for c in csv_file if c.isalnum() or c in "._- ")
    
    # Default output filename if not provided
    output_file = args.output
    if not output_file and (args.mode == 'visualize' or args.mode == 'both'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"discord_visualization_{timestamp}.mp4"
    
    try:
        # Extract messages
        if args.mode == 'extract' or args.mode == 'both':
            if not user_token or not channel_id:
                raise ValueError(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MISSING REQUIRED PARAMETERS                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Discord token, Server ID and Channel ID are required for extraction.

You can provide these values in one of two ways:

1. Command-line arguments:
   python discord_chat_race.py --token MZ..... --server 123456 --channel 123456

2. Environment variables in a .env file:
   Create a file named '.env' in the same directory with these contents (No quotes):
   
   DISCORD_USER_TOKEN=your_token_here
   SERVER_ID=your_server_id_here
   CHANNEL_ID=your_channel_id_here

How to find these values:
    • Token: Log into Discord web, open DevTools (F12), check Local Storage
    • Server ID: Right-click on server icon → Copy ID (Developer Mode required)
    • Channel ID: Right-click on channel name → Copy ID (Developer Mode required)

To enable Developer Mode: User Settings → Advanced → Developer Mode
""")
            
            print(f"Extracting messages from Discord channel {channel_id}...")
            messages = extract_messages(
                channel_id=channel_id,
                user_token=user_token,
                max_messages=args.max_messages
            )
            
            save_to_csv(messages, csv_file)
        
        # Visualize data
        if args.mode == 'visualize' or args.mode == 'both':
            if not csv_file:
                raise ValueError("CSV file path required for visualization")
            
            print(f"Creating visualization from {csv_file}...")
            create_animation(
                csv_file=csv_file,
                time_grouping=args.time_grouping,
                theme_name=args.theme,
                max_users=args.max_users,
                output_file=output_file,
                fps=args.fps,
                dpi=args.dpi
            )
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the script
if __name__ == "__main__":
    main()
