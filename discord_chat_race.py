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
import time

# Load environment variables from .env file
load_dotenv()

# PART 1: DISCORD CHANNEL ENUMERATION
# -----------------------------------

def enumerate_channels(server_id, user_token):
    """
    Enumerate all channels in a Discord server
    
    Args:
        server_id: Discord server ID
        user_token: Discord user token for authentication
        
    Returns:
        A dictionary mapping channel IDs to their names and types
    """
    # Headers for API requests
    headers = {
        'Authorization': user_token,
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Get server information to validate token and server ID
    try:
        server_response = requests.get(
            f'https://discord.com/api/v9/guilds/{server_id}',
            headers=headers
        )
        
        if server_response.status_code == 200:
            server_data = server_response.json()
            print(f"Successfully connected to server: {server_data.get('name')}")
        else:
            print(f"Error accessing server: {server_response.status_code} - {server_response.text}")
            return None
    except Exception as e:
        print(f"Error connecting to Discord API: {e}")
        return None
    
    # Get all channels in the server
    try:
        channel_response = requests.get(
            f'https://discord.com/api/v9/guilds/{server_id}/channels',
            headers=headers
        )
        
        if channel_response.status_code == 200:
            channels = channel_response.json()
            print(f"Successfully retrieved {len(channels)} channels from server")
        else:
            print(f"Error getting channels: {channel_response.status_code} - {channel_response.text}")
            return None
    except Exception as e:
        print(f"Error retrieving channels: {e}")
        return None
    
    # Process the channels into a dict
    channel_info = {}
    text_channels = []
    
    # First, build a map of category IDs to names
    categories = {}
    for channel in channels:
        if channel['type'] == 4:  # Category
            categories[channel['id']] = channel['name']
    
    # Now process all channels
    for channel in channels:
        # Skip categories themselves
        if channel['type'] == 4:
            continue
            
        # Get channel type
        channel_type = "Unknown"
        if channel['type'] == 0:
            channel_type = "Text"
            text_channels.append({
                'id': channel['id'],
                'name': channel['name'],
                'parent': categories.get(channel.get('parent_id', ''), 'Uncategorized')
            })
        elif channel['type'] == 2:
            channel_type = "Voice"
        elif channel['type'] == 5:
            channel_type = "Announcement"
            text_channels.append({
                'id': channel['id'],
                'name': channel['name'],
                'parent': categories.get(channel.get('parent_id', ''), 'Uncategorized')
            })
        elif channel['type'] == 13:
            channel_type = "Stage"
        elif channel['type'] == 15:
            channel_type = "Forum"
        elif channel['type'] == 16:
            channel_type = "MediaChannel"
        
        # Add to our mapping
        channel_info[channel['id']] = {
            'name': channel['name'],
            'type': channel_type,
            'category': categories.get(channel.get('parent_id', ''), 'Uncategorized')
        }
    
    # Print text channels for user selection
    print("\nAvailable text channels:")
    print("------------------------")
    
    # Group by category for better organization
    by_category = {}
    for channel in text_channels:
        if channel['parent'] not in by_category:
            by_category[channel['parent']] = []
        by_category[channel['parent']].append(channel)
    
    # Sort categories alphabetically
    sorted_categories = sorted(by_category.keys())
    
    # Print channels grouped by category
    for category in sorted_categories:
        print(f"\n{category.upper()}:")
        for i, channel in enumerate(sorted(by_category[category], key=lambda x: x['name'])):
            print(f"  - {channel['name']} (ID: {channel['id']})")
    
    return channel_info

# PART 2: DISCORD MESSAGE EXTRACTION
# -----------------------------------

def extract_messages(channel_id, user_token, channel_name="Unknown", limit=100, before=None, max_messages=None):
    """
    Extract messages from a channel using Discord API directly.
    
    Args:
        channel_id: The Discord channel ID to extract messages from
        user_token: Discord user token for authentication
        channel_name: The name of the channel (for display only)
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
    
    print(f"Starting to extract messages from channel: (ID: {channel_id})...")
    
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
                    'message_id': msg['id'],
                    'channel_id': channel_id,
                    'channel_name': channel_name
                })
            
            # Update the 'before' parameter to get the next batch
            before = batch[-1]['id']
            
            # Update total and print progress
            total_messages += len(batch)
            print(f"  Extracted {total_messages} messages so far...")
            
            # If we got fewer messages than we asked for, we're done
            if len(batch) < batch_limit:
                has_more = False
        elif response.status_code == 429:  # Rate limited
            # Parse rate limit information
            retry_after = response.json().get('retry_after', 5)
            print(f"  Rate limited. Waiting for {retry_after} seconds...")
            time.sleep(retry_after + 0.5)  # Add a small buffer
        else:
            print(f"  Error: {response.status_code} - {response.text}")
            has_more = False
    
    print(f"  Finished extracting {len(messages)} messages")
    return messages

def save_to_csv(messages, filename="discord_messages.csv"):
    """Save extracted messages to a CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Define CSV columns
        fieldnames = ['message_id', 'timestamp', 'author', 'channel_id', 'channel_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header and data
        writer.writeheader()
        for message in messages:
            writer.writerow(message)
    
    print(f"Messages saved to {filename}")

def extract_server_messages(server_id, user_token, channel_ids=None, max_messages=None, output_dir="output"):
    """Extract messages from multiple channels in a server"""
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # If no specific channel IDs provided, get all text channels
    all_channels = None
    if channel_ids is None:
        all_channels = enumerate_channels(server_id, user_token)
        if not all_channels:
            print("Failed to retrieve channels from server.")
            return None
        
        # Filter for text channels only
        text_channels = {
            id: info for id, info in all_channels.items() 
            if info['type'] in ['Text', 'Announcement']
        }
        
        channel_ids = list(text_channels.keys())
    
    # Check if we have any channels to process
    if not channel_ids:
        print("No text channels found in server.")
        return None
    
    print(f"\nExtracting messages from {len(channel_ids)} channels...")
    
    # Extract messages from each channel
    all_messages = []
    combined_csv = os.path.join(output_dir, f"discord_server_{server_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    for i, channel_id in enumerate(channel_ids):
        # Get channel name if we have it
        channel_name = "Unknown"
        if all_channels and channel_id in all_channels:
            channel_name = all_channels[channel_id]['name']
        
        print(f"\nProcessing channel {i+1}/{len(channel_ids)}")
        
        # Extract messages for this channel
        channel_messages = extract_messages(
            channel_id=channel_id,
            user_token=user_token,
            channel_name=channel_name,
            max_messages=max_messages
        )
        
        # Save channel-specific CSV
        if channel_messages:
            # Add to combined list
            all_messages.extend(channel_messages)
            
            # Save individual channel data
            channel_csv = os.path.join(output_dir, f"discord_channel_{channel_id}.csv")
            save_to_csv(channel_messages, channel_csv)
    
    # Save combined CSV with all messages
    if all_messages:
        save_to_csv(all_messages, combined_csv)
        print(f"\nComplete! Extracted {len(all_messages)} total messages across {len(channel_ids)} channels.")
        return combined_csv
    
    print("No messages were extracted.")
    return None


# PART 3: DATA VISUALIZATION
# --------------------------

def prepare_discord_data(csv_file, time_grouping='D', channel_filter=None):
    """
    Prepare Discord message data for visualization.
    
    Args:
        csv_file: Path to CSV file with Discord messages
        time_grouping: How to group timestamps ('D' for daily, 'W' for weekly, 'M' for monthly)
        channel_filter: Only include messages from this channel ID (None for all channels)
    
    Returns:
        DataFrame ready for animation
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Apply channel filter if specified
    if channel_filter:
        if 'channel_id' in df.columns:
            df = df[df['channel_id'] == channel_filter]
        else:
            print("Warning: CSV doesn't contain channel_id column, cannot filter by channel.")
    
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
                     output_file=None, fps=30, dpi=200, channel_filter=None, channel_name=None,
                     server_name=None, server_id=None):
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
        channel_filter: Only include messages from this channel ID (None for all channels)
        channel_name: Name of the channel (for display purposes)
        server_name: Name of the server (for display purposes)
        server_id: ID of the server (for display purposes)
    """
    # Get theme
    theme = THEMES[theme_name]
    
    # Load the Discord message data
    full_df = prepare_discord_data(csv_file, time_grouping, channel_filter)
    
    # Make sure we have data
    if full_df.empty:
        print("No data to visualize. The CSV may be empty or filtering removed all entries.")
        return None
    
    # Get the final message counts to identify top users
    final_counts = full_df.iloc[-1].sort_values(ascending=False)
    
    # Get top N users
    top_users = final_counts.head(max_users).index.tolist()
    
    # Filter dataframe to only include top users
    raw_df = full_df[top_users]
    
    # Ensure we have at least some data
    if raw_df.empty:
        print("No data to visualize after filtering for top users.")
        return None
    
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
        
        # Add title with channel info if specified
        if channel_name:
            title = f'Discord Messages in current channel (as of {current_date.strftime("%Y-%m-%d")})'
        else:
            title = f'Discord Messages Over Time (as of {current_date.strftime("%Y-%m-%d")})'
        plt.title(title, fontsize=16, pad=20, fontweight='bold', loc='center', color=theme['title_color'])
        
        # Add subtitle with server information instead of time grouping
        server_display = ""
        if server_name and server_id:
            server_display = f"Server: {server_name} (ID: {server_id})"
        elif server_name:
            server_display = f"Server: {server_name}"
        elif server_id:
            server_display = f"Server ID: {server_id}"
        
        if server_display:
            plt.suptitle(server_display, fontsize=10, color=theme['text_color'], alpha=0.7, y=0.91)
        
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
    #plt.show()
    print(f"Video file is ready as long as ffmpeg worked correctly, check the output folder for the mp4")
    return animator


# PART 4: MAIN FUNCTION
# ---------------------

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Discord Message Analyzer and Visualizer')
    
    # Main operation mode - simplify to focus on server-wide analysis
    parser.add_argument('--mode', choices=['extract', 'visualize', 'both'], default='both',
                      help='Operation mode: extract messages, visualize existing data, or both (default: both)')
    
    # Discord API parameters
    parser.add_argument('--token', help='Discord user token (or set DISCORD_USER_TOKEN env var)')
    parser.add_argument('--server', help='Discord server ID (or set SERVER_ID env var)')
    parser.add_argument('--max-messages', type=int, default=None, 
                      help='Maximum number of messages to extract per channel (default: all)')
    
    # Visualization parameters
    parser.add_argument('--csv', help='Path to CSV file (default: auto-generated)',
                      default=None)
    parser.add_argument('--time-grouping', choices=['D', 'W', 'M'], default='D',
                     help='Time grouping: D=daily, W=weekly, M=monthly (default: D)')
    parser.add_argument('--theme', choices=list(THEMES.keys()), default='discord',
                      help=f'Theme for visualization (default: discord)')
    parser.add_argument('--max-users', type=int, default=20,
                      help='Maximum number of users to show in visualization (default: 20)')
    parser.add_argument('--output', help='Output video file path (default: auto-generated)',
                      default=None)
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second for saved video (default: 30)')
    parser.add_argument('--dpi', type=int, default=200,
                      help='DPI for saved video (default: 200)')
    parser.add_argument('--output-dir', default='output',
                      help='Directory to save output files (default: output)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get token and IDs from arguments or environment variables
    user_token = args.token or os.getenv('DISCORD_USER_TOKEN')
    server_id = args.server or os.getenv('SERVER_ID')
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate or use provided CSV filename
    csv_file = args.csv
    
    # Default output filename if not provided
    output_file = args.output
    if not output_file and (args.mode == 'visualize' or args.mode == 'both'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"discord_server_{server_id}_visualization_{timestamp}.mp4")
    
    try:
        # First, ensure we have the required parameters
        if not user_token or not server_id:
            raise ValueError(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          MISSING REQUIRED PARAMETERS                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Discord token and Server ID are required for server-wide analysis.

You can provide these values in one of two ways:

1. Command-line arguments:
   python discord_chat_race.py --token MZ..... --server 123456

2. Environment variables in a .env file:
   Create a file named '.env' in the same directory with these contents (No quotes):
   
   DISCORD_USER_TOKEN=your_token_here
   SERVER_ID=your_server_id_here

How to find these values:
    • Token: Log into Discord web, open DevTools (F12), check Local Storage
    • Server ID: Right-click on server icon → Copy ID (Developer Mode required)

To enable Developer Mode: User Settings → Advanced → Developer Mode
""")
        
        # Get server name automatically for display
        server_name = None
        try:
            headers = {
                'Authorization': user_token,
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            server_response = requests.get(
                f'https://discord.com/api/v9/guilds/{server_id}',
                headers=headers
            )
            
            if server_response.status_code == 200:
                server_data = server_response.json()
                server_name = server_data.get('name')
                print(f"Server name: {server_name}")
            else:
                print(f"Could not fetch server name: HTTP {server_response.status_code}")
        except Exception as e:
            print(f"Error fetching server name: {e}")
        
        # Extract mode: Get messages from all server channels
        if args.mode == 'extract' or args.mode == 'both':
            print(f"Enumerating all channels in server {server_id}...")
            
            # Get all channels in the server
            all_channels = enumerate_channels(server_id, user_token)
            if not all_channels:
                raise ValueError("Failed to retrieve channels from server.")
            
            # Filter for text channels only
            text_channels = {
                id: info for id, info in all_channels.items() 
                if info['type'] in ['Text', 'Announcement']
            }
            
            channel_ids = list(text_channels.keys())
            
            if not channel_ids:
                raise ValueError("No text channels found in server.")
            
            print(f"\nFound {len(channel_ids)} text channels in server.")
            
            # If CSV file not specified, create one with timestamp and server ID
            if not csv_file:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = os.path.join(args.output_dir, f"discord_server_{server_id}_{timestamp}.csv")
            
            # Extract all messages from all channels
            print(f"\nExtracting messages from all {len(channel_ids)} channels...")
            csv_file = extract_server_messages(
                server_id=server_id,
                user_token=user_token,
                channel_ids=channel_ids,
                max_messages=args.max_messages,
                output_dir=args.output_dir
            )
            
            print(f"\nAll messages extracted and saved to {csv_file}")
        
        # Visualize data
        if args.mode == 'visualize' or args.mode == 'both':
            if not csv_file:
                raise ValueError("CSV file path required for visualization. Please run with '--mode extract' first or specify a CSV file with '--csv'.")
            
            print(f"\nCreating server-wide visualization from {csv_file}...")
            
            # No channel filter - we want the entire server
            create_animation(
                csv_file=csv_file,
                time_grouping=args.time_grouping,
                theme_name=args.theme,
                max_users=args.max_users,
                output_file=output_file,
                fps=args.fps,
                dpi=args.dpi,
                server_name=server_name,
                server_id=server_id,
                channel_filter=None  # No channel filter - use all data
            )
            
            print(f"\nVisualization complete and saved to {output_file}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the script
if __name__ == "__main__":
    main()
