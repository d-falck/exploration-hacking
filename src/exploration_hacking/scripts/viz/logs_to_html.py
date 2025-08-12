#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from html import escape


def jsonl_to_html(jsonl_path: Path) -> str:
    """Convert a JSONL file to an HTML table."""
    
    with open(jsonl_path, 'r') as f:
        lines = [json.loads(line) for line in f]
    
    if not lines:
        return "<p>No data found</p>"
    
    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{jsonl_path.stem} - Log Viewer</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 20px;
                background: #f5f5f5;
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                background: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            th {{
                background: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
                position: sticky;
                top: 0;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
                vertical-align: top;
            }}
            tr:hover {{
                background: #f5f5f5;
            }}
            .prompt, .completion {{
                max-width: 500px;
                max-height: 200px;
                overflow: auto;
                white-space: pre-wrap;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 12px;
                background: #f9f9f9;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
            }}
            .reward {{
                font-weight: bold;
                text-align: center;
            }}
            .reward.positive {{
                color: #4CAF50;
            }}
            .reward.negative {{
                color: #f44336;
            }}
            .task-id {{
                font-family: monospace;
                background: #e3f2fd;
                padding: 2px 6px;
                border-radius: 3px;
            }}
            .step {{
                text-align: center;
                font-weight: bold;
            }}
            .stats {{
                margin: 20px 0;
                padding: 15px;
                background: white;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            small[title] {{
                cursor: help;
                text-decoration: underline dotted;
            }}
            small[title]:hover {{
                background: #fffbdd;
            }}
        </style>
    </head>
    <body>
        <h1>{jsonl_path.stem} Completion Logs</h1>
    """
    
    # Calculate statistics
    rewards = [entry.get('reward', 0) for entry in lines]
    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)
        
        html += f"""
        <div class="stats">
            <h3>Statistics</h3>
            <p><strong>Total Entries:</strong> {len(lines)}</p>
            <p><strong>Mean Reward:</strong> {mean_reward:.4f}</p>
            <p><strong>Min Reward:</strong> {min_reward:.4f}</p>
            <p><strong>Max Reward:</strong> {max_reward:.4f}</p>
        </div>
        """
    
    html += """
        <table>
            <thead>
                <tr>
                    <th>Step</th>
                    <th>Task ID</th>
                    <th>Prompt</th>
                    <th>Completion</th>
                    <th>Reward</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for entry in lines:
        step = entry.get('step', '')
        task_id = entry.get('task_id', '')
        
        # Format prompt (handle both list and string formats)
        prompt = entry.get('prompt', '')
        if isinstance(prompt, list):
            prompt_text = '\n'.join([f"[{msg['role']}]: {msg['content']}" for msg in prompt])
        else:
            prompt_text = str(prompt)
        
        # Format completion
        completion = entry.get('completion', '')
        if isinstance(completion, list) and completion:
            completion_text = completion[0].get('content', '') if isinstance(completion[0], dict) else str(completion[0])
        else:
            completion_text = str(completion)
        
        reward = entry.get('reward', 0)
        reward_class = 'positive' if reward >= 0 else 'negative'
        
        # Check for auxiliary info
        aux_info = entry.get('aux_info', {})
        aux_html = ""
        if aux_info:
            if 'segment' in aux_info:
                aux_html += f"<br><small>Segment: {aux_info['segment']}</small>"
            if 'accuracy' in aux_info:
                aux_html += f"<br><small>Accuracy: {aux_info['accuracy']:.2f}</small>"
            if 'inverted' in aux_info and aux_info['inverted']:
                aux_html += f"<br><small>Inverted</small>"
            if 'raw_score' in aux_info:
                aux_html += f"<br><small>Raw: {aux_info['raw_score']:.2f}</small>"
            if 'criterion' in aux_info:
                aux_html += f"<br><small>Criterion: {escape(aux_info['criterion'][:50])}</small>"
            if 'reasoning' in aux_info:
                # Show first 200 chars of reasoning with tooltip for full
                reasoning_preview = escape(aux_info['reasoning'][:200])
                full_reasoning = escape(aux_info['reasoning']).replace('\n', '&#10;')
                aux_html += f'<br><small title="{full_reasoning}">Reasoning: {reasoning_preview}...</small>'
            if 'error' in aux_info:
                aux_html += f"<br><small style='color:red'>Error: {escape(aux_info['error'])}</small>"
        
        html += f"""
                <tr>
                    <td class="step">{step}</td>
                    <td><span class="task-id">{escape(task_id)}</span></td>
                    <td><div class="prompt">{escape(prompt_text)}</div></td>
                    <td><div class="completion">{escape(completion_text)}</div></td>
                    <td class="reward {reward_class}">{reward:.4f}{aux_html}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    return html


def main():
    parser = argparse.ArgumentParser(description='Convert JSONL log files to HTML')
    parser.add_argument('log_dir', type=Path, help='Directory containing completion_logs')
    args = parser.parse_args()
    
    log_dir = args.log_dir / 'completion_logs' if (args.log_dir / 'completion_logs').exists() else args.log_dir
    
    if not log_dir.exists():
        print(f"Error: Directory {log_dir} does not exist")
        return
    
    jsonl_files = list(log_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {log_dir}")
        return
    
    output_dir = log_dir / 'html'
    output_dir.mkdir(exist_ok=True)
    
    for jsonl_file in jsonl_files:
        print(f"Converting {jsonl_file.name}...")
        html_content = jsonl_to_html(jsonl_file)
        
        output_file = output_dir / f"{jsonl_file.stem}.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"  → Saved to {output_file}")
    
    # Create index file
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Log Files Index</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 40px;
                background: #f5f5f5;
            }
            h1 {
                color: #333;
            }
            .file-list {
                list-style: none;
                padding: 0;
            }
            .file-list li {
                margin: 10px 0;
            }
            .file-list a {
                display: inline-block;
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                transition: background 0.3s;
            }
            .file-list a:hover {
                background: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>Completion Log Files</h1>
        <ul class="file-list">
    """
    
    for jsonl_file in sorted(jsonl_files):
        html_file = f"{jsonl_file.stem}.html"
        index_html += f'            <li><a href="{html_file}">{jsonl_file.stem}</a></li>\n'
    
    index_html += """
        </ul>
    </body>
    </html>
    """
    
    index_file = output_dir / 'index.html'
    with open(index_file, 'w') as f:
        f.write(index_html)
    
    print(f"\n✓ Created index at {index_file}")
    print(f"✓ Converted {len(jsonl_files)} files")
    print(f"\nOpen {index_file} in your browser to view the logs")


if __name__ == '__main__':
    main()