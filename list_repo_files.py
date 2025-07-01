#!/usr/bin/env python3
"""
Repository File Listing Tool with GitHub Raw URLs
Lists all files and folders in the catalyst-trading-system repository with direct GitHub access URLs

This script generates both:
- Raw URLs (https://raw.githubusercontent.com/...) - for Claude/API access
- Web URLs (https://github.com/...) - for human viewing in browser
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# GitHub repository information
GITHUB_USERNAME = "TradingApplication"  # GitHub organization/username
REPO_NAME = "catalyst-trading-system"  # Repository name
BRANCH = "main"  # Branch name

def path_to_github_url(filepath):
    """Convert local file path to GitHub raw URL"""
    # Normalize path and remove leading './'
    normalized_path = filepath.replace('\\', '/').lstrip('./')
    
    # Create GitHub raw URL (corrected format)
    github_url = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/{normalized_path}"
    return github_url

def path_to_github_web_url(filepath):
    """Convert local file path to regular GitHub web URL"""
    # Normalize path and remove leading './'
    normalized_path = filepath.replace('\\', '/').lstrip('./')
    
    # Create regular GitHub URL
    github_url = f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}/blob/{BRANCH}/{normalized_path}"
    return github_url

def list_repository_contents():
    """List all files and folders in the repository with GitHub URLs"""
    
    # Get the current directory
    root_dir = Path.cwd()
    print(f"Repository Index Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Repository Root: {root_dir}")
    print(f"GitHub Repository: {GITHUB_USERNAME}/{REPO_NAME}")
    print(f"Branch: {BRANCH}")
    print("=" * 80)
    
    # Directories to skip
    skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env', '.env', 
                 'node_modules', '.idea', '.vscode', '*.egg-info'}
    
    # File extensions to group
    file_groups = {
        'Python': ['.py'],
        'Docker': ['Dockerfile', '.dockerfile'],
        'Config': ['.yml', '.yaml', '.json', '.toml', '.ini', '.conf'],
        'Documentation': ['.md', '.txt', '.rst'],
        'Scripts': ['.sh', '.bash'],
        'Data': ['.csv', '.sql', '.db'],
        'Web': ['.html', '.css', '.js'],
        'Other': []
    }
    
    # Collect all files
    all_files = []
    all_dirs = []
    files_with_urls = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden and cache directories
        dirnames[:] = [d for d in dirnames if not any(
            d.startswith('.') and d != '.github' or 
            d == skip_dir or 
            d.endswith(skip_dir) 
            for skip_dir in skip_dirs
        )]
        
        # Convert to relative path
        rel_dirpath = Path(dirpath).relative_to(root_dir)
        
        # Add directory if not root
        if str(rel_dirpath) != '.':
            all_dirs.append(str(rel_dirpath))
        
        # Add files
        for filename in filenames:
            if not filename.startswith('.') or filename in ['.env.example', '.gitignore']:
                file_path = rel_dirpath / filename if str(rel_dirpath) != '.' else Path(filename)
                file_path_str = str(file_path)
                all_files.append(file_path_str)
                files_with_urls.append({
                    'path': file_path_str,
                    'url': path_to_github_url(file_path_str),
                    'web_url': path_to_github_web_url(file_path_str)
                })
    
    # Sort directories and files
    all_dirs.sort()
    all_files.sort()
    files_with_urls.sort(key=lambda x: x['path'])
    
    # Print directory structure
    print("\n📁 DIRECTORY STRUCTURE:")
    print("-" * 40)
    if all_dirs:
        for dir_name in all_dirs:
            level = dir_name.count(os.sep)
            indent = "  " * level
            folder_name = os.path.basename(dir_name)
            print(f"{indent}📁 {folder_name}/")
    else:
        print("No subdirectories found")
    
    # Group files by type
    grouped_files = {group: [] for group in file_groups}
    grouped_files_with_urls = {group: [] for group in file_groups}
    
    for file_info in files_with_urls:
        file_path = file_info['path']
        categorized = False
        for group, extensions in file_groups.items():
            if group == 'Docker' and 'Dockerfile' in file_path:
                grouped_files[group].append(file_path)
                grouped_files_with_urls[group].append(file_info)
                categorized = True
                break
            elif any(file_path.endswith(ext) for ext in extensions):
                grouped_files[group].append(file_path)
                grouped_files_with_urls[group].append(file_info)
                categorized = True
                break
        
        if not categorized:
            grouped_files['Other'].append(file_path)
            grouped_files_with_urls['Other'].append(file_info)
    
    # Print files by category
    print("\n📄 FILES BY CATEGORY:")
    print("-" * 40)
    
    for group, files in grouped_files.items():
        if files:
            print(f"\n{group} Files ({len(files)}):")
            for file_path in sorted(files):
                print(f"  - {file_path}")
    
    # Print summary
    print("\n📊 SUMMARY:")
    print("-" * 40)
    print(f"Total Directories: {len(all_dirs)}")
    print(f"Total Files: {len(all_files)}")
    
    # Check for key files mentioned in the implementation document
    print("\n🔍 CHECKING FOR REQUIRED FILES:")
    print("-" * 40)
    
    required_files = [
        'docker-compose.yml',
        '.env.example',
        'requirements.txt',
        'nginx.conf',
        'prometheus.yml',
        'init_database.sql',
        'Dockerfile.coordination',
        'Dockerfile.news',
        'Dockerfile.scanner',
        'Dockerfile.patterns',
        'Dockerfile.technical',
        'Dockerfile.trading',
        'Dockerfile.reporting',
        'Dockerfile.dashboard',
        'README.md'
    ]
    
    for req_file in required_files:
        exists = any(req_file in file_path for file_path in all_files)
        status = "✅" if exists else "❌"
        print(f"{status} {req_file}")
    
    # Create a simple text listing for easy copying
    print("\n📋 SIMPLE FILE LISTING (for copying):")
    print("-" * 40)
    for file_path in all_files:
        print(file_path)
    
    # NEW SECTION: Files with GitHub Raw URLs for direct access
    print("\n🔗 FILES WITH GITHUB RAW URLS (for Claude access):")
    print("-" * 40)
    print("NOTE: Use the Raw URLs when sharing with Claude for direct file access")
    print("\nKEY CONFIGURATION FILES:")
    
    # List important files first with their URLs
    important_files = [
        'docker-compose.yml',
        '.env.example',
        'requirements.txt',
        'nginx.conf',
        'prometheus.yml',
        'init_database.sql'
    ]
    
    for important_file in important_files:
        for file_info in files_with_urls:
            if important_file in file_info['path']:
                print(f"\n{file_info['path']}:")
                print(f"  Raw URL: {file_info['url']}")
                print(f"  Web URL: {file_info['web_url']}")
    
    print("\nDOCKERFILES:")
    for file_info in files_with_urls:
        if 'Dockerfile' in file_info['path']:
            print(f"\n{file_info['path']}:")
            print(f"  Raw URL: {file_info['url']}")
            print(f"  Web URL: {file_info['web_url']}")
    
    print("\nSERVICE FILES:")
    for file_info in files_with_urls:
        if 'service' in file_info['path'].lower() and file_info['path'].endswith('.py'):
            print(f"\n{file_info['path']}:")
            print(f"  Raw URL: {file_info['url']}")
            print(f"  Web URL: {file_info['web_url']}")
    
    # Save URLs to a separate file for easy access
    with open('project_urls.txt', 'w') as f:
        f.write(f"GitHub Repository URLs for Direct Access\n")
        f.write(f"Repository: {GITHUB_USERNAME}/{REPO_NAME}\n")
        f.write(f"Branch: {BRANCH}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("RAW URLS (for Claude/API access):\n")
        f.write("-" * 40 + "\n")
        for file_info in files_with_urls:
            f.write(f"{file_info['path']}\n{file_info['url']}\n\n")
    
    print("\n✅ URL list saved to: project_urls.txt")
    print("\nExample URLs generated:")
    print(f"  Raw (for Claude): https://raw.githubusercontent.com/{GITHUB_USERNAME}/{REPO_NAME}/{BRANCH}/docker-compose.yml")
    print(f"  Web (for viewing): https://github.com/{GITHUB_USERNAME}/{REPO_NAME}/blob/{BRANCH}/docker-compose.yml")
    
    # Create markdown file with clickable links
    with open('project_urls.md', 'w') as f:
        f.write(f"# GitHub Repository File Index\n\n")
        f.write(f"**Repository:** [{GITHUB_USERNAME}/{REPO_NAME}](https://github.com/{GITHUB_USERNAME}/{REPO_NAME})\n")
        f.write(f"**Branch:** {BRANCH}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Key files section
        f.write("## Key Configuration Files\n\n")
        for important_file in important_files:
            for file_info in files_with_urls:
                if important_file in file_info['path']:
                    f.write(f"- [{file_info['path']}]({file_info['web_url']}) ([raw]({file_info['url']}))\n")
        
        # Dockerfiles section
        f.write("\n## Dockerfiles\n\n")
        for file_info in files_with_urls:
            if 'Dockerfile' in file_info['path']:
                f.write(f"- [{file_info['path']}]({file_info['web_url']}) ([raw]({file_info['url']}))\n")
        
        # All files by category
        f.write("\n## All Files by Category\n\n")
        for group, file_infos in grouped_files_with_urls.items():
            if file_infos:
                f.write(f"\n### {group}\n\n")
                for file_info in sorted(file_infos, key=lambda x: x['path']):
                    f.write(f"- [{file_info['path']}]({file_info['web_url']}) ([raw]({file_info['url']}))\n")
    
    print(f"✅ Markdown index saved to: project_urls.md")

if __name__ == "__main__":
    try:
        list_repository_contents()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)