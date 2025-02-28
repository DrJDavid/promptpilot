#!/usr/bin/env python3
"""
PromptPilot CLI - A tool for generating optimized prompts for code generation tasks.

This module provides the main command-line interface for PromptPilot.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import click
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("promptpilot")

# Ensure necessary paths exist
USER_HOME = os.path.expanduser("~")
PROMPTPILOT_DIR = os.path.join(USER_HOME, ".promptpilot")
REPOSITORIES_FILE = os.path.join(PROMPTPILOT_DIR, "repositories.json")

# Create directory if it doesn't exist
os.makedirs(PROMPTPILOT_DIR, exist_ok=True)

# Ensure repositories file exists
if not os.path.exists(REPOSITORIES_FILE):
    with open(REPOSITORIES_FILE, 'w') as f:
        json.dump([], f)


def get_repositories() -> List[Dict[str, Any]]:
    """Get list of processed repositories."""
    try:
        with open(REPOSITORIES_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error reading repositories file. Resetting file.")
        with open(REPOSITORIES_FILE, 'w') as f:
            json.dump([], f)
        return []


def save_repositories(repositories: List[Dict[str, Any]]) -> None:
    """Save list of processed repositories."""
    with open(REPOSITORIES_FILE, 'w') as f:
        json.dump(repositories, f, indent=2)


def add_repository(repo_path: str, output_dir: str) -> None:
    """Add repository to the list of processed repositories."""
    repositories = get_repositories()
    
    # Check if repository already exists
    for repo in repositories:
        if repo['path'] == repo_path:
            # Update existing repository
            repo['output_dir'] = output_dir
            save_repositories(repositories)
            return
    
    # Add new repository
    repositories.append({
        'path': repo_path,
        'name': os.path.basename(repo_path),
        'output_dir': output_dir
    })
    
    save_repositories(repositories)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """PromptPilot - Generate optimized prompts for code generation tasks."""
    pass


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option('--output-dir', '-o', type=click.Path(file_okay=False, dir_okay=True),
              help='Directory to store processed data (defaults to .promptpilot in the repository)')
@click.option('--skip-ast', is_flag=True, help='Skip AST analysis')
def process(repo_path: str, output_dir: Optional[str] = None, skip_ast: bool = False):
    """Process a repository to extract code information."""
    try:
        from core.ingest import RepositoryIngestor
        
        click.echo(f"Processing repository: {repo_path}")
        
        # Create ingestor and process repository
        ingestor = RepositoryIngestor(repo_path, output_dir)
        repo_data = ingestor.process_repository()
        
        # Register repository
        add_repository(repo_path, ingestor.output_dir)
        
        # Perform analysis
        if not skip_ast:
            click.echo("Performing code structure analysis...")
            try:
                from core.analyze import RepositoryAnalyzer
                from core.ast_analyzer import ASTAnalyzer
                
                # Analyze repository
                analyzer = RepositoryAnalyzer(ingestor.output_dir)
                analyzer.analyze_repository()
                
                # Perform AST analysis
                ast_analyzer = ASTAnalyzer(ingestor.output_dir)
                ast_analyzer.analyze_repository()
                
                click.echo("Analysis complete.")
            except ImportError as e:
                click.echo(f"Warning: Couldn't import analysis modules: {str(e)}")
                click.echo("Skipping code structure analysis.")
        else:
            click.echo("Skipping code structure analysis (--skip-ast flag used)")
        
        # Show summary
        click.echo("\nRepository processing complete!")
        click.echo(f"Processed {repo_data['file_count']} files")
        click.echo(f"Total size: {repo_data['total_size_bytes'] / 1024:.1f}KB")
        click.echo(f"Output directory: {ingestor.output_dir}")
        
    except ImportError as e:
        click.echo(f"Error: Failed to import required modules: {str(e)}")
        click.echo("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error processing repository: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument('task', type=str)
@click.option('--repo', '-r', type=str, help='Repository name to use for context')
@click.option('--output', '-o', type=click.Path(dir_okay=False), help='Output file for the prompt')
@click.option('--model', '-m', type=str, default='gemini-pro-latest', 
              help='Model to use for prompt generation (default: gemini-pro-latest)')
def prompt(task: str, repo: Optional[str] = None, output: Optional[str] = None,
           model: str = 'gemini-pro-latest'):
    """Generate an optimized prompt based on a task."""
    try:
        if not repo:
            # If no repo specified, list repositories
            repositories = get_repositories()
            if not repositories:
                click.echo("No repositories processed. Please process a repository first.")
                sys.exit(1)
            
            click.echo("Available repositories:")
            for i, repo_info in enumerate(repositories):
                click.echo(f"{i+1}. {repo_info['name']} ({repo_info['path']})")
            
            # Prompt user to select a repository
            repo_idx = click.prompt("Select a repository (number)", type=int) - 1
            if repo_idx < 0 or repo_idx >= len(repositories):
                click.echo("Invalid selection.")
                sys.exit(1)
            
            selected_repo = repositories[repo_idx]
            repo_output_dir = selected_repo['output_dir']
        else:
            # Find repository by name
            repositories = get_repositories()
            matched_repos = [r for r in repositories if r['name'] == repo or r['path'] == repo]
            
            if not matched_repos:
                click.echo(f"Repository '{repo}' not found. Use 'promptpilot list' to see available repositories.")
                sys.exit(1)
            
            selected_repo = matched_repos[0]
            repo_output_dir = selected_repo['output_dir']
        
        click.echo(f"Generating prompt for task: {task}")
        click.echo(f"Using repository: {selected_repo['name']}")
        
        # Generate prompt
        try:
            from core.enhanced_prompt_generator import PromptGenerator
            
            generator = PromptGenerator(repo_output_dir, model=model)
            prompt = generator.generate_prompt(task)
            
            if output:
                with open(output, 'w') as f:
                    f.write(prompt)
                click.echo(f"Prompt saved to: {output}")
            else:
                click.echo("\n" + "=" * 80)
                click.echo("GENERATED PROMPT:")
                click.echo("=" * 80)
                click.echo(prompt)
                click.echo("=" * 80)
            
        except ImportError as e:
            click.echo(f"Error: Failed to import prompt generator: {str(e)}")
            click.echo("Make sure all dependencies are installed: pip install -r requirements.txt")
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"Error generating prompt: {str(e)}")
        sys.exit(1)


@cli.command(name="list")
def list_repositories():
    """List processed repositories."""
    repositories = get_repositories()
    
    if not repositories:
        click.echo("No repositories processed. Use 'promptpilot process' to process a repository.")
        return
    
    click.echo("Processed repositories:")
    for i, repo in enumerate(repositories):
        click.echo(f"{i+1}. {repo['name']}")
        click.echo(f"   Path: {repo['path']}")
        click.echo(f"   Output directory: {repo['output_dir']}")
        
        # Check if analysis data exists
        metadata_path = os.path.join(repo['output_dir'], 'repository_metadata.json')
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                click.echo(f"   Files: {metadata['file_count']}")
                click.echo(f"   Size: {metadata['total_size_bytes'] / 1024:.1f}KB")
                
                # Show file types
                file_types = metadata['file_types']
                if file_types:
                    types_str = ", ".join([f"{ext}: {count}" for ext, count in list(file_types.items())[:5]])
                    click.echo(f"   Types: {types_str}...")
            except (json.JSONDecodeError, FileNotFoundError, KeyError):
                click.echo("   Metadata: Not available")
        else:
            click.echo("   Metadata: Not available")
        
        click.echo("")


if __name__ == "__main__":
    cli()
