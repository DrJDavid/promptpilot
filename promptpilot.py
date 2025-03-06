#!/usr/bin/env python3
"""
PromptPilot CLI - A tool for generating optimized prompts for code generation tasks.

This module provides the main command-line interface for PromptPilot.
"""

import os
import sys
import json
import logging
import asyncio
import time
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

# Utility functions
def format_size(size_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f}KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.1f}MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.1f}GB"

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


def add_repository(repo_path: str, output_dir: str, repo_id: Optional[int] = None) -> None:
    """Register a repository in the list of processed repositories."""
    repositories = get_repositories()
    
    # Check if repository already exists
    for repo in repositories:
        if repo['path'] == repo_path:
            repo['output_dir'] = output_dir
            # Update with repo_id if provided
            if repo_id is not None:
                repo['id'] = repo_id
                logger.info(f"Updated repository {os.path.basename(repo_path)} with ID {repo_id}")
            save_repositories(repositories)
            
            # Update in database if available
            try:
                from core.enhanced_db import get_db
                db = get_db()
                if db and db.is_available():
                    # Get repository data from output directory
                    repo_data_path = os.path.join(output_dir, 'repository_data.json')
                    if os.path.exists(repo_data_path):
                        with open(repo_data_path, 'r') as f:
                            repo_data = json.load(f)
                            repo_data['name'] = os.path.basename(repo_path)
                            repo_data['path'] = repo_path
                            db.store_repository(repo_data)
                            logger.info(f"Repository {repo_data['name']} updated in database")
            except ImportError:
                logger.warning("Enhanced database module not available. Repository only saved locally.")
            except Exception as e:
                logger.warning(f"Error updating repository in database: {str(e)}")
            
            return
    
    # Add new repository
    repo_name = os.path.basename(repo_path)
    new_repo = {
        'path': repo_path,
        'name': repo_name,
        'output_dir': output_dir
    }
    
    # Add ID if provided
    if repo_id is not None:
        new_repo['id'] = repo_id
    
    repositories.append(new_repo)
    
    save_repositories(repositories)
    
    # Also add to database if available
    try:
        from core.enhanced_db import get_db
        db = get_db()
        if db and db.is_available():
            # Get repository data from output directory
            repo_data_path = os.path.join(output_dir, 'repository_data.json')
            if os.path.exists(repo_data_path):
                with open(repo_data_path, 'r') as f:
                    repo_data = json.load(f)
                    repo_data['name'] = repo_name
                    repo_data['path'] = repo_path
                    stored_repo_id = db.store_repository(repo_data)
                    
                    # Update the ID in our local record
                    if stored_repo_id:
                        for repo in repositories:
                            if repo['path'] == repo_path:
                                repo['id'] = stored_repo_id
                                logger.info(f"Repository {repo_name} ID set to {stored_repo_id}")
                                save_repositories(repositories)
                                break
                    
                    logger.info(f"Repository {repo_name} added to database")
    except ImportError:
        logger.warning("Enhanced database module not available. Repository only saved locally.")
    except Exception as e:
        logger.warning(f"Error adding repository to database: {str(e)}")


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """PromptPilot - Generate optimized prompts for code generation tasks."""
    # Check for necessary credentials
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_SERVICE_ROLE_KEY"):
        logger.warning("Supabase credentials not found in environment variables.")
        logger.warning("GraphQL functionality and database storage will be disabled.")
        logger.warning("Please set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in your .env file.")
    
    # Check for Gemini API key
    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not found in environment variables.")
        logger.warning("Prompt generation with Gemini will be disabled.")
        logger.warning("Please set GEMINI_API_KEY in your .env file.")
        
    pass


@cli.command()
@click.argument('repo_path', type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
@click.option('--output-dir', '-o', type=click.Path(file_okay=False, dir_okay=True),
              help='Directory to store processed data (defaults to .promptpilot in the repository)')
@click.option('--skip-ast', is_flag=True, help='Skip AST analysis')
@click.option('--skip-analysis', is_flag=True, help='Skip embedding analysis')
def process(repo_path: str, output_dir: Optional[str] = None, skip_ast: bool = False, skip_analysis: bool = False):
    """Process a repository to extract code information."""
    try:
        from core.ingest import RepositoryIngestor
        
        click.echo(f"Processing repository: {repo_path}")
        
        # Create ingestor and process repository
        ingestor = RepositoryIngestor(repo_path, output_dir)
        repo_data = ingestor.process(apply_ast_analysis=not skip_ast)
        
        # Register repository
        try:
            # Get repository ID if available
            repo_id = None
            repo_meta_path = os.path.join(ingestor.output_dir, 'repository_metadata.json')
            if os.path.exists(repo_meta_path):
                try:
                    with open(repo_meta_path, 'r') as f:
                        repo_meta = json.load(f)
                        repo_id = repo_meta.get('id')
                        if repo_id:
                            logger.info(f"Found repository ID: {repo_id}")
                except Exception as meta_e:
                    logger.warning(f"Error reading repository metadata: {str(meta_e)}")
            
            # Add repository with ID
            add_repository(repo_path, ingestor.output_dir, repo_id)
            click.echo(f"Repository registered: {os.path.basename(repo_path)}")
        except Exception as e:
            click.echo(f"Warning: Failed to register repository: {str(e)}")
            click.echo("Continuing with processing...")
        
        # Perform analysis
        if not skip_analysis:
            click.echo("Performing embedding generation and analysis...")
            try:
                from core.analyze import RepositoryAnalyzer
                
                # Analyze repository
                analyzer = RepositoryAnalyzer(ingestor.output_dir)
                
                try:
                    analyzer.analyze_repository()
                    click.echo("Embedding analysis complete.")
                except PermissionError as pe:
                    click.echo(f"Warning: Permission error during analysis: {str(pe)}")
                    click.echo("This may be due to files being locked by another process.")
                    click.echo("The repository was still processed, but analysis was incomplete.")
                except Exception as ae:
                    click.echo(f"Warning: Error during analysis: {str(ae)}")
                    click.echo("The repository was still processed, but analysis was incomplete.")
                
            except ImportError as e:
                click.echo(f"Warning: Couldn't import analysis modules: {str(e)}")
                click.echo("Skipping embedding analysis.")
        else:
            click.echo("Skipping embedding analysis (--skip-analysis flag used)")
        
        # AST analysis is now handled in the process method
        if skip_ast:
            click.echo("Skipping AST analysis (--skip-ast flag used)")
        
        # Generate repository digest using gitingest
        try:
            from core.enhanced_db import get_db
            db = get_db()
            if db and db.is_available():
                # Get repository ID
                repo_meta_path = os.path.join(ingestor.output_dir, 'repository_metadata.json')
                if os.path.exists(repo_meta_path):
                    with open(repo_meta_path, 'r') as f:
                        repo_meta = json.load(f)
                        repo_id = repo_meta.get('id')
                    
                    if repo_id:
                        click.echo("Generating repository digest using gitingest...")
                        digest_path = db.generate_repository_digest_with_gitingest(repo_id, repo_path)
                        if digest_path:
                            click.echo(f"Repository digest generated and stored. Digest file: {digest_path}")
                        else:
                            click.echo("Warning: Failed to generate repository digest")
        except ImportError:
            click.echo("Warning: Enhanced database module not available. Repository digest not generated.")
        except Exception as e:
            click.echo(f"Warning: Error generating repository digest: {str(e)}")
        
        # Report success
        click.echo(f"\nRepository processing complete!")
        click.echo(f"Processed {repo_data.get('file_count', 0)} files")
        click.echo(f"Total size: {format_size(repo_data.get('total_size_bytes', 0))}")
        click.echo(f"Output directory: {ingestor.output_dir}")
        
    except Exception as e:
        click.echo(f"Error processing repository: {str(e)}")
        raise


@cli.command()
@click.argument('task', type=str)
@click.option('--repo', '-r', type=str, help='Repository name to use for context')
@click.option('--output', '-o', type=click.Path(dir_okay=False), help='Output file for the prompt')
@click.option('--model', '-m', type=str, default='gemini-2.0-flash-001', 
              help='Model to use for prompt generation (default: gemini-2.0-flash-001)')
def prompt(task: str, repo: Optional[str] = None, output: Optional[str] = None,
           model: str = 'gemini-2.0-flash-001'):
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
            from core.fixed_prompt_generator import PromptGenerator
            
            # Create async event loop and run prompt generation
            async def generate():
                generator = PromptGenerator(repo_output_dir, model=model)
                return await generator.generate_prompt(task)
                
            # Run the async function in the event loop
            prompt = asyncio.run(generate())
            
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


@cli.command()
@click.option('--repo', help='The repository name to chat about')
@click.option('--model', default='gemini-2.0-flash-001', help='The model to use for responses')
@click.option('--temperature', default=0.7, type=float, help='Temperature setting for the model (0.0-1.0)')
def chat(repo, model, temperature):
    """Start an interactive chat session about a repository."""
    # Set logging level to DEBUG to see more information
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('promptpilot.chat')
    logger.setLevel(logging.DEBUG)
    
    try:
        from core.chat_interface import RepositoryChat
        from utils.db_utils import RepositoryDB
        
        db = RepositoryDB()
        
        # If no repo specified, list available repositories
        if not repo:
            click.echo("Available repositories:")
            repositories = get_repositories()
            for i, r in enumerate(repositories, 1):
                click.echo(f"{i}. {r['name']} ({r['path']})")
                
            repo_num = click.prompt("Select a repository (number)", type=int)
            if repo_num < 1 or repo_num > len(repositories):
                click.echo("Invalid selection.")
                return
                
            selected_repo = repositories[repo_num - 1]
            repo_name = selected_repo['name']
            repo_path = selected_repo['path']
            repo_id = selected_repo.get('id')
        else:
            # Find the repository by name
            repositories = get_repositories()
            repo_found = False
            
            for r in repositories:
                if r['name'].lower() == repo.lower():
                    repo_name = r['name']
                    repo_path = r['path']
                    repo_id = r.get('id')
                    repo_found = True
                    break
                    
            if not repo_found:
                click.echo(f"Repository '{repo}' not found.")
                return
        
        # If we don't have the repo ID locally, try to look it up in Supabase
        if not repo_id:
            try:
                # Try by path first
                supabase_data = db.supabase.table('repositories').select('id').eq('path', repo_path).execute()
                if supabase_data and hasattr(supabase_data, 'data') and len(supabase_data.data) > 0:
                    repo_id = supabase_data.data[0]['id']
                    click.echo(f"Found repository ID in database by path: {repo_id}")
                    
                    # Update local record
                    repositories = get_repositories()
                    for r in repositories:
                        if r['path'] == repo_path:
                            r['id'] = repo_id
                            save_repositories(repositories)
                            break
                else:
                    # Try by name as fallback
                    supabase_data = db.supabase.table('repositories').select('id').eq('name', repo_name).execute()
                    if supabase_data and hasattr(supabase_data, 'data') and len(supabase_data.data) > 0:
                        repo_id = supabase_data.data[0]['id']
                        click.echo(f"Found repository ID in database by name: {repo_id}")
                        
                        # Update local record
                        repositories = get_repositories()
                        for r in repositories:
                            if r['name'] == repo_name and r['path'] == repo_path:
                                r['id'] = repo_id
                                save_repositories(repositories)
                                break
            except Exception as lookup_error:
                click.echo(f"Error looking up repository ID: {str(lookup_error)}")
        
        if not repo_id:
            click.echo(f"Could not find repository ID for {repo_name}.")
            click.echo("Please run 'promptpilot process' on this repository first.")
            sys.exit(1)
            
        click.echo(f"Starting chat session for repository: {repo_name} (ID: {repo_id})")
        click.echo("Type '/help' for available commands, or '/exit' to quit.")
        click.echo("You can also use 'promptpilot c' as a shortcut for this chat command.")
        
        # Initialize chat interface
        chat_interface = RepositoryChat(repo_id, model_name=model, temperature=temperature)
        # Store repo_name as an attribute for later use
        chat_interface.repo_name = repo_name
        
        # Start chat loop
        try:
            while True:
                # Get user input
                user_input = click.prompt("\nYou", prompt_suffix="> ")
                
                # Handle special commands
                if user_input.lower() == '/exit':
                    click.echo("Ending chat session.")
                    break
                    
                if user_input.lower() == '/help':
                    click.echo("\nAvailable commands:")
                    click.echo("  /exit           - Exit the chat session")
                    click.echo("  /help           - Show this help message")
                    click.echo("  /prompt <task>  - Generate a prompt for a specific task")
                    click.echo("  /clear          - Clear the conversation history")
                    click.echo("  /search <query> - Search for specific code in the repository")
                    click.echo("\nYou can also request file contents directly:")
                    click.echo("  - 'show me readme.md'        - Shows the content of README.md")
                    click.echo("  - 'display the file utils.py' - Shows the content of utils.py")
                    click.echo("  - 'what's in config.json'    - Shows the content of config.json")
                    continue
                    
                if user_input.lower().startswith('/prompt '):
                    task = user_input[8:].strip()
                    if task:
                        click.echo("\nGenerating prompt for task: " + task)
                        prompt = chat_interface.generate_prompt(task)
                        click.echo("\nGenerated Prompt:\n")
                        click.echo(prompt)
                    else:
                        click.echo("Please specify a task after /prompt")
                    continue
                    
                if user_input.lower() == '/clear':
                    chat_interface.clear_history()
                    click.echo("Conversation history cleared.")
                    continue
                    
                if user_input.lower().startswith('/search '):
                    query = user_input[8:].strip()
                    if query:
                        click.echo(f"\nSearching for: {query}")
                        results = chat_interface.search_code(query)
                        if results:
                            click.echo("\nFound relevant code:")
                            for i, result in enumerate(results):
                                click.echo(f"\n{i+1}. {result['path']}")
                                if 'snippet' in result:
                                    click.echo(f"   {result['snippet']}")
                        else:
                            click.echo("No relevant code found.")
                    else:
                        click.echo("Please specify a search query after /search")
                    continue
                
                # Generate response
                try:
                    with click.progressbar(length=100, label="Thinking") as bar:
                        # Update progress bar in chunks to show activity
                        for i in range(10):
                            time.sleep(0.1)
                            bar.update(10)
                        
                        # Get response from chat interface
                        response = chat_interface.chat(user_input)
                        
                        # Complete the progress bar
                        bar.update(100 - bar.pos)
                    
                    click.echo("\nAI", nl=False)
                    click.echo("> " + response)
                except Exception as e:
                    click.echo(f"Error getting response: {str(e)}")
        except Exception as e:
            click.echo(f"Error during chat session: {str(e)}")
            import traceback
            click.echo(traceback.format_exc())
        
    except ImportError as e:
        click.echo(f"Error: Required modules not available: {str(e)}")
        click.echo("Make sure you have the required packages installed.")
    except Exception as e:
        click.echo(f"Error starting chat: {str(e)}")


# Add a shortcut command 'c' that aliases to the 'chat' command
@cli.command(name="c")
@click.option('--repo', '-r', type=str, help='Repository name to chat about')
@click.option('--model', '-m', type=str, default='gemini-2.0-flash-001',
              help='Model to use for chat (default: gemini-2.0-flash-001)')
@click.option('--temperature', '-t', type=float, default=0.7,
              help='Temperature for model generation (0.0-1.0)')
def chat_shortcut(repo: Optional[str] = None, model: str = 'gemini-2.0-flash-001', temperature: float = 0.7):
    """Shortcut for the chat command."""
    # Set logging level to DEBUG to see more information
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('promptpilot.chat')
    logger.setLevel(logging.DEBUG)
    
    try:
        from core.chat_interface import RepositoryChat
        
        # If no repo specified, list repositories
        if not repo:
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
            repo_name = selected_repo['name']
            repo_path = selected_repo['path']
            repo_id = selected_repo.get('id')
        else:
            # Find repository by name
            repo_data = None
            for r in get_repositories():
                if r['name'] == repo:
                    repo_data = r
                    break
                    
            if not repo_data:
                click.echo(f"Repository '{repo}' not found.")
                sys.exit(1)
                
            repo_name = repo_data['name']
            repo_path = repo_data['path']
            repo_id = repo_data.get('id')
        
        # If repo_id not in local record, look it up in the database
        if not repo_id:
            click.echo(f"Repository ID not found in local record for {repo_name}. Looking up in database...")
            # Query the database for the repository
            try:
                # Try by exact path first
                supabase_data = db.supabase.table('repositories').select('id').eq('path', repo_path).execute()
                
                # Check if we got a result
                if supabase_data and hasattr(supabase_data, 'data') and len(supabase_data.data) > 0:
                    repo_id = supabase_data.data[0]['id']
                    click.echo(f"Found repository ID in database: {repo_id}")
                    
                    # Update local record
                    repositories = get_repositories()
                    for r in repositories:
                        if r['path'] == repo_path:
                            r['id'] = repo_id
                            save_repositories(repositories)
                            break
                else:
                    # Try by name as fallback
                    supabase_data = db.supabase.table('repositories').select('id').eq('name', repo_name).execute()
                    if supabase_data and hasattr(supabase_data, 'data') and len(supabase_data.data) > 0:
                        repo_id = supabase_data.data[0]['id']
                        click.echo(f"Found repository ID in database by name: {repo_id}")
                        
                        # Update local record
                        repositories = get_repositories()
                        for r in repositories:
                            if r['name'] == repo_name and r['path'] == repo_path:
                                r['id'] = repo_id
                                save_repositories(repositories)
                                break
            except Exception as lookup_error:
                click.echo(f"Error looking up repository ID: {str(lookup_error)}")
        
        if not repo_id:
            click.echo(f"Could not find repository ID for {repo_name}.")
            click.echo("Please run 'promptpilot process' on this repository first.")
            sys.exit(1)
            
        click.echo(f"Starting chat session for repository: {repo_name} (ID: {repo_id})")
        click.echo("Type '/help' for available commands, or '/exit' to quit.")
        click.echo("You can also use 'promptpilot c' as a shortcut for this chat command.")
        
        # Initialize chat interface
        chat_interface = RepositoryChat(repo_id, model_name=model, temperature=temperature)
        # Store repo_name as an attribute for later use
        chat_interface.repo_name = repo_name
        
        # Start chat loop
        try:
            while True:
                # Get user input
                user_input = click.prompt("\nYou", prompt_suffix="> ")
                
                # Handle special commands
                if user_input.lower() == '/exit':
                    click.echo("Ending chat session.")
                    break
                    
                if user_input.lower() == '/help':
                    click.echo("\nAvailable commands:")
                    click.echo("  /exit           - Exit the chat session")
                    click.echo("  /help           - Show this help message")
                    click.echo("  /prompt <task>  - Generate a prompt for a specific task")
                    click.echo("  /clear          - Clear the conversation history")
                    click.echo("  /search <query> - Search for specific code in the repository")
                    click.echo("\nYou can also request file contents directly:")
                    click.echo("  - 'show me readme.md'        - Shows the content of README.md")
                    click.echo("  - 'display the file utils.py' - Shows the content of utils.py")
                    click.echo("  - 'what's in config.json'    - Shows the content of config.json")
                    continue
                    
                if user_input.lower().startswith('/prompt '):
                    task = user_input[8:].strip()
                    if task:
                        click.echo("\nGenerating prompt for task: " + task)
                        prompt = chat_interface.generate_prompt(task)
                        click.echo("\nGenerated Prompt:\n")
                        click.echo(prompt)
                    else:
                        click.echo("Please specify a task after /prompt")
                    continue
                    
                if user_input.lower() == '/clear':
                    chat_interface.clear_history()
                    click.echo("Conversation history cleared.")
                    continue
                    
                if user_input.lower().startswith('/search '):
                    query = user_input[8:].strip()
                    if query:
                        click.echo(f"\nSearching for: {query}")
                        results = chat_interface.search_code(query)
                        if results:
                            click.echo("\nFound relevant code:")
                            for i, result in enumerate(results):
                                click.echo(f"\n{i+1}. {result['path']}")
                                if 'snippet' in result:
                                    click.echo(f"   {result['snippet']}")
                        else:
                            click.echo("No relevant code found.")
                    else:
                        click.echo("Please specify a search query after /search")
                    continue
                
                # Generate response
                try:
                    with click.progressbar(length=100, label="Thinking") as bar:
                        # Update progress bar in chunks to show activity
                        for i in range(10):
                            time.sleep(0.1)
                            bar.update(10)
                        
                        # Get response from chat interface
                        response = chat_interface.chat(user_input)
                        
                        # Complete the progress bar
                        bar.update(100 - bar.pos)
                    
                    click.echo("\nAI", nl=False)
                    click.echo("> " + response)
                except Exception as e:
                    click.echo(f"Error getting response: {str(e)}")
        except Exception as e:
            click.echo(f"Error during chat session: {str(e)}")
            import traceback
            click.echo(traceback.format_exc())
        
    except ImportError as e:
        click.echo(f"Error: Required modules not available: {str(e)}")
        click.echo("Make sure you have the required packages installed.")
    except Exception as e:
        click.echo(f"Error starting chat: {str(e)}")


if __name__ == "__main__":
    cli()
