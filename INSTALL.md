# Installing PromptPilot

PromptPilot can be installed as a command-line tool using pip, allowing you to simply type `promptpilot` to run it instead of using Python commands.

## Installation Options

### Option 1: Install from the current directory (Development)

If you've cloned the repository and want to install it locally for development:

```bash
# Navigate to the promptpilot directory
cd /path/to/promptpilot

# Install in development mode
pip install -e .
```

This creates an "editable" installation, meaning changes to the code will be reflected immediately without needing to reinstall.

### Option 2: Install from the current directory (Standard)

For a standard installation from the local directory:

```bash
# Navigate to the promptpilot directory
cd /path/to/promptpilot

# Install the package
pip install .
```

## Verifying Installation

After installation, you should be able to run PromptPilot directly from the command line:

```bash
# View help information
promptpilot --help

# List available repositories
promptpilot list

# Start a chat with a repository
promptpilot chat --repo your-repository-name
```

You can also use the shortcut command for chat:

```bash
promptpilot c --repo your-repository-name
```

## Uninstalling

If you need to uninstall PromptPilot:

```bash
pip uninstall promptpilot
```

## Troubleshooting

If you encounter any issues with the installation:

1. Ensure you have Python 3.8 or newer installed
2. Check that you have the latest version of pip: `pip install --upgrade pip`
3. Verify that all dependencies in requirements.txt are installed correctly

For Windows users, you may need to add the Python Scripts directory to your PATH if the `promptpilot` command is not recognized. 