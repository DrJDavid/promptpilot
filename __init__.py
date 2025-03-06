"""
PromptPilot - A tool for generating optimized prompts based on code repositories.

This package provides tools for code repository analysis, AST generation,
and enhanced prompt generation using embeddings and semantic analysis.
"""

__version__ = '0.1.0'
__author__ = 'PromptPilot Team'

# Import key functionality to make it available at the package level
from core.ingest import ingest_repository
from core.analyze import RepositoryAnalyzer
from core.ast_analyzer import ASTAnalyzer
from core.enhanced_prompt_generator import PromptGenerator
from core.enhanced_db import get_db

# Convenience function for CLI usage
def get_version():
    """Return the current version of the package."""
    return __version__
