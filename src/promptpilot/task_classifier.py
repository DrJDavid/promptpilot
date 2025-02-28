# src/promptpilot/task_classifier.py
"""Module for classifying user tasks and mapping them to repository components."""

import re
from enum import Enum
from typing import Dict, List, Set, Tuple


class TaskCategory(str, Enum):
    """Enumeration of task categories for code-related operations."""

    CODE_GENERATION = "code_generation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    EXPLANATION = "explanation"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    ARCHITECTURE = "architecture"
    ANALYSIS = "analysis"
    OPTIMIZATION = "optimization"


TASK_KEYWORDS = {
    TaskCategory.CODE_GENERATION: {
        "create", "generate", "implement", "build", "write", "develop", "add", "make", 
        "new feature", "new function", "new class", "new component"
    },
    TaskCategory.DEBUGGING: {
        "debug", "fix", "bug", "issue", "error", "crash", "exception", "incorrect", 
        "not working", "fails", "problem", "investigate", "resolve", "troubleshoot"
    },
    TaskCategory.REFACTORING: {
        "refactor", "improve", "clean", "restructure", "simplify", "modernize", 
        "rewrite", "reorganize", "rename", "cleaner", "better", "optimize"
    },
    TaskCategory.EXPLANATION: {
        "explain", "understand", "describe", "clarify", "how does", "what is", 
        "analyze", "examine", "review", "walk through", "teach me", "learn"
    },
    TaskCategory.DOCUMENTATION: {
        "document", "comment", "write docs", "documentation", "readme", "api docs", 
        "examples", "guide", "tutorial", "instruction"
    },
    TaskCategory.TESTING: {
        "test", "unit test", "integration test", "e2e test", "mock", "stub", "assert", 
        "validate", "verify", "check", "coverage", "QA"
    },
    TaskCategory.ARCHITECTURE: {
        "design", "architect", "structure", "system", "pattern", "overview", "high-level", 
        "flow", "diagram", "layout", "organization", "component"
    },
    TaskCategory.ANALYSIS: {
        "analyze", "evaluate", "assess", "measure", "profile", "benchmark", 
        "perform analysis", "review", "check", "investigate"
    },
    TaskCategory.OPTIMIZATION: {
        "optimize", "performance", "faster", "efficient", "speed up", "reduce", 
        "improve performance", "bottleneck", "slow", "memory usage"
    }
}

# File extensions and their relevance to different tasks
FILE_RELEVANCE = {
    TaskCategory.CODE_GENERATION: {
        'primary': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.go', '.rs'],
        'secondary': ['.html', '.css', '.scss', '.less', '.php', '.rb']
    },
    TaskCategory.DEBUGGING: {
        'primary': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.go', '.rs'],
        'secondary': ['.log', '.test.js', '.test.py', '.spec.js']
    },
    TaskCategory.REFACTORING: {
        'primary': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.go', '.rs'],
        'secondary': ['.html', '.css', '.scss', '.less', '.php', '.rb']
    },
    TaskCategory.EXPLANATION: {
        'primary': ['.md', '.py', '.js', '.ts', '.jsx', '.tsx', '.java'],
        'secondary': ['.html', '.css', '.txt', '.rst', '.json', '.yaml', '.toml']
    },
    TaskCategory.DOCUMENTATION: {
        'primary': ['.md', '.rst', '.txt', '.docx'],
        'secondary': ['.py', '.js', '.java', '.html', '.css']
    },
    TaskCategory.TESTING: {
        'primary': ['.test.js', '.test.py', '.spec.js', '.spec.ts', '_test.go', 'Test.java', 'test_*.py'],
        'secondary': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.go', '.rs']
    },
    TaskCategory.ARCHITECTURE: {
        'primary': ['.md', '.py', '.js', '.ts', '.java', '.go', '.rs', '.c', '.cpp'],
        'secondary': ['.json', '.yaml', '.toml', '.xml', '.html', '.css']
    },
    TaskCategory.ANALYSIS: {
        'primary': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.go', '.rs'],
        'secondary': ['.json', '.yaml', '.toml', '.xml', '.log', '.csv']
    },
    TaskCategory.OPTIMIZATION: {
        'primary': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.go', '.rs'],
        'secondary': ['.json', '.yaml', '.toml', '.xml', '.html', '.css']
    }
}

# Important files for different tasks
IMPORTANT_FILES = {
    TaskCategory.CODE_GENERATION: [
        'package.json', 'setup.py', 'requirements.txt', 'Cargo.toml', 'pom.xml', 'build.gradle',
        'CMakeLists.txt', 'Makefile', 'README.md'
    ],
    TaskCategory.DEBUGGING: [
        'package.json', 'setup.py', 'requirements.txt', 'Cargo.toml', 'pom.xml', 'build.gradle',
        'CMakeLists.txt', 'Makefile', 'docker-compose.yml', 'Dockerfile', '.env.example'
    ],
    TaskCategory.REFACTORING: [
        'package.json', 'setup.py', 'requirements.txt', 'Cargo.toml', 'pom.xml', 'build.gradle',
        'CMakeLists.txt', 'Makefile', 'README.md', '.eslintrc', 'tsconfig.json', 'pyproject.toml'
    ],
    TaskCategory.EXPLANATION: [
        'README.md', 'ARCHITECTURE.md', 'DESIGN.md', 'docs/index.md', 'package.json', 'setup.py'
    ],
    TaskCategory.DOCUMENTATION: [
        'README.md', 'ARCHITECTURE.md', 'DESIGN.md', 'docs/*', 'package.json', 'setup.py'
    ],
    TaskCategory.TESTING: [
        'jest.config.js', 'pytest.ini', '.nycrc', 'phpunit.xml', 'karma.conf.js',
        'cypress.json', 'Gruntfile.js', 'Gulpfile.js', 'package.json', 'setup.py'
    ],
    TaskCategory.ARCHITECTURE: [
        'README.md', 'ARCHITECTURE.md', 'DESIGN.md', 'docs/design.md', 'package.json', 'setup.py',
        'docker-compose.yml', 'Dockerfile', 'kubernetes/*', 'helm/*'
    ],
    TaskCategory.ANALYSIS: [
        'package.json', 'setup.py', 'requirements.txt', 'Cargo.toml', 'pom.xml', 'build.gradle',
        'CMakeLists.txt', 'Makefile', 'README.md', 'docs/*'
    ],
    TaskCategory.OPTIMIZATION: [
        'package.json', 'setup.py', 'requirements.txt', 'Cargo.toml', 'pom.xml', 'build.gradle',
        'webpack.config.js', 'babel.config.js', 'tsconfig.json', 'Dockerfile', 'docker-compose.yml'
    ]
}


class TaskClassifier:
    """Analyzes user tasks and maps them to repository components."""

    def __init__(self):
        """Initialize the task classifier."""
        self.categories = TASK_KEYWORDS
        self.file_relevance = FILE_RELEVANCE
        self.important_files = IMPORTANT_FILES

    def classify_task(self, task_description: str) -> List[Tuple[TaskCategory, float]]:
        """
        Classify a task description into categories with confidence scores.

        Args:
            task_description: A string describing the user's task

        Returns:
            A list of tuples (category, confidence) sorted by confidence (highest first)
        """
        # Normalize task description
        task_lower = task_description.lower()
        
        # Calculate raw scores based on keyword matches
        scores = {}
        for category, keywords in self.categories.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                matches = len(re.findall(r'\b' + re.escape(keyword) + r'\b', task_lower))
                score += matches
            
            # Normalize by number of keywords to prevent bias towards categories with more keywords
            if score > 0:
                scores[category] = score / len(keywords)
        
        # If no matches, default to explanation (most general category)
        if not scores:
            return [(TaskCategory.EXPLANATION, 1.0)]
        
        # Normalize scores to sum to 1.0
        total_score = sum(scores.values())
        normalized_scores = {cat: score/total_score for cat, score in scores.items()}
        
        # Sort by score (highest first)
        sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_scores
    
    def get_primary_category(self, task_description: str) -> TaskCategory:
        """
        Get the primary category for a task description.

        Args:
            task_description: A string describing the user's task

        Returns:
            The primary TaskCategory
        """
        classifications = self.classify_task(task_description)
        if not classifications:
            return TaskCategory.EXPLANATION
        return classifications[0][0]
    
    def get_relevant_file_patterns(self, task_description: str) -> Dict[str, Set[str]]:
        """
        Get relevant file patterns for a task.

        Args:
            task_description: A string describing the user's task

        Returns:
            Dictionary with 'include' and 'exclude' sets of file patterns
        """
        primary_category = self.get_primary_category(task_description)
        
        # Get file relevance for the primary category
        relevance = self.file_relevance.get(primary_category, {})
        
        # Start with important files
        include_patterns = set(self.important_files.get(primary_category, []))
        
        # Add primary extensions
        for ext in relevance.get('primary', []):
            include_patterns.add(f"*{ext}")
        
        # Add secondary extensions with lower priority
        secondary_patterns = set()
        for ext in relevance.get('secondary', []):
            secondary_patterns.add(f"*{ext}")
        
        return {
            'primary': include_patterns,
            'secondary': secondary_patterns
        }
    
    def get_project_context_requirements(self, task_description: str) -> List[str]:
        """
        Get list of project context requirements for a task.

        Args:
            task_description: A string describing the user's task

        Returns:
            List of context requirements
        """
        primary_category = self.get_primary_category(task_description)
        
        common_requirements = [
            "README.md or project overview",
            "Main entry points (e.g., index.js, main.py)",
            "Project structure and organization"
        ]
        
        category_specific = {
            TaskCategory.CODE_GENERATION: [
                "Similar components or files",
                "Project conventions and patterns",
                "Type definitions and interfaces",
                "Configuration files"
            ],
            TaskCategory.DEBUGGING: [
                "Error logs or error details",
                "Related tests",
                "Recent changes to the codebase",
                "Configuration files"
            ],
            TaskCategory.REFACTORING: [
                "Code quality tools configuration",
                "Project style guides",
                "Related components",
                "Test coverage"
            ],
            TaskCategory.EXPLANATION: [
                "Documentation files",
                "Examples and usage",
                "Design decisions",
                "Architecture overview"
            ],
            TaskCategory.DOCUMENTATION: [
                "Existing documentation",
                "Code comments",
                "API definitions",
                "Usage examples"
            ],
            TaskCategory.TESTING: [
                "Test configuration",
                "Existing tests",
                "Code coverage reports",
                "CI/CD configuration"
            ],
            TaskCategory.ARCHITECTURE: [
                "Design documents",
                "System diagrams",
                "Infrastructure code",
                "Dependency graph"
            ],
            TaskCategory.ANALYSIS: [
                "Performance data",
                "User feedback",
                "Error logs",
                "Related components"
            ],
            TaskCategory.OPTIMIZATION: [
                "Performance benchmarks",
                "Profiling data",
                "Configuration files",
                "Resource usage metrics"
            ]
        }
        
        return common_requirements + category_specific.get(primary_category, [])