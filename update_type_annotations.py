#!/usr/bin/env python3
"""
Script to update type annotations to modern Python 3.10+ syntax.

Changes:
- Dict -> dict
- List -> list
- Optional[X] -> X | None
- Tuple -> tuple
- Set -> set
"""

import re
from pathlib import Path

def modernize_type_hints(content: str) -> tuple[str, list[str]]:
    """Modernize type hints to Python 3.10+ syntax."""
    changes = []
    original = content

    # Replace Dict[K, V] with dict[K, V]
    pattern = r'\bDict\[([^\]]+)\]'
    if re.search(pattern, content):
        content = re.sub(pattern, r'dict[\1]', content)
        changes.append("Dict -> dict")

    # Replace List[T] with list[T]
    pattern = r'\bList\[([^\]]+)\]'
    if re.search(pattern, content):
        content = re.sub(pattern, r'list[\1]', content)
        changes.append("List -> list")

    # Replace Tuple[...] with tuple[...]
    pattern = r'\bTuple\[([^\]]+)\]'
    if re.search(pattern, content):
        content = re.sub(pattern, r'tuple[\1]', content)
        changes.append("Tuple -> tuple")

    # Replace Set[T] with set[T]
    pattern = r'\bSet\[([^\]]+)\]'
    if re.search(pattern, content):
        content = re.sub(pattern, r'set[\1]', content)
        changes.append("Set -> set")

    # Replace Optional[X] with X | None
    # Handle nested optionals carefully
    pattern = r'\bOptional\[([^\[\]]+)\]'
    while re.search(pattern, content):
        old_content = content
        content = re.sub(pattern, r'\1 | None', content)
        if content == old_content:
            break
        if "Optional[" not in changes:
            changes.append("Optional[X] -> X | None")

    # Remove unnecessary typing imports if everything is replaced
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        # Update typing imports - remove unused, keep needed
        if line.strip().startswith('from typing import'):
            # Extract what's being imported
            import_match = re.match(r'from typing import (.+)', line)
            if import_match:
                imports = [i.strip() for i in import_match.group(1).split(',')]
                # Keep only what's still needed
                keep_imports = []
                for imp in imports:
                    # Keep these as they're still needed
                    if imp in ['Any', 'Literal', 'TypedDict', 'Protocol', 'Callable',
                               'Union', 'cast', 'TYPE_CHECKING', 'Generic', 'TypeVar']:
                        keep_imports.append(imp)

                if keep_imports:
                    new_lines.append(f"from typing import {', '.join(keep_imports)}")
                else:
                    # Skip the import line entirely
                    changes.append("Removed unused typing imports")
                    continue
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    content = '\n'.join(new_lines)

    return content, changes

def process_file(file_path: Path) -> None:
    """Process a single Python file."""
    print(f"\nProcessing: {file_path}")

    content = file_path.read_text()
    new_content, changes = modernize_type_hints(content)

    if new_content != content:
        file_path.write_text(new_content)
        print(f"  ✓ Updated: {', '.join(changes)}")
    else:
        print(f"  ✓ No changes needed")

def main():
    """Process all Python files."""
    base_dir = Path(__file__).parent

    # Process main.py
    main_py = base_dir / "main.py"
    if main_py.exists():
        process_file(main_py)

    # Process all src/*.py files
    src_dir = base_dir / "src"
    if src_dir.exists():
        for py_file in sorted(src_dir.glob("*.py")):
            if py_file.name not in ['__init__.py']:
                process_file(py_file)

    print("\n✅ Type annotation modernization complete!")
    print("\nNext steps:")
    print("1. Run: mypy --strict main.py src/")
    print("2. Fix any type errors")
    print("3. Test the server")

if __name__ == "__main__":
    main()
