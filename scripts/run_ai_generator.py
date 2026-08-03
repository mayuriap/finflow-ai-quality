"""
scripts/run_ai_generator.py

Demonstrates AI-Powered Test Generation.
Run this to see the AI generate pytest tests from requirements.

Usage:
    python scripts/run_ai_generator.py
"""

import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from src.ai_test_generator.test_generator import AITestGenerator

console = Console()

def main():
    console.print(Panel.fit(
        "[bold blue]FinFlow AI Quality Platform[/bold blue]\n"
        "[green]Module 1: AI-Powered Test Case Generator[/green]",
        border_style="blue"
    ))

    generator = AITestGenerator()

    console.print("\n[yellow]Step 1: Loading requirements...[/yellow]")
    requirements = generator.load_requirements(
        "data/requirements/payment_requirements.txt"
    )
    console.print(f"[green]✓ Loaded {len(requirements)} characters[/green]")

    console.print("\n[yellow]Step 2: AI generating test scenarios...[/yellow]")
    scenarios = generator.generate_test_scenarios(requirements)

    # Display scenarios in a table
    table = Table(title=f"AI Generated {len(scenarios)} Test Scenarios")
    table.add_column("ID",          style="cyan",  width=8)
    table.add_column("Requirement", style="green", width=10)
    table.add_column("Test Name",   style="white", width=35)
    table.add_column("Type",        style="yellow",width=12)

    for s in scenarios:
        table.add_row(
            s.get("scenario_id", ""),
            s.get("requirement_id", ""),
            s.get("scenario_name", ""),
            s.get("test_type", "")
        )
    console.print(table)

    console.print("\n[yellow]Step 3: AI generating pytest code...[/yellow]")
    result = generator.generate_and_save(
        requirements_file="data/requirements/payment_requirements.txt",
        output_file="tests/ai_validation/test_payment_ai_RAW_generated.py"
    )

    console.print(Panel(
        f"[green]✓ Generated {result['scenarios_count']} test scenarios[/green]\n"
        f"[green]✓ Wrote {result['code_lines']} lines of pytest code[/green]\n"
        f"[green]✓ Saved to: {result['output_file']}[/green]\n\n"
        f"[yellow]Run the tests with:[/yellow]\n"
        f"[white]pytest {result['output_file']} -v[/white]",
        title="[bold green]Generation Complete[/bold green]",
        border_style="green"
    ))

    # Save scenarios summary
    with open("reports/scenarios_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    console.print("[green]✓ Summary saved to reports/scenarios_summary.json[/green]")

if __name__ == "__main__":
    import os
    os.makedirs("reports", exist_ok=True)
    main()