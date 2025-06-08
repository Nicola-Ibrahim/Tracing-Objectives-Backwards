import click

from ..generators.biobj import ParetoDataGenerating


@click.command()
@click.option(
    "--problem-id", required=True, help='Pareto problem ID (e.g., "F1", "F5")'
)
def generate_data(problem_id):
    ParetoDataGenerating(pareto_problem_id=int(problem_id)).generate()


if __name__ == "__main__":
    generate_data()
