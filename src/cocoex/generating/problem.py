import cocoex


def get_problem():
    """Initialize the bbob-biobj F1 (Sphere/Sphere) problem."""
    suite = cocoex.Suite(
        "bbob-biobj",  # suite_name
        "",  # suite_instance
        "year: 2016 dimensions:2 instance_indices:1 function_indices:26",  # suite_options
    )
    problem = suite.get_problem(0)
    return problem


# suite = cocoex.Suite("bbob-biobj", "", "")
# observer = cocoex.Observer("bbob-biobj", "result_folder:doctest")
# for fun in suite:
#     print("Problem name:", fun.name)
#     print("Problem ID:", fun.id)
