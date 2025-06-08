from ..adapters.interpolators.linear import LinearInterpolator
from ..adapters.loggers.wandb_logger import WandbLogger
from ..application.analyze_pareto_data import analyze_pareto_data
from ..domain.services.metrics import mean_squared_error
from ..domain.services.similarities import cosine_similarity

analyze_pareto_data()
