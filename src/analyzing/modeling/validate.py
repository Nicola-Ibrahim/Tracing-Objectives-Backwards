from ..domain.model import InterpolatorModel
from ..domain.preference import ObjectivePreferences


def validate():
    # Load interpolator with metadata
    loaded = InterpolatorModel.load_from_file("rbf_local")
    print(loaded.model_type)
    print(loaded.created_at)
    print(type(loaded.model))  # should be RBFInterpolator

    # Evaluate on validation set
    mse_vals = []
    for idx, time_weight, energy_weight in enumerate(y_val):
        target_prefs = ObjectivePreferences(time_weight, energy_weight)

        # Generate recommendation (returns normalized decision vector)
        X_pred_norm, y_pred_norm = interpolator.generate(target_prefs)

        # Denormalize predictions to original scale
        X_pred = x_normalizer.inverse_transform(X_pred_norm)
        Y_pred = y_normalizer.inverse_transform(y_pred_norm)

        # Calculate error against true validation target
        mse = mean_squared_error(y_val[idx], Y_pred)
        mse_vals.append(mse)

    print(f"Average validation MSE: {np.mean(mse_vals):.6f}")


if __name__ == "__main__":
    validate()
