import traceback

import numpy as np

# Assuming Training class is in src.training (adjust if different)
# from ..training import Training # This relative import might need adjustment based on how you run the script
# For now, let's assume Training will be passed or imported directly in main


def run_single_training(config_module, train_log_fn, TrainingClass, load_timestamp=None):
    """
    Runs a single training instance (the default case when testing=False).
    """
    run_message = ""
    if load_timestamp:
        run_message = f"\nAttempting to load models from timestamp: {load_timestamp} and continue training...\n"
    else:
        run_message = "\nRunning with default configuration from config.py (new training)...\n"

    print(run_message.strip())  # Console feedback
    train_log_fn(run_message, config_module.get_log_path())  # Log to file

    try:
        # Pass the timestamp to the Training constructor if provided
        # The TrainingClass itself will handle the loading logic via load_models_if_specified (or similar)
        training_instance = TrainingClass(
            load_timestamp=load_timestamp, config_module=config_module, train_log_fn=train_log_fn
        )
        training_instance.run()
        success_msg = "\nDefault configuration run COMPLETED SUCCESSFULLY.\n"
        print(success_msg.strip())  # Console feedback
        train_log_fn(success_msg, config_module.get_log_path())  # Log to file
    except Exception as e:
        tb_str = traceback.format_exc()
        error_msg = (
            f"\n{'-' * 60}\n"
            f"Default configuration run FAILED!\n"
            f"Error Type: {type(e).__name__}\n"
            f"Error Message: {e!s}\n"
            f"Traceback:\n{tb_str}"
            f"{'=' * 60}\n"
        )
        print(error_msg)  # Console feedback
        train_log_fn(error_msg, config_module.get_log_path())  # Log to file


def run_test_configurations(config_module, train_log_fn, TrainingClass, test_configurations):
    """
    Runs a suite of test configurations.
    """
    # Store original config values that will be changed
    original_system_size = config_module.system_size
    original_generator_layers = config_module.generator_layers
    original_extra_ancilla = config_module.extra_ancilla
    original_iterations_epoch = config_module.iterations_epoch
    original_epochs = config_module.epochs
    original_label_base = "run_default_label"  # Default base if not otherwise set
    original_plot_every_epochs = getattr(config_module, "plot_every_epochs", config_module.epochs)
    original_discriminator_layers = config_module.discriminator_layers

    all_passed = True
    for i, config_params in enumerate(test_configurations):
        test_header_msg = f"\n{'=' * 60}\nRunning Test Configuration {i + 1}/{len(test_configurations)}: {config_params['label_suffix']}\n{'-' * 60}\n"
        print(test_header_msg)  # Essential for console feedback during tests
        train_log_fn(test_header_msg, config_module.get_log_path())  # Also log to file

        # Set config for the current test run
        config_module.system_size = config_params["system_size"]
        config_module.generator_layers = config_params["generator_layers"]  # Corrected key
        config_module.extra_ancilla = config_params["extra_ancilla"]
        config_module.iterations_epoch = config_params["iterations_epoch"]
        config_module.epochs = config_params["epochs"]
        setattr(config_module, "plot_every_epochs", config_params["epochs"])  # Plot at the end of this short run
        config_module.discriminator_layers = config_params["discriminator_layers"]  # Corrected key
        current_run_label = f"{original_label_base}_{config_params['label_suffix']}"
        setattr(config_module, "current_run_label", current_run_label)

        # Apply other test-specific config overrides from config_params
        if "target_type" in config_params:
            config_module.target_type = config_params["target_type"]
        if "tfim_h_param" in config_params:
            config_module.tfim_h_param = config_params["tfim_h_param"]
        if "ansatz_gen_type" in config_params:
            config_module.ansatz_gen_type = config_params["ansatz_gen_type"]
        if "ansatz_disc_type" in config_params:
            config_module.ansatz_disc_type = config_params["ansatz_disc_type"]
        if "generator_initial_state_type" in config_params:
            config_module.generator_initial_state_type = config_params["generator_initial_state_type"]
        if "cost_function_type" in config_params:
            config_module.cost_function_type = config_params["cost_function_type"]
            # Recalculate dependent constants if cost_function_type changes
            if config_module.cost_function_type == "original_qgan":
                config_module.s = np.exp(-1 / (2 * config_module.lamb)) - 1
                config_module.cst1 = (config_module.s / 2 + 1) ** 2
                config_module.cst2 = (config_module.s / 2) * (config_module.s / 2 + 1)
                config_module.cst3 = (config_module.s / 2) ** 2
            else:
                config_module.s = config_module.cst1 = config_module.cst2 = config_module.cst3 = None

        try:
            # For test configurations, we don't load previous models
            training_instance = TrainingClass(config_module=config_module, train_log_fn=train_log_fn)
            training_instance.run()
            success_msg = f"\n{'-' * 60}\nTest Configuration {i + 1} ({config_params['label_suffix']}) COMPLETED SUCCESSFULLY.\n{'=' * 60}\n"
            print(success_msg)  # Essential for console feedback
            train_log_fn(success_msg, config_module.get_log_path())
        except Exception as e:
            all_passed = False
            tb_str = traceback.format_exc()
            error_msg = (
                f"\n{'-' * 60}\n"
                f"Test Configuration {i + 1} ({config_params['label_suffix']}) FAILED!\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Message: {e!s}\n"
                f"Traceback:\n{tb_str}"
                f"{'=' * 60}\n"
            )
            print(error_msg)  # Essential for console feedback
            train_log_fn(error_msg, config_module.get_log_path())
            # Continue with other test configurations

    # Restore original config_module values
    config_module.system_size = original_system_size
    config_module.generator_layers = original_generator_layers
    config_module.extra_ancilla = original_extra_ancilla
    config_module.iterations_epoch = original_iterations_epoch
    config_module.epochs = original_epochs
    setattr(config_module, "plot_every_epochs", original_plot_every_epochs)
    if hasattr(config_module, "current_run_label"):
        delattr(config_module, "current_run_label")
    config_module.discriminator_layers = original_discriminator_layers

    final_summary_msg = ""
    if all_passed:
        final_summary_msg = "\nAll test configurations ran successfully! No errors detected during these runs.\n"
    else:
        final_summary_msg = "\nSome test configurations failed. Please review the logs above and the log file.\n"
    print(final_summary_msg)  # Essential for console feedback
    train_log_fn(final_summary_msg, config_module.get_log_path())
