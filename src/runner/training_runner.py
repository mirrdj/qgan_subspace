import traceback

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
    train_log_fn(run_message, config_module.log_path)  # Log to file

    try:
        # Pass the timestamp to the Training constructor if provided
        # The TrainingClass itself will handle the loading logic via load_models_if_specified (or similar)
        training_instance = TrainingClass(
            load_timestamp=load_timestamp, config_module=config_module, train_log_fn=train_log_fn
        )
        training_instance.run()
        success_msg = "\nDefault configuration run COMPLETED SUCCESSFULLY.\n"
        print(success_msg.strip())  # Console feedback
        train_log_fn(success_msg, config_module.log_path)  # Log to file
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
        train_log_fn(error_msg, config_module.log_path)  # Log to file


def run_test_configurations(config_module, train_log_fn, TrainingClass, test_configurations):
    """
    Runs a suite of test configurations.
    """
    # Store original config values that will be changed
    original_system_size = config_module.system_size
    original_layer = config_module.layer
    original_extra_ancilla = config_module.extra_ancilla
    original_iterations_epoch = config_module.iterations_epoch
    original_epochs = config_module.epochs
    original_label = getattr(config_module, "label", "run_default_label")
    original_plot_every_epochs = config_module.plot_every_epochs
    original_num_discriminator_layers = config_module.num_discriminator_layers

    all_passed = True
    for i, config_params in enumerate(test_configurations):
        test_header_msg = f"\n{'=' * 60}\nRunning Test Configuration {i + 1}/{len(test_configurations)}: {config_params['label_suffix']}\n{'-' * 60}\n"
        print(test_header_msg)  # Essential for console feedback during tests
        train_log_fn(test_header_msg, config_module.log_path)  # Also log to file

        # Set config for the current test run
        config_module.system_size = config_params["system_size"]
        config_module.layer = config_params["layer"]
        config_module.extra_ancilla = config_params["extra_ancilla"]
        config_module.iterations_epoch = config_params["iterations_epoch"]
        config_module.epochs = config_params["epochs"]
        config_module.plot_every_epochs = config_params["epochs"]  # Plot at the end of this short run
        config_module.num_discriminator_layers = config_params["num_discriminator_layers"]
        config_module.label = f"{original_label}_{config_params['label_suffix']}"

        try:
            # For test configurations, we don't load previous models
            training_instance = TrainingClass(config_module=config_module, train_log_fn=train_log_fn)
            training_instance.run()
            success_msg = f"\n{'-' * 60}\nTest Configuration {i + 1} ({config_params['label_suffix']}) COMPLETED SUCCESSFULLY.\n{'=' * 60}\n"
            print(success_msg)  # Essential for console feedback
            train_log_fn(success_msg, config_module.log_path)
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
            train_log_fn(error_msg, config_module.log_path)
            # Continue with other test configurations

    # Restore original config_module values
    config_module.system_size = original_system_size
    config_module.layer = original_layer
    config_module.extra_ancilla = original_extra_ancilla
    config_module.iterations_epoch = original_iterations_epoch
    config_module.epochs = original_epochs
    config_module.label = original_label
    config_module.plot_every_epochs = original_plot_every_epochs
    config_module.num_discriminator_layers = original_num_discriminator_layers

    final_summary_msg = ""
    if all_passed:
        final_summary_msg = "\nAll test configurations ran successfully! No errors detected during these runs.\n"
    else:
        final_summary_msg = "\nSome test configurations failed. Please review the logs above and the log file.\n"
    print(final_summary_msg)  # Essential for console feedback
    train_log_fn(final_summary_msg, config_module.log_path)
