# filepath: /Users/guillermoabadlopez/Documents/GitHub/qgan_subspace/src/runner/loading_helpers.py
import os
import traceback


def load_models_if_specified(training_instance, load_timestamp, config_module, train_log_fn):
    """
    Loads generator and discriminator models if a load_timestamp is provided.
    Modifies training_instance.gen and training_instance.dis by calling their load_model methods.
    """
    if not load_timestamp:
        return

    loading_msg_prefix = f"[Timestamp: {load_timestamp}] "
    train_log_fn(f"{loading_msg_prefix}Attempting to load models.\\n", config_module.log_path)
    print(f"{loading_msg_prefix}Attempting to load models.")

    try:
        gen_model_filename = os.path.basename(config_module.model_gen_path)
        dis_model_filename = os.path.basename(config_module.model_dis_path)

        # Path structure from user's file: "generated_data/<timestamp>/saved_model/<model_filename>"
        load_gen_path = os.path.join("generated_data", load_timestamp, "saved_model", gen_model_filename)
        load_dis_path = os.path.join("generated_data", load_timestamp, "saved_model", dis_model_filename)

        train_log_fn(
            f"{loading_msg_prefix}Attempting to load Generator parameters from: {load_gen_path}\\n",
            config_module.log_path,
        )
        training_instance.gen.load_model(load_gen_path)
        train_log_fn(
            f"{loading_msg_prefix}Generator parameters loaded successfully from {load_gen_path}\\n",
            config_module.log_path,
        )
        print(f"{loading_msg_prefix}Generator parameters loaded from {load_gen_path}")

        train_log_fn(
            f"{loading_msg_prefix}Attempting to load Discriminator parameters from: {load_dis_path}\\n",
            config_module.log_path,
        )
        training_instance.dis.load_model(load_dis_path)
        train_log_fn(
            f"{loading_msg_prefix}Discriminator parameters loaded successfully from {load_dis_path}\\n",
            config_module.log_path,
        )
        print(f"{loading_msg_prefix}Discriminator parameters loaded from {load_dis_path}")

        train_log_fn(f"{loading_msg_prefix}Models loaded successfully. Continuing training.\\n", config_module.log_path)
        print(f"{loading_msg_prefix}Models loaded successfully. Continuing training.")

    except FileNotFoundError as e:
        error_msg = f"{loading_msg_prefix}ERROR: Could not load model files. File not found: {e}. Starting training from scratch instead.\\n"
        train_log_fn(error_msg, config_module.log_path)
        print(error_msg)
    except Exception as e:
        error_msg = f"{loading_msg_prefix}ERROR: An unexpected error occurred while loading models: {e}. Traceback: {traceback.format_exc()}. Starting training from scratch instead.\\n"
        train_log_fn(error_msg, config_module.log_path)
        print(error_msg)
