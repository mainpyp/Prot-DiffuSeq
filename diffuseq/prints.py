logging.warning(f"Max seq length set to {max_seq_length}. Input length set to {expanded_inputs_length}. Target length set to {targets_length}.")
logging.warning(f"Training for {num_epochs} epochs with a train_batch_size of {train_batch_size} on {jax.device_count()} devices.")

logging.warning(f"Example Input shape {model_inputs['input_ids'].shape}.")
logging.warning(f"Example Input IDs {model_inputs['input_ids'][0]}.")
logging.warning(f"Lovely tensor analysis: {model_inputs['input_ids']}")
logging.warning(f"Example Input Tokens {tokenizer.decode(model_inputs['input_ids'][0])}.")

logging.warning(f"Example Label shape {model_inputs['labels'].shape}.")
logging.warning(f"Example Label IDs {model_inputs['labels'][0]}.")
logging.warning(f"Lovely tensor analysis: {model_inputs['labels']}")
logging.warning(f"Example Label Tokens {tokenizer.decode(model_inputs['labels'][0])}.")

logging.warning(f"Example Decoder input IDs {model_inputs['decoder_input_ids'].shape}.")
logging.warning(f"Example Decoder input IDs {model_inputs['decoder_input_ids'][0]}.")
logging.warning(f"Lovely tensor analysis: {model_inputs['decoder_input_ids']}")
logging.warning(f"Example Decoder input IDs {tokenizer.decode(model_inputs['decoder_input_ids'][0])}.")