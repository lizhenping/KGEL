        output_config_file = "/home/DGNet/pre_model/config.json"
        output_model_file = "/home/DGNet/pre_model/pytorch_model.bin"
        config = GPT2Config.from_json_file(output_config_file)
        model = GPT2LMHeadModel(config)
        state_dict = torch.load(output_model_file)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            print("layer not same")
            pass