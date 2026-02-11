from segpinndic.DIC_importlib import *

# ============================================
# 读取种子点配置文件
# ============================================
def seed_config_txt(path, verbose=True):
    """
    Parse a config.txt file like the provided monocular/stereo configuration.
    Each parameter is preceded by a comment line starting with '# key:'.
    Automatically infer types (int, float, list, bool, None, str).
    """

    config = {}
    current_key = None
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            # Comment or key line
            if line.startswith("#"):
                if ":" in line:
                    # Extract key name after '#'
                    parts = line[1:].split(":", 1)
                    current_key = parts[0].strip()
                    if verbose:
                        print(f"[line {lineno}] ⏳ Detected key '{current_key}' -> waiting for value...")
                else:
                    current_key = None
                    if verbose:
                        print(f"[line {lineno}] 📝 Ignored comment: {line}")
                continue
            # Value line
            if current_key is None:
                if verbose:
                    print(f"[line {lineno}] ⚠️ Value without key ignored: {line}")
                continue
            raw_value = line
            # Try to parse value safely
            if raw_value.lower() == 'null':
                value = None
            elif raw_value.lower() == 'true':
                value = True
            elif raw_value.lower() == 'false':
                value = False
            else:
                try:
                    value = ast.literal_eval(raw_value)
                except Exception:
                    value = raw_value  # leave as string if eval fails
            config[current_key] = value
            if verbose:
                print(f"[line {lineno}] ✅ Loaded key '{current_key}' = {value}")
            current_key = None  # reset after reading value
    if verbose:
        print("\n=== ✅ Configuration loaded successfully ===")
        for k, v in config.items():
            print(f"  {k}: {v}")
    return SimpleNamespace(**config)


ALL_KEYS = [
    "input_dir", "output_dir",
    "hidden_units", "activation", "first_activation", "interpolation",
    "adam_epochs", "bfgs_epochs", "patience_adam", "patience_bfgs", 
    "delta_adam", "delta_bfgs", "seed_flag", "seed_train_epochs",
    "adam_lr", "adam_betas", "adam_decay", "bfgs_lr", "bfgs_max_iter",
    "print_loss_freq"
]
# ============================================
# 读取DIC配置文件
# ============================================
def DIC_config_txt(path, required_keys=ALL_KEYS, verbose=True):
    """
    Parse a config.txt file with lines like:
    # key: comment
    value
    Automatically infer types (int, float, list, None, bool, str).
    Checks all required_keys present.
    """
    config = {}
    current_key = None
    with open(path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            raw_line = line
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if ":" in line:  # 有冒号的，识别为参数
                    parts = line[1:].split(":", 1)
                    current_key = parts[0].strip()
                    if verbose:
                        print(f"[line {lineno}] ⏳ Detected key '{current_key}' -> waiting for value...")
                else:
                    if verbose:
                        print(f"[line {lineno}] 📝 Comment ignored: {raw_line.strip()}")
                    current_key = None
            else:
                if current_key is None:
                    if verbose:
                        print(f"[line {lineno}] ⚠️ Value without key, ignored: {raw_line.strip()}")
                    continue
                raw_value = line
                if raw_value.lower() == 'null':
                    value = None
                elif raw_value.lower() == 'true':
                    value = True
                elif raw_value.lower() == 'false':
                    value = False
                else:
                    try:
                        value = ast.literal_eval(raw_value)
                    except Exception:
                        value = raw_value
                config[current_key] = value
                if verbose:
                    print(f"[line {lineno}] ✅ Loaded key '{current_key}' = {value}")
                current_key = None

    # Check required keys
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    if verbose:
        print("\n=== ✅ All config loaded successfully ===")
        for k, v in config.items():
            print(f"  {k}: {v}")

    return SimpleNamespace(**config)