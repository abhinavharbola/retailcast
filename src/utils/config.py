import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path) as f:
        config = yaml.safe_load(f)
    config["env"] = {
        "nim_api_key": os.getenv("NIM_API_KEY"),
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "supabase_url": os.getenv("SUPABASE_URL"),
        "supabase_key": os.getenv("SUPABASE_KEY"),
        "dagshub_token": os.getenv("DAGSHUB_TOKEN"),
        "dagshub_repo": os.getenv("DAGSHUB_REPO"),
    }
    return config


CONFIG = load_config()
