from supabase import create_client

from src.utils.config import CONFIG


def get_client():
    env = CONFIG["env"]
    if not env.get("supabase_url") or not env.get("supabase_key"):
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not set in .env")
    return create_client(env["supabase_url"], env["supabase_key"])


def save_forecast_run(run_metadata: dict):
    client = get_client()
    return client.table("forecast_runs").insert(run_metadata).execute()


def fetch_forecast_runs(limit=20):
    client = get_client()
    return client.table("forecast_runs").select("*").order("created_at", desc=True).limit(limit).execute().data


def save_report(report_text: str, facts: dict, provider: str, grounding_ratio: float):
    client = get_client()
    payload = {
        "report_text": report_text,
        "facts": facts,
        "provider": provider,
        "grounding_ratio": grounding_ratio,
    }
    return client.table("reports").insert(payload).execute()


def fetch_reports(limit=20):
    client = get_client()
    return client.table("reports").select("*").order("created_at", desc=True).limit(limit).execute().data


def save_anomaly_flags(flags_df):
    client = get_client()
    records = flags_df.to_dict(orient="records")
    return client.table("anomaly_flags").insert(records).execute()


def fetch_anomaly_flags(limit=500):
    client = get_client()
    return client.table("anomaly_flags").select("*").order("date", desc=True).limit(limit).execute().data
