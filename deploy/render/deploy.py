"""Create/update the SpeechMirror API service on Render (free tier).

Uses the Render Blueprint API against the repo-root render.yaml:
  1. GET  /v1/me                       -> owner id
  2. POST /v1/blueprints/create-plan   -> validate render.yaml
  3. POST /v1/blueprints               -> deploy the blueprint
  4. poll GET /v1/services             -> wait until LIVE, print URL
  5. --cors <origin>  : PUT service envVars to set CORS_ORIGINS

Usage:
  $env:RENDER_API_KEY="..."            # or pass --key
  python deploy/render/deploy.py                       # create + wait
  python deploy/render/deploy.py --cors https://foo.vercel.app
Requires the repo (render.yaml, Dockerfile.render, API fixes) pushed to
GitHub — Render builds from the branch, not from local files.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

API = "https://api.render.com/v1"
ROOT = Path(__file__).resolve().parents[2]


def req(method: str, path: str, key: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        API + path,
        data=data,
        method=method,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def get_service(key: str, name: str) -> dict | None:
    for item in req("GET", "/services?key=" + name, key).get("services", []):
        svc = item.get("service") or {}
        if svc.get("name") == name:
            return svc
    return None


def put_cors(key: str, svc: dict, origin: str) -> None:
    env = svc.get("envVars") or []
    env = [e for e in env if e.get("key") != "CORS_ORIGINS"]
    env.append({"key": "CORS_ORIGINS", "value": origin})
    req("PUT", f"/services/{svc['id']}", key, {"envVars": env})
    print(f"CORS_ORIGINS -> {origin} (Render will redeploy)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default=None, help="Render API key (or RENDER_API_KEY env)")
    parser.add_argument("--name", default="speechmirror-api")
    parser.add_argument("--cors", default=None, help="set CORS_ORIGINS on the existing service and exit")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    import os

    key = args.key or os.getenv("RENDER_API_KEY")
    if not key:
        sys.exit("No API key: set RENDER_API_KEY or pass --key")

    owner = req("GET", "/me", key)["id"]
    svc = get_service(key, args.name)

    if args.cors:
        if not svc:
            sys.exit(f"Service '{args.name}' not found — run without --cors first.")
        svc = req("GET", f"/services/{svc['id']}", key)["service"]
        put_cors(key, svc, args.cors)
        return

    if svc:
        print(f"Service already exists: {svc['name']} ({svc.get('id')})")
    else:
        yaml_text = (ROOT / "render.yaml").read_text(encoding="utf-8")
        plan = req("POST", "/blueprints/create-plan", key,
                   {"ownerId": owner, "blueprint": yaml_text})
        payload = plan.get("payload") or plan.get("serviceDetails") or []
        created = req("POST", "/blueprints", key,
                      {"ownerId": owner, "blueprintId": plan["blueprintId"], "payload": payload})
        print("Blueprint deployed:", json.dumps(
            {k: created.get(k) for k in ("id", "name", "jobId") if k in created}, indent=2))

    if args.no_wait:
        return

    print("Waiting for build (free tier can take ~5-10 min)...")
    for _ in range(80):
        time.sleep(15)
        s = get_service(key, args.name)
        if not s:
            continue
        s = req("GET", f"/services/{s['id']}", key)["service"]
        status = s.get("serviceDetails", {}).get("status") or s.get("status")
        print(f"  status: {status}")
        if status == "LIVE":
            print("\nAPI URL:", s["externalUrl"] or s["url"])
            return
        if status in ("ERRORED", "UPDATE_ERRORED"):
            sys.exit(f"Build errored — check https://dashboard.render.com ({s.get('id')})")
    sys.exit("Timed out waiting for service to go LIVE")


if __name__ == "__main__":
    main()
