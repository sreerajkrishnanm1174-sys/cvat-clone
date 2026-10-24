"""CVAT calls this detector with an image and nothing else.

The UI action registers the task first. This function asks the analyzer,
which already knows the project, what to draw. If nothing is prepared,
it returns no shapes.
"""

import json
import os

import requests

ANALYZER = os.environ.get("ANALYZER_URL", "http://analyzer:8090")


def handler(context, event):
    image = event.body.get("image") if isinstance(event.body, dict) else None
    try:
        response = requests.post(
            ANALYZER.rstrip("/") + "/nuclio/detect",
            json={"image": image, "threshold": (event.body or {}).get("threshold")},
            timeout=120,
        )
        response.raise_for_status()
        shapes = response.json()
    except Exception:
        shapes = []
    return context.Response(
        body=json.dumps(shapes),
        headers={},
        content_type="application/json",
        status_code=200,
    )
