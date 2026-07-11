from fastapi import APIRouter
from fastapi.responses import HTMLResponse

report_router = APIRouter()


@report_router.get("/{insight_id}", response_class=HTMLResponse)
async def get_report(insight_id: int) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html>
<head><title>App Store Review Report</title></head>
<body>
<h1>Review analysis report (insight #{insight_id})</h1>
<p>Mocked report — metrics, sentiment, keywords and actionable insights render here.</p>
</body>
</html>"""
    )
