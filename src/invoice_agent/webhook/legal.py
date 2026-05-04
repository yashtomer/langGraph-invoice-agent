"""Static legal pages required by Meta App Review (Privacy, Terms, Data Deletion).

Served as plain HTML from the FastAPI app so Meta can verify the URLs are
live and reachable. Update ``COMPANY``, ``CONTACT_EMAIL`` and ``LAST_UPDATED``
when the legal text changes.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

COMPANY = "Aeologic Technologies Ltd."
CONTACT_EMAIL = "support@aeologic.in"
LAST_UPDATED = "2026-05-02"

_BASE_CSS = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 760px; margin: 2.5rem auto; padding: 0 1.25rem;
         color: #1f2937; line-height: 1.55; }
  h1 { font-size: 1.6rem; margin-bottom: .25rem; }
  h2 { font-size: 1.15rem; margin-top: 1.6rem; }
  .meta { color: #6b7280; font-size: .85rem; margin-bottom: 1.6rem; }
  ul { padding-left: 1.2rem; }
  a { color: #2563eb; }
  hr { border: 0; border-top: 1px solid #e5e7eb; margin: 2rem 0; }
  footer { color: #6b7280; font-size: .85rem; }
</style>
"""

_PRIVACY = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>Privacy Policy — {COMPANY}</title>{_BASE_CSS}
</head><body>
<h1>Privacy Policy</h1>
<p class="meta">{COMPANY} · Last updated {LAST_UPDATED}</p>

<p>This Invoice Agent application (the "Service") is operated by {COMPANY} for
internal billing automation. This policy describes what data the Service
processes, why, and how to request its deletion.</p>

<h2>Data we process</h2>
<ul>
  <li><strong>WhatsApp phone number</strong> of the authorized billing contact, configured by the operator.</li>
  <li><strong>Inbound WhatsApp message text</strong> (project name, approval/rejection replies) sent by that contact during a monthly invoice flow.</li>
  <li><strong>Generated invoice metadata</strong>: invoice number, project name, month, PDF file path, status (started / sent / cancelled / errored).</li>
</ul>

<h2>Purpose</h2>
<p>The Service uses this data solely to (a) compose a monthly invoice PDF, (b)
deliver a preview to the authorized contact for approval over WhatsApp, and
(c) email the approved PDF to the configured accounts recipients. Data is not
used for advertising, profiling, or any unrelated purpose.</p>

<h2>Sub-processors</h2>
<ul>
  <li><strong>Meta Platforms, Inc.</strong> — WhatsApp Cloud API, for sending and
      receiving WhatsApp messages.</li>
  <li><strong>Microsoft Corporation</strong> — Microsoft Graph, for sending the
      final invoice email from a {COMPANY} mailbox.</li>
</ul>
<p>No data is shared with any other third party.</p>

<h2>Storage and retention</h2>
<p>All operational data lives in a local SQLite database on {COMPANY}
infrastructure. Generated PDFs live on the same server. Records are retained
for accounting and audit purposes until you request deletion (see below).</p>

<h2>Your rights</h2>
<p>You may at any time request access to, correction of, or deletion of any
data the Service holds about you. See our <a href="/data-deletion">Data Deletion Instructions</a>
or email <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a>.</p>

<h2>Contact</h2>
<p>{COMPANY} · <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a></p>

<hr><footer>© {COMPANY}. All rights reserved.</footer>
</body></html>"""

_TERMS = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>Terms of Service — {COMPANY}</title>{_BASE_CSS}
</head><body>
<h1>Terms of Service</h1>
<p class="meta">{COMPANY} · Last updated {LAST_UPDATED}</p>

<p>By using the Invoice Agent Service operated by {COMPANY} (the "Service"),
you agree to these Terms.</p>

<h2>1. Service description</h2>
<p>The Service automates monthly invoice generation and delivery between
{COMPANY} and pre-authorized billing contacts via WhatsApp and email.</p>

<h2>2. Authorized use</h2>
<p>The Service is provided to designated billing contacts only. You agree
not to attempt unauthorized access, reverse engineering, or interference
with the Service's operation.</p>

<h2>3. Accuracy of information</h2>
<p>You agree to provide accurate project and approval information. {COMPANY}
relies on those inputs to issue invoices and is not liable for billing
errors caused by incorrect information you supply.</p>

<h2>4. Disclaimer of warranties</h2>
<p>The Service is provided "as is" without warranty of any kind, express or
implied. {COMPANY} does not guarantee uninterrupted operation.</p>

<h2>5. Limitation of liability</h2>
<p>To the fullest extent permitted by law, {COMPANY}'s aggregate liability
arising out of or relating to the Service is limited to the fees paid for
the invoiced services in the preceding twelve months.</p>

<h2>6. Governing law</h2>
<p>These Terms are governed by the laws of India, with exclusive
jurisdiction in the courts of New Delhi.</p>

<h2>Contact</h2>
<p>{COMPANY} · <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a></p>

<hr><footer>© {COMPANY}. All rights reserved.</footer>
</body></html>"""

_DATA_DELETION = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><title>Data Deletion Instructions — {COMPANY}</title>{_BASE_CSS}
</head><body>
<h1>Data Deletion Instructions</h1>
<p class="meta">{COMPANY} · Last updated {LAST_UPDATED}</p>

<p>To request deletion of any data the Invoice Agent Service holds about
you, send an email to
<a href="mailto:{CONTACT_EMAIL}?subject=Invoice%20Agent%20data%20deletion">{CONTACT_EMAIL}</a>
from the WhatsApp-registered email or include the WhatsApp phone number
that was used with the Service.</p>

<h2>What gets deleted</h2>
<ul>
  <li>The <code>invoice_history</code> row for each month associated with you.</li>
  <li>LangGraph checkpoint state for each monthly thread (<code>invoice-YYYY-MM</code>).</li>
  <li>All generated PDF invoice files referencing your data.</li>
</ul>

<h2>Timeline</h2>
<p>Deletion is completed within 30 days of receipt of a verified request.
We will reply to confirm completion.</p>

<h2>Records we may retain</h2>
<p>If we are legally required to retain certain billing records for tax or
audit purposes, we will retain only the minimum data necessary and isolate
it from any active processing.</p>

<h2>Contact</h2>
<p>{COMPANY} · <a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a></p>

<hr><footer>© {COMPANY}. All rights reserved.</footer>
</body></html>"""


def build_legal_router() -> APIRouter:
    router = APIRouter()

    @router.get("/privacy", response_class=HTMLResponse)
    def privacy() -> str:
        return _PRIVACY

    @router.get("/terms", response_class=HTMLResponse)
    def terms() -> str:
        return _TERMS

    @router.get("/data-deletion", response_class=HTMLResponse)
    @router.get("/datadeletion", response_class=HTMLResponse)
    def data_deletion() -> str:
        return _DATA_DELETION

    return router
