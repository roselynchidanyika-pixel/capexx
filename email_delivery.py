"""E-mail delivery for the management report.

Reads SMTP credentials ONLY from environment variables (or Streamlit
secrets surfaced via os.environ): SMTP_HOST / SMTP_PORT / SMTP_USER /
SMTP_PASS. Credentials are never hard-coded.

The recipient is always masked in the success message (r****@gmail.com)
and the address is format-validated before any send attempt.
"""
from __future__ import annotations

import os
import re
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr
from typing import Any, Dict, Optional

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


def validate_email(address: str) -> bool:
    """Basic format validation of an e-mail address."""
    return bool(address and isinstance(address, str) and _EMAIL_RE.match(address.strip()))


def mask_email(address: str) -> str:
    """Mask a recipient for display: r****@gmail.com."""
    address = (address or "").strip()
    if "@" not in address:
        return "****"
    local, _, domain = address.partition("@")
    if not local:
        return "****@" + domain
    head = local[0]
    return "%s****@%s" % (head, domain)


def _get_credential(key: str) -> Optional[str]:
    """Read one credential from env, honouring Streamlit secrets."""
    try:
        import streamlit as st  # may be absent in headless mode
        x = st.secrets.get("smtp", {}).get(key)
        if x:
            return str(x)
    except Exception:
        pass
    return os.environ.get(key)


def send_report_email(
    recipient: str,
    subject: str,
    body: str,
    attachment_path: Optional[str] = None,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    smtp_user: Optional[str] = None,
    smtp_pass: Optional[str] = None,
) -> Dict[str, Any]:
    """Send the report e-mail and return {success, message}.

    Resolution order for SMTP settings: explicit kwargs -> SMTP_* env vars
    -> Streamlit secrets. Fails with a readable message when credentials are
    missing or the recipient address is invalid.
    """
    recipient = (recipient or "").strip()
    if not validate_email(recipient):
        return {"success": False,
                "message": "Invalid recipient e-mail address format: '%s'." % recipient}

    host = smtp_host or _get_credential("SMTP_HOST")
    port = smtp_port or _int_or(_get_credential("SMTP_PORT"), 587)
    user = smtp_user or _get_credential("SMTP_USER")
    password = smtp_pass or _get_credential("SMTP_PASS")

    if not host:
        return {"success": False,
                "message": "SMTP host is not configured. Set SMTP_HOST (and SMTP_USER/SMTP_PASS) in the environment or .env."}
    if not user or not password:
        return {"success": False,
                "message": "SMTP credentials missing. Set SMTP_USER and SMTP_PASS in the environment or .env."}

    try:
        msg = MIMEMultipart("mixed")
        msg["From"] = formataddr(("Vision 2030 CapEx AI Agent", user))
        msg["To"] = recipient
        msg["Subject"] = subject or "Vision 2030 CapEx AI Agent - Management Report"

        msg.attach(MIMEText(body or "", "plain", "utf-8"))

        if attachment_path and os.path.exists(attachment_path):
            part = MIMEBase("application", "octet-stream")
            with open(attachment_path, "rb") as fh:
                part.set_payload(fh.read())
            from email import encoders
            encoders.encode_base64(part)
            fname = os.path.basename(attachment_path)
            part.add_header("Content-Disposition", "attachment",
                            filename=("utf-8", "", fname))
            msg.attach(part)

        with smtplib.SMTP(host, int(port), timeout=25) as server:
            server.ehlo()
            if int(port) == 465:
                server.starttls()
            try:
                server.login(user, password)
            except smtplib.SMTPAuthenticationError as e:
                return {"success": False,
                        "message": "SMTP authentication failed: %s" % e}
            server.sendmail(user, [recipient], msg.as_string())

        return {"success": True,
                "message": "Report sent to %s." % mask_email(recipient)}
    except smtplib.SMTPException as e:
        return {"success": False, "message": "SMTP error: %s" % e}
    except Exception as e:  # noqa: BLE001 - surface readable error to UI
        return {"success": False, "message": "E-mail delivery failed: %s" % e}


def _int_or(v: Optional[str], default: int) -> int:
    try:
        return int(str(v)) if v else default
    except (TypeError, ValueError):
        return default