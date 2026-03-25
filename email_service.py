import logging
from typing import List, Optional

import resend

from config import settings


logger = logging.getLogger("vizai")

if settings.RESEND_API_KEY:
    resend.api_key = settings.RESEND_API_KEY


def send_scan_results_email(
    to_email: str,
    business_name: str,
    scan_id: str,
    discovery_score: int,
    accuracy_score: int,
    authority_score: int,
    overall_score: int,
    findings: List[str],
    package_recommendation: str,
    strategy_summary: str,
    request_id: str = "-",
) -> bool:
    """Send scan results to the user via Resend."""
    if not settings.email_notifications_enabled:
        logger.debug("Email notifications disabled, skipping scan results email", extra={"request_id": request_id})
        return False

    try:
        findings_html = ""
        if findings:
            findings_items = "".join([f"<li style='margin-bottom: 8px;'>{finding}</li>" for finding in findings[:5]])
            findings_html = f"""
            <div style="margin: 24px 0;">
                <h3 style="color: #b89cff; margin-bottom: 12px;">Key Findings:</h3>
                <ul style="padding-left: 20px; line-height: 1.6;">
                    {findings_items}
                </ul>
            </div>
            """

        if overall_score >= 80:
            score_color = "#4ade80"
            score_label = "Strong Visibility"
        elif overall_score >= 40:
            score_color = "#fbbf24"
            score_label = "Partial Visibility"
        else:
            score_color = "#f87171"
            score_label = "High-Risk Visibility"

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #0b0d13; color: #e4e6f1; margin: 0; padding: 0;">
            <div style="max-width: 600px; margin: 0 auto; padding: 40px 20px;">
                <div style="text-align: center; margin-bottom: 32px;">
                    <h1 style="color: #b89cff; margin: 0; font-size: 28px;">VizAI Scan Results</h1>
                    <p style="color: #a0a3b1; margin: 8px 0 0;">Your AI Visibility Report for {business_name}</p>
                </div>

                <div style="background: #14161f; border: 1px solid #1e2029; border-radius: 12px; padding: 24px; margin-bottom: 24px;">
                    <div style="text-align: center; margin-bottom: 24px;">
                        <div style="display: inline-block; background: {score_color}; color: #0b0d13; padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 14px; margin-bottom: 12px;">
                            {score_label}
                        </div>
                        <h2 style="font-size: 48px; margin: 8px 0; color: #e4e6f1;">{overall_score}/100</h2>
                        <p style="color: #a0a3b1; margin: 0; font-size: 14px;">Overall AI Visibility Score</p>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-top: 24px; padding-top: 24px; border-top: 1px solid #1e2029;">
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #b89cff;">{discovery_score}</div>
                            <div style="font-size: 12px; color: #a0a3b1; margin-top: 4px;">Discovery</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #b89cff;">{accuracy_score}</div>
                            <div style="font-size: 12px; color: #a0a3b1; margin-top: 4px;">Accuracy</div>
                        </div>
                        <div style="text-align: center;">
                            <div style="font-size: 24px; font-weight: bold; color: #b89cff;">{authority_score}</div>
                            <div style="font-size: 12px; color: #a0a3b1; margin-top: 4px;">Authority</div>
                        </div>
                    </div>
                </div>

                {findings_html}

                <div style="background: rgba(123, 92, 255, 0.1); border: 1px solid rgba(123, 92, 255, 0.2); border-radius: 12px; padding: 20px; margin: 24px 0;">
                    <h3 style="color: #b89cff; margin: 0 0 12px;">Recommended Next Step:</h3>
                    <p style="margin: 0; line-height: 1.6; color: #e4e6f1; font-weight: 500;">{package_recommendation}</p>
                    <p style="margin: 12px 0 0; line-height: 1.6; color: #a0a3b1; font-size: 14px;">{strategy_summary}</p>
                </div>

                <div style="text-align: center; margin: 32px 0;">
                    <a href="https://vizai.app/packages.html" style="display: inline-block; background: linear-gradient(120deg, #7b5cff, #b39cff); color: #0b0d13; padding: 14px 32px; border-radius: 8px; text-decoration: none; font-weight: bold; font-size: 16px;">
                        View Pricing & Packages
                    </a>
                </div>

                <div style="margin-top: 40px; padding-top: 24px; border-top: 1px solid #1e2029; text-align: center; font-size: 14px; color: #a0a3b1;">
                    <p style="margin: 8px 0;">Questions about your results? Reply to this email or contact us at <a href="mailto:hello@vizai.io" style="color: #b89cff; text-decoration: none;">hello@vizai.io</a></p>
                    <p style="margin: 8px 0;">Visit us at <a href="https://vizai.app" style="color: #b89cff; text-decoration: none;">vizai.app</a></p>
                    <p style="margin: 16px 0 0; font-size: 12px; color: #6b6e7f;">Scan ID: {scan_id}</p>
                </div>
            </div>
        </body>
        </html>
        """

        response = resend.Emails.send(
            {
                "from": settings.NOTIFY_EMAIL_FROM,
                "to": [to_email],
                "subject": f"Your VizAI Scan Results - {business_name} ({overall_score}/100)",
                "html": html_content,
            }
        )
        logger.info(
            "Scan results email sent successfully to %s (resend_id: %s)",
            to_email,
            response.get("id"),
            extra={"request_id": request_id},
        )
        return True
    except Exception as e:
        logger.error(
            "Failed to send scan results email to %s: %s",
            to_email,
            str(e),
            extra={"request_id": request_id},
        )
        return False


def send_contact_request_notification(
    business_name: str,
    contact_email: str,
    website: str,
    industry: Optional[str],
    scan_id: str,
    overall_score: int,
    request_id: str = "-",
) -> bool:
    """Notify admin team when someone requests to be contacted."""
    if not settings.email_notifications_enabled:
        logger.debug("Email notifications disabled, skipping contact request notification", extra={"request_id": request_id})
        return False

    try:
        industry_text = industry if industry else "Not specified"
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #f5f5f5; margin: 0; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; padding: 32px;">
                <h2 style="color: #7b5cff; margin: 0 0 24px;">New Contact Request from Scan</h2>
                <div style="background: #f8f9fa; border-left: 4px solid #7b5cff; padding: 16px; margin-bottom: 24px;">
                    <p style="margin: 0; font-weight: bold; color: #333;">Someone wants to discuss improving their results</p>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Business Name:</td><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; color: #333;">{business_name}</td></tr>
                    <tr><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Contact Email:</td><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0;"><a href="mailto:{contact_email}" style="color: #7b5cff;">{contact_email}</a></td></tr>
                    <tr><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Website:</td><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0;"><a href="{website}" style="color: #7b5cff;">{website}</a></td></tr>
                    <tr><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Industry:</td><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; color: #333;">{industry_text}</td></tr>
                    <tr><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; font-weight: bold; color: #666;">Overall Score:</td><td style="padding: 12px 0; border-bottom: 1px solid #e0e0e0; color: #333;"><strong>{overall_score}/100</strong></td></tr>
                    <tr><td style="padding: 12px 0; font-weight: bold; color: #666;">Scan ID:</td><td style="padding: 12px 0; color: #999; font-family: monospace; font-size: 12px;">{scan_id}</td></tr>
                </table>
                <div style="margin-top: 32px; padding: 16px; background: #f0f7ff; border-radius: 6px;">
                    <p style="margin: 0; color: #555; font-size: 14px;">
                        <strong>Next step:</strong> Reply to {contact_email} within 1 business day to discuss their AI visibility needs.
                    </p>
                </div>
            </div>
        </body>
        </html>
        """

        response = resend.Emails.send(
            {
                "from": settings.NOTIFY_EMAIL_FROM,
                "to": [settings.NOTIFY_EMAIL_TO],
                "subject": f"Contact Request: {business_name} (Score: {overall_score}/100)",
                "html": html_content,
                "reply_to": contact_email,
            }
        )
        logger.info(
            "Contact request notification sent to admin (resend_id: %s)",
            response.get("id"),
            extra={"request_id": request_id},
        )
        return True
    except Exception as e:
        logger.error(
            "Failed to send contact request notification: %s",
            str(e),
            extra={"request_id": request_id},
        )
        return False
