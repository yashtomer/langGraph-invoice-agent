"""APScheduler setup — fires on the configured day-of-month.

Notes:
  * Uses ``SQLAlchemyJobStore`` so missed jobs (laptop asleep / process down)
    are not silently dropped.
  * ``misfire_grace_time=86400`` (24h) ensures a missed firing on the 25th
    triggers as soon as the process comes back up that day.
  * Job is replaced on every startup (``replace_existing=True``) so a config
    change to the day/time is honoured.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import Settings, get_settings
from .logging_setup import get_logger
from .runner import start_for_month

log = get_logger(__name__)

JOB_ID = "invoice_monthly_run"
MISFIRE_GRACE = 86_400  # 24h


def _current_month(tz: str) -> str:
    return datetime.now(ZoneInfo(tz)).strftime("%Y-%m")


def _scheduled_run(settings: Optional[Settings] = None) -> None:
    s = settings or get_settings()
    month = _current_month(s.timezone)
    log.info("scheduler.fire", month=month)
    try:
        start_for_month(month, settings=s)
    except Exception:  # noqa: BLE001
        log.exception("scheduler.run_failed", month=month)


def build_scheduler(settings: Optional[Settings] = None) -> BackgroundScheduler:
    s = settings or get_settings()
    jobstore = SQLAlchemyJobStore(url=f"sqlite:///{s.db_path}")
    sched = BackgroundScheduler(
        jobstores={"default": jobstore},
        timezone=ZoneInfo(s.timezone),
    )

    hh, mm = (s.invoice_time_local.split(":") + ["0"])[:2]
    trigger = CronTrigger(
        day=s.invoice_day_of_month,
        hour=int(hh),
        minute=int(mm),
        timezone=ZoneInfo(s.timezone),
    )

    sched.add_job(
        _scheduled_run,
        trigger=trigger,
        id=JOB_ID,
        name="Monthly invoice run",
        misfire_grace_time=MISFIRE_GRACE,
        coalesce=True,
        max_instances=1,
        replace_existing=True,
    )
    log.info(
        "scheduler.configured",
        day=s.invoice_day_of_month,
        time=s.invoice_time_local,
        tz=s.timezone,
    )
    return sched
