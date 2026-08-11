"""Thorlabs PM100(D) power meter via ``ThorlabsPM100`` + PyVISA."""

from __future__ import annotations

import time
from dataclasses import dataclass

# Thorlabs vendor / PM100D product IDs in VISA resource strings
_THORLABS_VID = "0x1313"
_PM100_PIDS = ("0x8078", "0x8072", "0x8076", "0x8079", "0x807A", "0x807B")


@dataclass
class PowerSample:
    watts: float
    t_s: float
    label: str = ""


def _visa_hint(exc: BaseException) -> str:
    msg = str(exc)
    if "VI_ERROR_IO" in msg or "-1073807298" in msg:
        return (
            "VISA I/O error — USB session is stuck. Close Thorlabs Optical Power Monitor, "
            "unplug the PM100 USB cable for ~3 s, plug it back in, then re-run."
        )
    if "VI_ERROR_NCIC" in msg or "-1073807264" in msg:
        return (
            "VISA NCIC — another host/app owns the USB interface. "
            "Close Optical Power Monitor / NI MAX and retry."
        )
    if "VI_ERROR_RSRC_BUSY" in msg or "VI_ERROR_RSRC_LOCKED" in msg:
        return "VISA resource busy/locked — close other software using the PM100."
    if "VI_ERROR_TMO" in msg:
        return "VISA timeout — check the sensor head is attached and the meter is powered."
    return "Check USB cable, sensor head, and that Optical Power Monitor is closed."


class PowerMeter:
    """Thin wrapper: open PM100, set wavelength, read power in watts."""

    def __init__(
        self,
        *,
        wavelength_nm: float = 400.0,
        visa_resource: str = "",
        serial: str = "",
        settle_s: float = 0.15,
        samples_per_read: int = 3,
        timeout_ms: int = 5000,
        open_retries: int = 4,
        open_retry_delay_s: float = 1.0,
    ) -> None:
        self.wavelength_nm = float(wavelength_nm)
        self.visa_resource = str(visa_resource or "").strip()
        self.serial = str(serial or "").strip()
        self.settle_s = float(settle_s)
        self.samples_per_read = max(1, int(samples_per_read))
        self.timeout_ms = int(timeout_ms)
        self.open_retries = max(1, int(open_retries))
        self.open_retry_delay_s = float(open_retry_delay_s)
        self._inst = None
        self._pm = None
        self._rm = None

    @staticmethod
    def list_thorlabs_resources() -> list[str]:
        import pyvisa

        rm = pyvisa.ResourceManager()
        out: list[str] = []
        try:
            for r in rm.list_resources():
                s = str(r)
                if _THORLABS_VID not in s:
                    continue
                if any(pid in s for pid in _PM100_PIDS) or "PM100" in s.upper():
                    out.append(s)
        finally:
            try:
                rm.close()
            except Exception:
                pass
        return out

    def open(self) -> None:
        if self._pm is not None:
            return

        candidates = self._candidate_resources()
        last_exc: BaseException | None = None
        for attempt in range(1, self.open_retries + 1):
            for resource in candidates:
                try:
                    self._open_resource(resource)
                    return
                except Exception as exc:
                    last_exc = exc
                    print(
                        f"[power_measure] open attempt {attempt}/{self.open_retries} "
                        f"failed on {resource}: {exc}"
                    )
                    self.close()
            if attempt < self.open_retries:
                time.sleep(self.open_retry_delay_s)

        hint = _visa_hint(last_exc) if last_exc is not None else ""
        raise RuntimeError(
            f"Failed to open Thorlabs PM100 (tried {candidates}). {hint}"
        ) from last_exc

    def _candidate_resources(self) -> list[str]:
        if self.visa_resource:
            return [self.visa_resource]
        resources = self.list_thorlabs_resources()
        if not resources:
            raise RuntimeError(
                "No Thorlabs PM100 VISA resource found. "
                "Install NI-VISA / Thorlabs drivers and plug in the meter."
            )
        if self.serial:
            needle = self.serial.upper()
            matched = [r for r in resources if needle in r.upper()]
            if not matched:
                raise RuntimeError(
                    f"No PM100 matching serial={self.serial!r}. Candidates: {resources}"
                )
            # Prefer exact serial match, then other Thorlabs resources as fallback.
            rest = [r for r in resources if r not in matched]
            return matched + rest
        if len(resources) > 1:
            print(
                f"[power_measure] multiple PM100 devices {resources}; "
                "trying in list order (set power_meter.serial to pin)"
            )
        return resources

    def _open_resource(self, resource: str) -> None:
        import pyvisa
        from ThorlabsPM100 import ThorlabsPM100

        print(f"[power_measure] opening power meter: {resource}")
        self._rm = pyvisa.ResourceManager()
        inst = self._rm.open_resource(resource)
        inst.timeout = self.timeout_ms
        try:
            inst.read_termination = "\n"
            inst.write_termination = "\n"
        except Exception:
            pass

        # Brief settle after USB open (helps after re-plug).
        time.sleep(0.35)

        try:
            idn = str(inst.query("*IDN?")).strip()
            print(f"[power_measure] meter IDN: {idn}")
        except Exception as exc:
            try:
                inst.close()
            except Exception:
                pass
            raise RuntimeError(f"PM100 *IDN? failed on {resource}: {exc}") from exc

        pm = ThorlabsPM100(inst=inst)
        try:
            pm.sense.correction.wavelength = float(self.wavelength_nm)
            time.sleep(0.2)
            wl = float(pm.sense.correction.wavelength)
            print(
                f"[power_measure] wavelength set to {wl:.1f} nm "
                f"(requested {self.wavelength_nm:.1f})"
            )
        except Exception as exc:
            try:
                inst.close()
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to set wavelength {self.wavelength_nm} nm — "
                f"is a power sensor head attached? ({exc})"
            ) from exc

        try:
            _ = float(pm.read)
        except Exception as exc:
            try:
                inst.close()
            except Exception:
                pass
            raise RuntimeError(
                f"PM100 read failed — check sensor head / close Optical Power Monitor "
                f"if another app holds the device ({exc})"
            ) from exc

        self._inst = inst
        self._pm = pm

    def close(self) -> None:
        if self._inst is not None:
            try:
                self._inst.close()
            except Exception:
                pass
        if self._rm is not None:
            try:
                self._rm.close()
            except Exception:
                pass
        self._inst = None
        self._pm = None
        self._rm = None

    def __enter__(self) -> "PowerMeter":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def read_watts(self, *, label: str = "") -> PowerSample:
        if self._pm is None:
            raise RuntimeError("PowerMeter not open")
        if self.settle_s > 0:
            time.sleep(self.settle_s)
        vals: list[float] = []
        for _ in range(self.samples_per_read):
            vals.append(float(self._pm.read))
            if self.samples_per_read > 1:
                time.sleep(0.02)
        watts = float(sum(vals) / len(vals))
        return PowerSample(watts=watts, t_s=time.time(), label=label)


def open_power_meter_from_config(cfg_power: object) -> PowerMeter:
    """Build + open from a ``PowerMeterConfig``-like object."""
    meter = PowerMeter(
        wavelength_nm=float(getattr(cfg_power, "wavelength_nm", 400.0)),
        visa_resource=str(getattr(cfg_power, "visa_resource", "") or ""),
        serial=str(getattr(cfg_power, "serial", "") or ""),
        settle_s=float(getattr(cfg_power, "settle_s", 0.15)),
        samples_per_read=int(getattr(cfg_power, "samples_per_read", 3)),
        timeout_ms=int(getattr(cfg_power, "timeout_ms", 5000)),
    )
    meter.open()
    return meter