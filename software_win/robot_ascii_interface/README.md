# Robot ASCII interface (Windows)

```powershell
$env:PYTHONPATH = "$PWD\software_win"
python software_win\robot_ascii_interface\robot_ascii_interface.py --port COM3
```

Default port: `ROBOT_PORT` → VID/PID auto-detect → `COM3`.

Linux tty latency tuning is skipped on Windows; pyserial CDC works with the stock Windows driver.
