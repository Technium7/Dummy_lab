# Robot ASCII interface

Small Python helper for the Dummy robot **ASCII serial protocol**: connect over a COM/tty port, send lines like `!START` or `#OFFSET_J 1`, and read text replies.

The authoritative command list lives in firmware: `firmware/dummy-ref-core-fw/UserApp/protocols/ascii_protocol.cpp`.

## Requirements

- Python 3
- [pyserial](https://pypi.org/project/pyserial/): `pip install pyserial`

## Raw command REPL

From the repository root you can run the module file directly:

```bash
python software/robot_ascii_interface/robot_ascii_interface.py --port COM3
```

Or, with the `software` directory on `PYTHONPATH`:

```bash
cd software
python -m robot_ascii_interface.robot_ascii_interface --port COM3
```

Options:

| Option | Default | Meaning |
|--------|---------|---------|
| `--port` | `COM3` | Serial device (Windows: `COMn`; Linux/macOS: e.g. `/dev/ttyUSB0`) |
| `--baudrate` | `115200` | Baud rate |
| `--timeout` | `1.0` | Serial read timeout (seconds) |
| `--send-wait-ms` | `200` | Delay after sending a line before reading the reply; increase if you often see timeouts |

At the prompt, type **full lines** exactly as the firmware expects (no extra spaces unless the protocol requires them). Examples:

- `#OFFSET_J 1`
- `!START`
- `#GETJPOS`
- `&0,0,90,0,0,0,20`

Built-in REPL commands: `help`, `exit`.

## Using from Python

Ensure the parent `software` folder is importable (for example insert it at the front of `sys.path`), then:

```python
from robot_ascii_interface import RobotAsciiInterface

robot = RobotAsciiInterface(port="COM3", baudrate=115200, timeout=1.0)
robot.connect()
robot.start()
print(robot.get_joint_pos())
robot.apply_joint_home_offset(1)  # same as sending "#OFFSET_J 1" on USB path
robot.disconnect()
```

You can also call `send_command("raw line")` for anything not wrapped as a method.

## USB vs UART4

In the reference firmware, **many `#` commands (including `#OFFSET_J`) are only implemented on the USB ASCII handler**, not on the smaller UART4 `#` branch. If a command works in the REPL over USB but not over a UART4 dongle, compare `OnUsbAsciiCmd` and `OnUart4AsciiCmd` in `ascii_protocol.cpp`.

## Related project code

- `software/manual_control/` uses `RobotAsciiInterface` for streaming joint commands during manual control.
