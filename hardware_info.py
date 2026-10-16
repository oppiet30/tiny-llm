"""Best-effort hardware detection without additional Python dependencies."""

import json
import os
import platform
import subprocess


def detect_hardware():
    info = {"cpu_model": platform.processor() or "Unknown",
            "cpu_cores": None, "ram_bytes": None}
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                info["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except (OSError, ImportError):
            pass
        try:
            script = (
                "$c=Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum;"
                "$m=Get-CimInstance Win32_ComputerSystem;"
                "@{cores=$c.Sum;ram=[long]$m.TotalPhysicalMemory}|ConvertTo-Json -Compress"
            )
            result = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, timeout=12, check=True,
            )
            data = json.loads(result.stdout)
            info["cpu_cores"] = int(data["cores"]) if data.get("cores") else None
            info["ram_bytes"] = int(data["ram"]) if data.get("ram") else None
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            pass
    elif platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                content = f.read()
            for line in content.splitlines():
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
            core_pairs = set()
            physical_id = core_id = None
            for block in content.strip().split("\n\n"):
                fields = dict(
                    line.split(":", 1) for line in block.splitlines() if ":" in line
                )
                if "physical id" in fields and "core id" in fields:
                    core_pairs.add((fields["physical id"].strip(), fields["core id"].strip()))
            if core_pairs:
                info["cpu_cores"] = len(core_pairs)
        except OSError:
            pass
        try:
            info["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (OSError, ValueError, AttributeError):
            pass
    return info
