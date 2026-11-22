"""Probe what the Start menu actually looks like on THIS PC.

Run it in PowerShell (or cmd), from the Jas folder:

    python jas_selfcheck.py

While it counts down 10 seconds, open the Start menu yourself (click Start / press Win)
and type a couple of letters. It prints every window it sees, with its process and class,
so we can see exactly which one is the Start search - no guessing.

Nothing is clicked or typed except Windows' own windows being enumerated (read-only).
"""
import ctypes
import sys
import time

if sys.platform != "win32":
    sys.exit("This only runs on Windows.")

from ctypes import wintypes

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32


def proc_name(pid: int) -> str:
    """Image name of a pid, without psutil: OpenProcess + QueryFullProcessImageNameW."""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return "?"
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return buf.value.rsplit("\\", 1)[-1]
        return "?"
    finally:
        kernel32.CloseHandle(h)


def hwnd_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def title_of(hwnd: int) -> str:
    n = user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def windows():
    """[(hwnd, class, title, proc, visible)] - handles are 64-bit, so they must be c_void_p."""
    out = []
    PFN = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _l):
        try:
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            out.append((hwnd, hwnd_class(hwnd), title_of(hwnd), proc_name(pid.value),
                        bool(user32.IsWindowVisible(hwnd))))
        except Exception as e:
            out.append((hwnd, f"<error {e}>", "", "?", False))
        return True
    user32.EnumWindows(PFN(cb), 0)
    return out


def start_state():
    """What the current detection says, and why."""
    wins = windows()
    interesting = [(h, c, t, p) for h, c, t, p, v in wins if v and (
        p.lower() in ("searchhost.exe", "startmenuexperiencehost.exe", "shellexperiencehost.exe")
        or "start" in c.lower() or "corewindow" in c.lower() or "applicationframe" in c.lower()
        or t.strip().lower() in ("start", "démarrer", "hôte contextuel"))]
    print(f"\nWindows that could be Start ({len(interesting)}):")
    for h, c, t, p in interesting:
        print(f"  hwnd={hex(h) if h else h}  proc={p:<32} class={c:<45} title={t[:40]!r}")
    if not interesting:
        print("  (none - is the Start menu open right now?)")
    return wins


def sample(seconds: float = 14.0):
    """Poll the window list ~5x/second and record every window seen, with how often.

    Start is a transient flyout: a single sample will miss it. Also records the foreground window and whether it
    has a text caret, so we can tell 'Start is open' from 'Start is open AND has the keyboard'."""
    seen = {}                                  # (proc, class, title) -> [count, first_seen]
    fg_log = []                                # (proc, class, title) of the foreground window, deduped
    t0 = time.time()
    user32 = ctypes.windll.user32
    while time.time() - t0 < seconds:
        for hwnd, cls, title, proc, visible in windows():
            # count EVERY window, visible or not: Start's search is DWM-composited and reports visible=False
            key = (proc, cls, title, visible)
            seen.setdefault(key, [0, time.time() - t0])[0] += 1
        fg = user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(fg, ctypes.byref(pid))
        row = (proc_name(pid.value), hwnd_class(fg), title_of(fg), caret())
        if not fg_log or fg_log[-1] != row:
            fg_log.append(row)
        time.sleep(0.2)
    return seen, fg_log


def caret() -> bool:
    try:
        class GUITHREADINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("flags", wintypes.DWORD), ("hwndActive", wintypes.HWND),
                        ("hwndFocus", wintypes.HWND), ("hwndCapture", wintypes.HWND), ("hwndMenuOwner", wintypes.HWND),
                        ("hwndMoveSize", wintypes.HWND), ("hwndCaret", wintypes.HWND), ("rcCaret", wintypes.RECT)]
        info = GUITHREADINFO()
        info.cbSize = ctypes.sizeof(GUITHREADINFO)
        tid = ctypes.windll.user32.GetWindowThreadProcessId(ctypes.windll.user32.GetForegroundWindow(), None)
        return bool(ctypes.windll.user32.GetGUIThreadInfo(tid, ctypes.byref(info)) and info.hwndCaret)
    except Exception:
        return False


def main():
    print("Jas self-check - read-only. Ctrl+C to stop.")
    print(f"Elevated: {bool(ctypes.windll.shell32.IsUserAnAdmin())}")
    print("\n>>> NOW press the Win key (or click Start) and TYPE a few letters, e.g. 'not'.")
    print(">>> Keep the search open until the countdown ends.\n")
    import threading
    box = {}
    th = threading.Thread(target=lambda: box.update(zip(("seen", "fg"), sample(14.0))), daemon=True)
    th.start()
    for remaining in range(14, 0, -1):
        print(f"  ...{remaining}  ", end="\r", flush=True)
        time.sleep(1)
    th.join(timeout=20)
    seen, fg_log = box.get("seen", {}), box.get("fg", [])

    print("\n\nWindows seen while you had Start open (count | first seen | proc | class | title):")
    interesting = [(k, v) for k, v in seen.items() if (
        k[0].lower() in ("searchhost.exe", "startmenuexperiencehost.exe", "shellexperiencehost.exe",
                         "textinputhost.exe", "sihost.exe", "explorer.exe")
        or "start" in k[1].lower() or "corewindow" in k[1].lower()
        or (k[2] or "").strip().lower() in ("start", "démarrer", "hôte contextuel", "rechercher"))]
    for (proc, cls, title, vis), (count, first) in sorted(interesting, key=lambda r: -r[1][0]):
        print(f"  {count:>4}x  +{first:4.1f}s  {proc:<28} {cls[:32]:<32} {title[:24]!r:<26} visible={vis}")
    if not interesting:
        print("  (nothing plausible: the Start UI did not appear at all, or it is named differently above)")

    print("\nForeground window over time (proc | class | title | text caret):")
    for proc, cls, title, has_caret in fg_log[:40]:
        print(f"  {proc:<30} {cls[:34]:<34} {title[:30]!r:<32} caret={has_caret}")

    print("""
What this tells us:
  - visible=False next to SearchHost.exe is EXPECTED: Start's search is composited by DWM. Jas ignores that bit for
    Start detection now, and also checks the foreground window, so it is no longer a blocker.
  - If SearchHost / StartMenuExperienceHost appears ABOVE with a count, detection works.
  - If only TextInputHost or an untitled explorer/ApplicationFrameWindow appears, Start has no window we can
    identify, and Jas must never gate on it (it already does not: it uses the Start app list + shortcuts instead).
  - caret=True on the foreground line means the Start search box really has the keyboard.

Copy everything above into the chat.""")


if __name__ == "__main__":
    main()
