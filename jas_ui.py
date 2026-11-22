#!/usr/bin/env python3
"""
jas_ui.py - lets Jas operate apps that only have a graphical interface (Windows).

    pip install pywinauto pyautogui psutil
    pip install rapidocr-onnxruntime        # optional: lets Jas READ the screen of apps that expose no controls

How it works (no vision model needed - your models are text-only)
  * ui_windows / ui_inspect read the app's accessibility tree (UI Automation): Jas gets a numbered list of the
    window's buttons, fields, menus and texts, then clicks/types by number. This is exact and fast.
  * ui_read_screen (optional OCR) turns what is visible into text with coordinates, for apps that expose
    nothing (games, custom-drawn UIs); ui_click then works with x/y.

Safety
  * the first time Jas touches an app window, you must approve THAT window; risky buttons (Delete, Send, Pay,
    Install, Allow, ...) ask again every time; /yolo in Jas skips these prompts
  * Jas can never operate the terminal it runs in (otherwise it could answer its own approval prompts)
  * it refuses password fields and anything that looks like a password/PIN/card field
  * emergency stop: push the mouse into any screen corner, or press Ctrl+C in the Jas terminal
  * apps running as administrator can only be controlled if Jas is also run as administrator
"""
import ctypes
import difflib
import os
import re
import subprocess
import sys
import time
from collections import deque

APPROVE = lambda msg: False          # jas.py replaces this with its own y/N prompt
GRANTED = set()                      # process ids the user has allowed Jas to control (this session)
ELEMS = {"pid": None, "items": {}}   # element number -> UIA element, from the latest ui_inspect
LAST = {"hwnd": None, "pid": None, "title": "", "w": None}   # the app window Jas is currently working on
SCREEN = {"granted": False}                                  # whole-screen reading allowed this session
CLICKED = {"last": 0.0}   # when Jas last clicked: a screen read after navigation is not a landing page
KNOWN = set()                                                # pids seen by the previous ui_windows call
TERMINALS = {"windowsterminal.exe", "openconsole.exe", "conhost.exe", "cmd.exe", "powershell.exe", "pwsh.exe",
             "wt.exe", "alacritty.exe", "wezterm-gui.exe", "mintty.exe"}
CONTROL_TYPES = {"Button", "SplitButton", "Edit", "Document", "ComboBox", "CheckBox", "RadioButton", "MenuItem",
                 "Menu", "MenuBar", "TabItem", "ListItem", "TreeItem", "DataItem", "Hyperlink", "Slider", "Spinner",
                 "ToolBar", "Text"}
RISKY = re.compile(r"\b(delete|remove|erase|format|uninstall|reset|empty|clear all|send|submit|post|publish|pay|buy|"
                   r"purchase|place order|checkout|confirm|transfer|withdraw|sign out|log ?out|discard|overwrite|"
                   r"install|allow|accept|agree|grant|authorize|run anyway)\b", re.I)
SENSITIVE = re.compile(r"(?i)pass(word|code|phrase)?|\bpin\b|\bcvv\b|\bcvc\b|card number|secret|api key|token|2fa|one-time")
TERMINAL_HINT = re.compile(r"(?i)^(jas|jas monitor|llama-swap)\b")
DESKTOP = {"program manager", "progman", "workerw"}      # the shell's desktop window: not an app
SHELL_PROC = {"explorer.exe"}                            # Explorer: only its real app windows, never the desktop/taskbar


class UIError(Exception):
    pass


def guard(fn):
    """Turn every failure into an 'ERROR: ...' string that Jas shows in red and the model can act on."""
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except UIError as e:
            return f"ERROR: {e}"
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {str(e)[:300]}"
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


def configure(approve):
    global APPROVE
    APPROVE = approve


def available() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import pywinauto, pyautogui, psutil  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------- pure helpers (testable anywhere)
def escape_keys(text: str) -> str:
    """Make arbitrary text safe for pywinauto's send_keys (which treats + ^ % ~ ( ) { } [ ] as commands)."""
    return "".join("{" + c + "}" if c in "+^%~(){}[]" else c for c in text)


KEY_ALIASES = {"control": "ctrl", "return": "enter", "escape": "esc", "del": "delete", "pgup": "pageup",
               "pgdn": "pagedown", "pagedn": "pagedown", "windows": "win", "cmd": "win", "super": "win",
               "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right", "spacebar": "space",
               "bksp": "backspace", "ins": "insert"}
KNOWN_KEYS = set("abcdefghijklmnopqrstuvwxyz0123456789") | {
    "ctrl", "alt", "shift", "win", "enter", "tab", "esc", "space", "backspace", "delete", "insert", "home", "end",
    "pageup", "pagedown", "up", "down", "left", "right", "capslock", "printscreen",   # no "apps": that is the
                                                                                     # context-menu key
    *{f"f{i}" for i in range(1, 13)}, ",", ".", "/", ";", "'", "[", "]", "\\", "-", "=", "`"}


def parse_keys(spec: str) -> list:
    """'ctrl+a delete' -> [['ctrl','a'], ['delete']]. Raises UIError for unknown key names."""
    combos = []
    for combo in (spec or "").lower().split():
        parts = [KEY_ALIASES.get(p, p) for p in combo.split("+") if p]
        bad = [p for p in parts if p not in KNOWN_KEYS]
        if bad or not parts:
            raise UIError(f"unknown key name(s) {bad or [combo]}; use names like ctrl, alt, shift, enter, tab, esc, "
                          f"delete, f5, left, a, 1 - for example 'ctrl+s' or 'ctrl+a delete'")
        combos.append(parts)
    if not combos:
        raise UIError("no keys given")
    return combos


def is_risky(name: str) -> bool:
    return bool(RISKY.search(name or ""))


def is_sensitive(name: str) -> bool:
    return bool(SENSITIVE.search(name or ""))


def is_desktop(title: str, pid: int) -> bool:
    """The desktop / taskbar is not an app: Jas must never pick it as a target."""
    if (title or "").strip().lower() in DESKTOP:
        return True
    return _proc_name(pid).lower() in SHELL_PROC and not (title or "").strip()


def fmt_element(n: int, ctype: str, name: str, auto_id: str, value: str, enabled: bool) -> str:
    s = f'[{n}] {ctype}'
    if name:
        s += f' "{name[:70]}"'
    if auto_id and auto_id != name:
        s += f" #{auto_id[:30]}"
    if value:
        s += f' value="{value[:60]}"'
    if not enabled:
        s += " (disabled)"
    return s


# ---------------------------------------------------------------- Windows plumbing (replaced in tests)
def _deps():
    if sys.platform != "win32":
        raise UIError("UI control only works on Windows")
    try:
        import pywinauto
        import pyautogui
    except ImportError as e:
        raise UIError(f"missing package ({e}). Run: pip install pywinauto pyautogui psutil")
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    pyautogui.FAILSAFE = False                    # we do our own corner check (it also covers pywinauto clicks)
    pyautogui.PAUSE = 0.05
    return pywinauto, pyautogui


def _failsafe():
    """Abort if the mouse sits in a screen corner: the user's emergency stop."""
    if sys.platform != "win32":
        return

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
    w, h = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
    if pt.x <= 2 and pt.y <= 2 or pt.x >= w - 3 and pt.y <= 2 or pt.x <= 2 and pt.y >= h - 3 or pt.x >= w - 3 and pt.y >= h - 3:
        raise UIError("aborted: the mouse is in a screen corner (emergency stop). Move it away to continue.")


def _foreground_hwnd() -> int:
    return ctypes.windll.user32.GetForegroundWindow()


def _foreground_pid() -> int:
    user32 = ctypes.windll.user32
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
    return pid.value


def _pid_at(x: int, y: int) -> int:
    user32 = ctypes.windll.user32

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    hwnd = user32.WindowFromPoint(POINT(int(x), int(y)))
    root = user32.GetAncestor(hwnd, 2) or hwnd
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(root, ctypes.byref(pid))
    return pid.value


def _foreground_rect():
    class RECT(ctypes.Structure):
        _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long), ("r", ctypes.c_long), ("b", ctypes.c_long)]
    r = RECT()
    ctypes.windll.user32.GetWindowRect(ctypes.windll.user32.GetForegroundWindow(), ctypes.byref(r))
    return r.l, r.t, r.r, r.b


def _proc_name(pid: int) -> str:
    """Image name of a pid. '' when it cannot be read (never a placeholder: callers must be able to trust '')."""
    try:
        import psutil
        return psutil.Process(pid).name() or ""
    except Exception:
        pass
    try:                                                     # psutil missing or refused: ask Windows directly
        from ctypes import wintypes
        kernel32 = ctypes.windll.kernel32
        h = kernel32.OpenProcess(0x1000, False, int(pid))    # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                return buf.value.rsplit("\\", 1)[-1]
            return ""
        finally:
            kernel32.CloseHandle(h)
    except Exception:
        return ""


def _protected_pids() -> set:
    """The terminal Jas runs in (and its ancestors): Jas must never be able to operate these."""
    pids = {os.getpid()}
    try:
        import psutil
        pids.update(p.pid for p in psutil.Process(os.getpid()).parents())
    except Exception:
        pass
    return pids


def _check_pid(pid: int, title: str = ""):
    """Raise unless Jas may operate this process; asks the user the first time."""
    name = _proc_name(pid)
    if is_desktop(title, pid):
        raise UIError(f"'{title or name}' is the Windows desktop, not an app. Only operate real application "
                      f"windows; call ui_windows to list them.")
    if pid in _protected_pids() or name.lower() in TERMINALS or TERMINAL_HINT.match(title or ""):
        raise UIError(f"'{title or name}' is a terminal/console (possibly the one Jas runs in). Jas is not allowed to "
                      f"operate terminals; use run_shell for command-line work.")
    if pid in GRANTED:
        return
    if not APPROVE(f"Jas wants to CONTROL the app window '{title or name}' ({name}): it will be able to click and "
                   f"type in it until you quit Jas."):
        raise UIError("the user did not allow Jas to control that window. Do not retry; ask what they would prefer.")
    GRANTED.add(pid)


# ---------------------------------------------------------------- focus handling
# Approval prompts happen in Jas's terminal, which takes the focus. So before every action that needs the keyboard
# or the mouse, Jas brings the app it is working on back to the front - and it never types into anything else.
def _remember(w, title: str, pid: int):
    LAST.update(hwnd=getattr(w, "handle", None), pid=pid, title=title, w=w)


def _front_is_target() -> bool:
    if LAST["hwnd"] and _foreground_hwnd() == LAST["hwnd"]:
        return True
    return bool(LAST["pid"]) and _foreground_pid() == LAST["pid"]


def _focus_target():
    """Bring the current app window to the front, or raise (so nothing is ever sent to the wrong window)."""
    if not LAST["pid"]:
        raise UIError("no app is selected yet: call ui_windows(focus='name') or ui_inspect first")
    if _front_is_target():
        return
    w = LAST["w"]
    try:
        if w.is_minimized():
            w.restore()
        w.set_focus()
    except Exception:
        pass
    for _ in range(12):
        time.sleep(0.1)
        if _front_is_target():
            return
    raise UIError(f"could not bring '{LAST['title']}' to the front (Windows may be blocking it, or the app runs as "
                  f"administrator). Click on it once yourself, then ask Jas to continue.")


def _target_ready():
    """Focus the current app and make sure Jas may operate it."""
    _focus_target()
    _check_pid(LAST["pid"], LAST["title"])


def focus_jas() -> bool:
    """Bring Jas's own terminal to the front (before an approval prompt, which would otherwise hide behind the app)."""
    try:
        mine = _protected_pids()
        if _foreground_pid() in mine:
            return True
        for w, title, pid in _windows():
            if pid in mine or TERMINAL_HINT.match(title or ""):
                if w.is_minimized():
                    w.restore()
                w.set_focus()
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------- windows
def _windows():
    pw, _ = _deps()
    out = []
    for w in pw.Desktop(backend="uia").windows():
        try:
            title = w.window_text()
            if title and w.is_visible():
                out.append((w, title, w.process_id()))
        except Exception:
            continue
    return out


def _find_window(query: str):
    q = (query or "").strip().lower()
    if not q:
        raise UIError("no window name given; call ui_windows to see the open windows")
    if q in DESKTOP or q == "desktop":
        raise UIError(f"'{query}' is the Windows desktop, not an app window; Jas cannot operate it. "
                      f"Call ui_windows to list the real app windows.")
    wins = [(w, t, p) for w, t, p in _windows() if not is_desktop(t, p)]
    for test in (lambda t, n: t.lower() == q, lambda t, n: t.lower().startswith(q),
                 lambda t, n: q in t.lower(), lambda t, n: q in n.lower()):
        hits = [(w, t, p) for w, t, p in wins if test(t, _proc_name(p))]
        if hits:
            return hits[0]
    names = ", ".join(f'"{t}"' for _, t, _ in wins[:12])
    raise UIError(f"no open window matches '{query}'. Open windows: {names or '(none)'}")


def _hwnd_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _caret_present() -> bool:
    """Best-effort: does the foreground window have a focused text field? (Start search has one; a plain Start
    menu, the desktop or an unfocused app do not.) Read-only: no hooks, no state change."""
    if sys.platform != "win32":
        return False
    try:
        class GUITHREADINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_ulong), ("flags", ctypes.c_ulong), ("hwndActive", ctypes.c_void_p),
                        ("hwndFocus", ctypes.c_void_p), ("hwndCapture", ctypes.c_void_p),
                        ("hwndMenuOwner", ctypes.c_void_p), ("hwndMoveSize", ctypes.c_void_p),
                        ("hwndCaret", ctypes.c_void_p), ("rcCaret", ctypes.c_long * 4)]
        user32 = ctypes.windll.user32
        info = GUITHREADINFO()
        info.cbSize = ctypes.sizeof(GUITHREADINFO)
        tid = user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), None)
        return bool(user32.GetGUIThreadInfo(tid, ctypes.byref(info)) and info.hwndCaret)
    except Exception:
        return False


START_TITLES = {"rechercher", "search", "start", "démarrer", "demarrer"}


def is_start_window(cls: str, proc: str, title: str = "") -> bool:
    """Does this window belong to the Start menu / its search box?

    Proven on the user's PC: the Start search is SearchHost.exe, class Windows.UI.Core.CoreWindow,
    title "Rechercher" (localised), and it reports IsWindowVisible = FALSE while being the foreground window.
    StartMenuExperienceHost (Win10 Start / Win11 panel) and ShellExperienceHost count too.
    Generic UWP windows must NOT: ApplicationFrameWindow is every Store app (Settings, Calculator, Photos)."""
    cls, proc, title = (cls or ""), (proc or "").lower(), (title or "").strip().lower()
    if proc == "searchhost.exe":
        return True
    if proc.startswith("startmenuexperiencehost"):
        return True
    if proc == "shellexperiencehost.exe":                 # Win10 search UI reports through this host
        return True
    return cls == "Windows.UI.Core.CoreWindow" and (proc == "" or title in START_TITLES)


def visible_windows(include_hidden: bool = True):
    """[(hwnd, class, proc, title)] for the top-level windows Windows reports.

    Handles stay 64-bit (wintypes.HWND). `include_hidden` defaults to True on purpose: the Start search window is
    composited by DWM and reports IsWindowVisible = FALSE, so the user's own diagnostic never listed it even while it
    was the FOREGROUND window (title "Rechercher", proc SearchHost.exe). Filtering on "visible" is therefore wrong
    for Start detection - use is_window_shown() when a real on-screen test is needed."""
    if sys.platform != "win32":
        return []
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    out = []
    PFN = getattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE)(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def cb(hwnd, _l):
        try:
            if not include_hidden and not user32.IsWindowVisible(hwnd):
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            title = ""
            n = user32.GetWindowTextLengthW(hwnd)
            if n > 0:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                title = buf.value
            out.append((hwnd, _hwnd_class(hwnd), _proc_name(pid.value), title))
        except Exception:
            pass
        return True
    user32.EnumWindows(PFN(cb), 0)
    return out


def is_window_shown(hwnd) -> bool:
    """Is the window really on screen? True when IsWindowVisible, or when DWM reports it cloaked-free but the
    classic bit is off (Start's composited search window)."""
    try:
        user32 = ctypes.windll.user32
        if user32.IsWindowVisible(hwnd):
            return True
        cloaked = ctypes.c_int(0)
        if ctypes.windll.dwmapi.DwmGetWindowAttribute(ctypes.c_void_p(hwnd), 14, ctypes.byref(cloaked),
                                                      ctypes.sizeof(cloaked)) == 0:
            return cloaked.value == 0                      # DWMWA_CLOAKED = 14; 0 = not cloaked, so it is drawn
    except Exception:
        pass
    return False


def _start_menu_open() -> bool:
    """Is the Start menu / its search box on screen? Uses the same window walk as the self-check, and treats
    'unknown process name' as unknown rather than 'not Start'. Raises UIError only when Windows itself cannot be
    enumerated, so a broken probe can never masquerade as 'the menu is closed'."""
    if sys.platform != "win32":
        return False
    try:
        wins = visible_windows()                           # includes DWM-composited windows on purpose
    except Exception as e:
        raise UIError(f"could not enumerate windows ({type(e).__name__}: {e})")
    for hwnd, cls, proc, title in wins:
        low = (proc or "").lower()
        if is_start_window(cls, low, title):
            return True
    if _foreground_is_start():
        return True                                        # seen only as the foreground window (user's diagnostic)
    return _caret_present()


# Windows we know are on screen right now (fed by the last Start attempt). The foreground window is composited and
# often NOT returned by EnumWindows, so a handle we already saw counts as Start.
KNOWN_START = {"hwnds": set()}


def _register_start_windows():
    """Remember every Start window currently enumerable, so the focus check can recognise them later."""
    try:
        for hwnd, cls, proc, title in visible_windows():
            if is_start_window(cls, proc, title):
                KNOWN_START["hwnds"].add(hwnd)
    except Exception:
        pass
    return KNOWN_START["hwnds"]


def _fg_proc_quick() -> str:
    try:
        return (_proc_name(_foreground_pid()) or "").lower()
    except Exception:
        return ""


def _foreground_is_start() -> bool:
    """Does the Start menu have the keyboard? True when the foreground window is SearchHost / that family, when it is
    a Start window we registered a moment ago, or when the foreground process is SearchHost."""
    if sys.platform != "win32":
        return False
    try:
        fg = ctypes.windll.user32.GetForegroundWindow()
        if fg and fg in KNOWN_START["hwnds"]:
            return True
        for hwnd, cls, proc, title in visible_windows():
            if hwnd == fg:
                return is_start_window(cls, proc, title)
        return _fg_proc_quick() == "searchhost.exe"
    except Exception:
        return False


def _window_rect(hwnd):
    """(left, top, right, bottom) of a window, or None."""
    try:
        from ctypes import wintypes
        r = wintypes.RECT()
        if ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(r)):
            return r.left, r.top, r.right, r.bottom
    except Exception:
        pass
    return None


def _start_has_focus() -> bool:
    """Is the Start menu the foreground window (or holding the text caret)? Only then is typing safe."""
    if sys.platform != "win32":
        return False
    try:
        fg = ctypes.windll.user32.GetForegroundWindow()
        for hwnd, cls, proc, _ in visible_windows():
            if hwnd == fg and is_start_window(cls, (proc or "").lower()):
                return True
        return _caret_present()
    except Exception:
        return False


def focus_start() -> bool:
    """Bring the open Start menu to the front, so typing lands in it. Only touches Start/SearchHost windows."""
    if sys.platform != "win32":
        return False
    if _start_has_focus():
        return True
    user32 = ctypes.windll.user32
    for hwnd, cls, proc, _ in visible_windows():
        if is_start_window(cls, (proc or "").lower()):
            try:
                if user32.IsIconic(hwnd):
                    user32.ShowWindow(hwnd, 9)               # SW_RESTORE
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.3)
                if _start_has_focus():
                    return True
            except Exception:
                continue
    return _start_has_focus()


def _elevated_foreground() -> str:
    """Name of the foreground process when it runs elevated, else ''. UIPI: a normal process's synthetic keys are
    silently dropped while an elevated window has the focus - the usual reason 'the Win key does nothing'."""
    if sys.platform != "win32":
        return ""
    try:
        import psutil
        from ctypes import wintypes
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
        if not pid.value or ctypes.windll.shell32.IsUserAnAdmin():
            return ""                                      # nothing focused, or we are elevated ourselves
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
        if not h:
            return f"pid {pid.value}"                      # access denied: elevated, and we cannot even name it
        try:
            return ""                                      # we can query it, so it runs at our level
        finally:
            kernel32.CloseHandle(h)
    except Exception:
        return ""

# Only the Left Windows key is ever sent. VK_APPS (0x5D) is the CONTEXT-MENU key - it pops up the right-click
# menu instead of Start - so it must never be used here.
KEY_WIN = 0x5B                      # VK_LWIN
KEYUP = 0x0002


def _send_key(vk: int):
    """Press and release one key with keybd_event (SendInput underneath)."""
    if vk not in (KEY_WIN,):        # whitelist: nothing else, least of all the context-menu key
        raise UIError(f"refusing to send key {vk:#x}")
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.05)
    user32.keybd_event(vk, 0, KEYUP, 0)


def _shell_execute(args: str) -> bool:
    """Ask the Windows shell to open something (Start included). Runs as the shell's own command: no keystrokes."""
    try:
        r = ctypes.windll.shell32.ShellExecuteW(None, "open", "explorer.exe", args, None, 5)   # 5 = SW_SHOW
        return r > 32
    except Exception:
        return False


def _open_start(uri: bool = True) -> bool:
    """Open the Start menu with ONE Win press and verify it.

    Returns True only when Start can be confirmed (window seen, or the keyboard ended up in it). When it cannot be
    confirmed the menu is LEFT OPEN - never pressed twice: the Win key toggles, and a second press is exactly what
    made Start flash open and closed without ever being searched."""
    if uri and sys.platform == "win32":
        if _shell_execute("ms-search:"):                  # Windows Search UI (Win11: Start with search)
            for _ in range(12):
                time.sleep(0.1)
                _register_start_windows()
                if _foreground_is_start() or _start_menu_open():
                    return True
    blocked = _elevated_foreground()
    if blocked:
        raise UIError(f"the foreground app ({blocked}) runs as administrator, so Windows ignores keys sent by a "
                      f"non-elevated program. Nothing was typed. Start Jas as administrator, or click the app "
                      f"yourself and use the mouse route.")
    _send_key(KEY_WIN)                                     # exactly one press
    for _ in range(15):                                    # up to 1.5 s to appear
        time.sleep(0.1)
        _register_start_windows()
        if _foreground_is_start() or _start_menu_open():
            return True
    return False                                           # leave it open; the caller decides what to do


# --- the Start menu's own app list: no keystrokes at all, read through the shell's AppsFolder ---
APPS_CACHE = {"t": 0.0, "items": []}
SHORTCUTS_CACHE = {"t": 0.0, "items": []}
CLSID_SHELL = "{20D04FE0-3AEA-1069-A2D8-08002B30309D}"     # This PC
APPS_HRESULT = 0x80070005                                   # access denied


def list_start_apps(refresh: bool = False):
    """[(name, AppUserModelId)] from the Start menu's All apps list (shell:AppsFolder).

    Pure PowerShell + Shell.Application COM, launched without a console. Read-only. Cached for 10 minutes."""
    if sys.platform != "win32":
        raise UIError("listing Start apps only works on Windows")
    if not refresh and APPS_CACHE["items"] and time.time() - APPS_CACHE["t"] < 600:
        return APPS_CACHE["items"]
    script = ("$ErrorActionPreference='Stop';"
              "$s=New-Object -ComObject Shell.Application;"
              "$f=$s.NameSpace('shell:AppsFolder');"
              "if($f -eq $null){exit 3};"
              "$f.Items()|ForEach-Object{ if($_.Name -and $_.Path){ '{0}`t{1}' -f $_.Name,$_.Path } }")
    p, attempts, outcome = None, [0x08000000, 0, 0x01000000], ""   # no console, plain, then no profile flag
    last = None
    for flags in attempts:
        try:
            p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=60, creationflags=flags)
            last = p
            if (p.stdout or "").strip():
                break                                      # got data: stop retrying
        except FileNotFoundError:
            raise UIError("PowerShell was not found on PATH")
        except subprocess.TimeoutExpired:
            outcome = "timed out after 60 s"
        except Exception as e:
            outcome = f"{type(e).__name__}: {e}"
    if p is None:
        raise UIError(f"could not run PowerShell to read the Start app list ({outcome})")
    items = []
    for line in (p.stdout or "").splitlines():
        if "\t" in line:
            name, _, appid = line.partition("\t")
            if name.strip() and appid.strip():
                items.append((name.strip(), appid.strip()))
    if not items:
        err = " / ".join(((p.stderr or "") + (p.stdout or "")).strip().splitlines()[-2:])[:300]
        hint = (" PowerShell itself ran, so most likely the shell COM connection to the AppsFolder is blocked "
                "(Explorer not running, or a policy) - the .lnk shortcuts are searched independently.")
        raise UIError("the Start app list came back empty (exit %s%s)"
                      % (p.returncode, f", said: {err}" if err else ", no output") + hint)
    APPS_CACHE["items"], APPS_CACHE["t"] = items, time.time()
    return items


def list_start_shortcuts(refresh: bool = False):
    """[(name, .lnk path)] from the Start menu's own shortcuts - the same source Windows' app index uses.

    Covers desktop programs that the AppsFolder list misses (ported games, portable tools, custom installs).
    Read-only, cached for 10 minutes."""
    if sys.platform != "win32":
        raise UIError("listing Start shortcuts only works on Windows")
    if not refresh and SHORTCUTS_CACHE["items"] and time.time() - SHORTCUTS_CACHE["t"] < 600:
        return SHORTCUTS_CACHE["items"]
    script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "$d=@( \"$env:APPDATA\\Microsoft\\Windows\\Start Menu\\Programs\", "
        "\"$env:ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\" );"
        "foreach($p in $d){ Get-ChildItem -LiteralPath $p -Filter *.lnk -Recurse -ErrorAction SilentlyContinue } | "
        "ForEach-Object { '{0}`t{1}' -f $_.BaseName, $_.FullName }"
    )
    p = None
    for flags in (0x08000000, 0):
        try:
            p = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                               capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=60, creationflags=flags)
            break
        except Exception:
            continue
    if p is None:
        raise UIError("could not run PowerShell to read the Start shortcuts")
    items = []
    for line in (p.stdout or "").splitlines():
        if "\t" in line:
            name, _, path = line.partition("\t")
            if name.strip() and path.strip().lower().endswith(".lnk"):
                items.append((name.strip(), path.strip()))
    if not items:
        err = " / ".join((p.stderr or "").strip().splitlines()[-2:])[:200]
        raise UIError("no Start-menu shortcuts were found" + (f" ({err})" if err else ""))
    SHORTCUTS_CACHE["items"], SHORTCUTS_CACHE["t"] = items, time.time()
    return items


def _score(name: str, query: str):
    """100 exact, 80 all words, 50 any word / substring, 0..40 fuzzy. None when it does not match."""
    q = " ".join((query or "").lower().split())
    if not q:
        return None
    low = " ".join((name or "").lower().split())
    want = [w for w in q.split() if w]
    if low == q:
        return 100
    if all(w in low for w in want):
        return 80 + (20 if low.startswith(q) else 0)
    if any(w in low for w in want) or q in low:
        return 50
    ratio = difflib.SequenceMatcher(None, low, q).ratio()
    return int(ratio * 40) if ratio >= 0.6 else None


def find_start_shortcut(query: str, k: int = 8):
    """Search the Start menu's .lnk shortcuts by name (Exact > all words > any word > fuzzy)."""
    out = []
    for name, path in list_start_shortcuts():
        score = _score(name, query)
        if score:
            out.append({"name": name, "path": path, "score": score})
    out.sort(key=lambda r: (-r["score"], len(r["name"]), r["name"].lower()))
    return out[:max(1, int(k))]


def launch_shortcut(path: str) -> str:
    """.lnk files are executed by Explorer: os.startfile hands it to the shell, which starts the target."""
    try:
        os.startfile(path)                                  # type: ignore[attr-defined]
        return "shortcut"
    except OSError as e:
        raise UIError(f"could not start the shortcut {path} ({e})")


def find_start_app(query: str, k: int = 8):
    """Search the Start app list by name. Exact > all words > any word > substring, then fuzzy."""
    q = " ".join((query or "").lower().split())
    if not q:
        return []
    rows, want = list_start_apps(), [w for w in q.split() if w]
    scored = []
    for name, appid in rows:
        low = " ".join(name.lower().split())
        if low == q:
            score = 100
        elif all(w in low for w in want):
            score = 80 + (20 if low.startswith(q) else 0)
        elif any(w in low for w in want) or q in low:
            score = 50
        else:
            ratio = difflib.SequenceMatcher(None, low, q).ratio()
            score = int(ratio * 40) if ratio >= 0.6 else 0
        if score:
            scored.append((score, name, appid))
    scored.sort(key=lambda r: (-r[0], len(r[1]), r[1].lower()))
    out, seen = [], set()
    for score, name, appid in scored:
        if name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "appid": appid, "score": score})
    return out[:max(1, int(k))]


def launch_start_app(appid: str) -> str:
    """Launch an app directly from its AppUserModelId: no Start menu, no keystrokes. Returns the method used."""
    if not appid:
        raise UIError("no AppUserModelId given")
    if re.match(r"^[a-z][a-z0-9+.-]*:", appid) and not appid.lower().startswith(("http", "microsoft")):
        try:
            os.startfile(appid)                            # the app's own protocol (Spotify and friends)
            return "URI"
        except OSError:
            pass
    pid_before = _snapshot()
    try:
        # explorer.exe hands the item to the shell's AppsFolder handler, which activates the packaged app itself
        p = subprocess.run(["explorer.exe", "shell:AppsFolder\\" + appid], capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        raise UIError("explorer.exe was not found")
    except subprocess.TimeoutExpired:
        raise UIError("launching the app timed out")
    if p.returncode not in (0, 1):                         # explorer.exe returns 1 even when it works
        raise UIError(f"could not activate '{appid}' (explorer exit {p.returncode})")
    for _ in range(12):                                    # verify: did a new window appear?
        time.sleep(0.5)
        now = _snapshot()
        if pid_before is not None and now is not None and set(now) - set(pid_before):
            return "AppsFolder"
    return "AppsFolder (no new window seen yet)"




BROWSERS = {"chrome.exe", "msedge.exe", "firefox.exe", "vivaldi.exe", "brave.exe", "opera.exe", "operagx.exe",
            "librewolf.exe"}


def _snapshot():
    """{pid: title} for every visible window, or None when the list cannot be read."""
    try:
        out = {}
        for _, t, p in _windows():
            out.setdefault(p, t)
        return out
    except Exception:
        return None


def _fg_proc() -> str:
    try:
        return (_proc_name(_foreground_pid()) or "").lower()
    except Exception:
        return "?"


def start_menu_search(app_name: str, enter: bool = True):
    """Type into the Windows Start search, optionally press Enter, and VERIFY what happened.

    Returns: typed, pressed_enter, start_open, new_windows [(pid, title)], browser.
    Never claims success: describe_start_result turns this into honest text.

    Guards: never types while a terminal or an elevated window has the keyboard (keystrokes there could answer
    Jas's own prompts); refuses names mentioning Jas. Start cannot be relied on as a detectable precondition - it is
    a transient flyout - so this path is the LAST resort, after start_app()."""
    name = (app_name or "").strip()
    if not name:
        raise UIError("no app name given")
    if len(name) > 60 or re.search(r"[\r\n\t]", name):
        raise UIError("that does not look like an app name")
    if re.search(r"(?i)\bjas\b", name):
        raise UIError("refusing to search Start for Jas's own name (that could focus the agent itself)")
    _deps()
    if not _foreground_is_start():
        # ONE Win press. Whether or not detection succeeds, the search box now has the keyboard, so type next:
        # checking again here is what previously closed the menu (a second Win press toggles it shut).
        _open_start()
        _register_start_windows()
        time.sleep(0.4)
    # Refuse only when the keys would land somewhere dangerous and Start is demonstrably NOT focused.
    if not _foreground_is_start():
        fg_proc = _fg_proc()
        if fg_proc in TERMINALS or "terminal" in fg_proc or "powershell" in fg_proc or "cmd" in fg_proc:
            raise UIError("the keyboard focus is on a terminal (" + fg_proc + ") and the Start search could not be "
                          "confirmed. Nothing was typed: keystrokes there could answer Jas's own prompts. Click the "
                          "Start search box yourself and ask again, or let Jas start the app from the Start catalogues.")
        blocked = _elevated_foreground()
        if blocked:
            raise UIError(f"the window with the keyboard ({blocked}) runs as administrator, so Windows would ignore "
                          f"the keys. Nothing was typed. Run Jas as administrator, or use the Start app list.")
    from pywinauto import keyboard
    keyboard.send_keys(escape_keys(name), with_spaces=True, pause=0.02)
    time.sleep(1.8)                                       # result list needs a moment
    res = {"typed": name, "pressed_enter": bool(enter), "start_open": True, "new_windows": [], "browser": ""}
    if enter:
        before = _snapshot()
        fg_before = _fg_proc()
        keyboard.send_keys("{ENTER}")
        for _ in range(12):                               # up to 6 s for something to appear
            time.sleep(0.5)
            now = _snapshot()
            if now is not None and before is not None:
                res["new_windows"] = sorted((p, t) for p, t in now.items() if p not in before)
                if res["new_windows"]:
                    break
            fg = _fg_proc()
            if fg in BROWSERS and fg != fg_before:
                res["browser"] = fg
                break
        if not res["browser"]:
            for p, _t in res["new_windows"]:
                proc = (_proc_name(p) or "").lower()
                if proc in BROWSERS:
                    res["browser"] = proc
                    break
    try:
        res["start_open"] = _start_menu_open()
    except Exception:
        res["start_open"] = False
    return res


def start_app(name: str, enter: bool = True):
    """Start an app WITHOUT any keystrokes: check the Start menu's app list, then its .lnk shortcuts.

    Both are the same catalogues the Start menu itself shows, so this works even when the menu refuses to open
    or Windows blocks synthetic keys. Returns the shape describe_start_result() understands, plus 'method'."""
    _deps()
    reasons = []
    hits, source = [], ""
    try:
        hits = [h for h in find_start_app(name, k=6) if h["score"] >= 50]
        source = "Start app list"
    except Exception as e:
        reasons.append(f"app list: {e}")
    if not hits:
        try:
            hits = [h for h in find_start_shortcut(name, k=6) if h["score"] >= 50]
            source = "Start menu shortcuts"
        except Exception as e:
            reasons.append(f"shortcuts: {e}")
    if not hits:
        raise UIError(f'"{name}" is not in the Start menu app list or its shortcuts'
                      + (f" ({'; '.join(reasons)})" if reasons else "")
                      + ". It may not be installed. Nothing was launched.")
    best = hits[0]
    before = _snapshot()
    method = launch_start_app(best["appid"]) if best.get("appid") else launch_shortcut(best["path"])
    new, browser = [], ""
    for _ in range(12):
        time.sleep(0.5)
        now = _snapshot()
        if now is not None and before is not None:
            new = sorted((p, t) for p, t in now.items() if p not in before)
        if new:
            for p, _t in new:
                proc = (_proc_name(p) or "").lower()
                if proc in BROWSERS:
                    browser = proc
            break
    return {"typed": name, "pressed_enter": True, "start_open": False, "new_windows": new,
            "browser": browser, "method": f"{method} ({source})", "app": best["name"],
            "alternatives": [h["name"] for h in hits[1:4]]}


def describe_start_result(r: dict) -> str:
    """Honest description of what a Start-menu launch actually did, for the model and the user."""
    name = r.get("typed", "")
    if r.get("app"):                                      # start_app: launched from the app list, not by keystrokes
        extra = f' (other matches: {", ".join(r["alternatives"])})' if r.get("alternatives") else ""
        if r.get("new_windows"):
            pid, title = r["new_windows"][0]
            return (f'Launched "{r["app"]}"{extra}: a new window appeared, "{title[:70]}" ({_proc_name(pid)}), '
                    f'via {r.get("method", "the shell")}. Next: ui_inspect that window.')
        return (f'Started "{r["app"]}"{extra} via {r.get("method", "the shell")}, but no new window appeared within '
                f'6 s. It may still be starting, or it may have opened in the tray. Check ui_windows once before '
                f'concluding anything.')
    if not r.get("pressed_enter"):
        return (f'Typed "{name}" into the Start menu search; results are showing. Read them with ui_read_screen, then '
                f'ui_click x,y on the right result (the first one is not always the app).')
    new = r.get("new_windows") or []
    if new and not r.get("browser"):                      # a real app window (not a browser tab opened by Enter)
        pid, title = new[0]
        extra = f" ({len(new)} new windows)" if len(new) > 1 else ""
        return (f'Launched it: a new window appeared, "{title[:70]}" ({_proc_name(pid)}){extra}. Next: ui_inspect '
                f'that window.')
    if r.get("browser"):
        return (f'NOT launched as an app: pressing Enter opened a WEB result in {r["browser"]}, so "{name}" does not '
                f'look installed on this PC. Do not retry with other spellings: tell the user, or open its website if '
                f'that is what they want.')
    if r.get("start_open"):
        return (f'NOT launched: the Start menu is still open, so the search for "{name}" matched no app to open. It is '
                f'probably not installed. Do not retry with other spellings; if the results do show a matching app, '
                f'ui_read_screen then ui_click x,y on it; otherwise tell the user.')
    return (f'NOT confirmed: Enter closed the Start menu but no new window appeared for "{name}". It is probably not '
            f'installed, or still starting. Check ui_windows once; if there is no matching app, tell the user instead '
            f'of trying more spellings.')


@guard
def ui_windows(focus: str = ""):
    """List open windows; with focus='name', bring that window to the front."""
    wins = _windows()
    if focus:
        w, title, pid = _find_window(focus)
        _check_pid(pid, title)
        _remember(w, title, pid)
        _failsafe()
        try:
            if w.is_minimized():
                w.restore()
            w.set_focus()
        except Exception as e:
            raise UIError(f"could not bring '{title}' to the front: {type(e).__name__}: {str(e)[:120]}")
        time.sleep(0.4)
        return f'Focused "{title}" ({_proc_name(pid)}). Next: ui_inspect("{title}").'
    me = _protected_pids()
    flag = bool(KNOWN)                          # no baseline yet: everything would be "just opened", so don't say it
    wins = [(w, t, p) for w, t, p in wins if not is_desktop(t, p)]
    lines = [f'- "{t[:80]}" ({_proc_name(p)})'
             + ("  [terminal: off limits]" if p in me or _proc_name(p).lower() in TERMINALS else "")
             + ("  [just opened]" if flag and p not in KNOWN else "")
             for _, t, p in wins]
    KNOWN.update(p for _, _, p in wins)
    if not lines:
        return "No app windows are open right now. To start an app, use open (e.g. open('spotify'))."
    return ("Open windows (newest are often what you just launched; '[just opened]' appeared since the last "
            "ui_windows call):\n" + "\n".join(lines))


@guard
def ui_start_search(name: str, enter: bool = True):
    """Launch an app through the Windows Start menu search and verify that it actually opened."""
    return describe_start_result(start_menu_search(name, bool(enter)))


@guard
def ui_start_app(name: str):
    """Launch an app straight from the Start menu's app list (no keystrokes): find it by name, then activate it."""
    try:
        return describe_start_result(start_app(name))
    except UIError as e:
        if "not in the Start menu app list" in str(e):
            raise
        # the list itself could not be read: fall back to the keystroke route
        return describe_start_result(start_menu_search(name, True))


@guard
def ui_inspect(window: str, filter: str = "", wait: float = 0, budget: int = 5000):
    """List a window's visible controls as numbered elements (valid until the next ui_inspect)."""
    w, title, pid = _find_window(window)
    _check_pid(pid, title)
    _remember(w, title, pid)
    _failsafe()
    if wait:
        time.sleep(min(float(wait), 10.0))
    f = (filter or "").lower()
    items, rows, truncated = {}, [], 0
    queue, t0, seen, n = deque([w.element_info]), time.time(), 0, 0
    header = f'Window "{title}" ({_proc_name(pid)}):'
    used = len(header)
    while queue and seen < 2500 and time.time() - t0 < 6.0:
        info = queue.popleft()
        seen += 1
        try:
            ctype, name = info.control_type, (info.name or "").strip()
            visible = info.visible
            children = info.children()
        except Exception:
            continue
        queue.extend(children)
        if not visible or ctype not in CONTROL_TYPES:
            continue
        auto_id = ""
        try:
            auto_id = info.automation_id or ""
        except Exception:
            pass
        if f and f not in name.lower() and f not in auto_id.lower():
            continue
        if not f and ctype == "Text" and len(name) < 2:
            continue
        if not name and not auto_id and ctype not in ("Edit", "Document", "ComboBox"):
            continue
        value = _doc_text(info) if ctype in ("Edit", "Document", "ComboBox") else ""
        enabled = True
        try:
            enabled = bool(info.enabled)
        except Exception:
            pass
        n += 1
        line = fmt_element(n, ctype, name, auto_id, "" if is_sensitive(name + auto_id) else value, enabled)
        if used + len(line) + 1 > budget:
            truncated += 1
            n -= 1
            continue
        items[n] = info
        rows.append(line)
        used += len(line) + 1
    ELEMS["pid"], ELEMS["items"] = pid, items
    KNOWN.add(pid)
    if not rows:
        return (header + "\n(no controls found" + (f" matching '{filter}'" if filter else "") +
                "). This app may not expose its interface; try ui_read_screen instead.")
    content = any(
        (t.control_type in ("Text", "ListItem", "TreeItem", "DataItem", "Hyperlink")
         and len((t.name or "").strip()) >= 2)
        or (t.control_type in ("Edit", "Document", "ComboBox") and _has_text_content(t))
        for t in items.values())
    tail = f"\n...{truncated} more not shown; use filter='text' to find a specific control." if truncated else ""
    if not content:
        return (header + "\n" + "\n".join(rows) + tail +
                "\nNOTE: these are window controls only (title bar, menus) - NO page or list content was read, so "
                "you know NOTHING about what this app displays. Do not describe or summarise its content. Use "
                "ui_read_screen (or inspect a child window) to read the actual text first.")
    return header + "\n" + "\n".join(rows) + tail


UI_TEXT_RX = re.compile(r"(?i)text(document|pattern)|value\.pattern|document")


def _doc_text(info, limit: int = 6000) -> str:
    """The text of a Document/Edit control (TextPattern), even when the app reports no ValuePattern.

    Apps like BrickLog show their content in a Document (a web view / custom control): get_value() returns "" and
    the actual list is only in the TextPattern. Also used for a window whose ONLY content is a document."""
    try:
        from pywinauto.controls.uiawrapper import UIAWrapper
        w = UIAWrapper(info)
    except Exception:
        return ""
    for getter in (lambda: w.get_value(),
                   lambda: w.iface_text.DocumentText(),
                   lambda: " ".join(w.texts())):
        try:
            text = getter()
        except Exception:
            continue
        text = " ".join(str(text or "").split())
        if text:
            return text[:limit]
    return ""


def _has_text_content(info) -> bool:
    """Does this control actually contain readable text? (a Document named 'My app' does not count)"""
    return len(_doc_text(info, limit=200)) >= 2


def _element(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise UIError("element must be a number from the latest ui_inspect")
    info = ELEMS["items"].get(n)
    if info is None:
        if not ELEMS["items"]:
            raise UIError(f"there are no numbered elements right now (element {n} does not exist) because the "
                          f"latest ui_inspect found no controls. Do not click numbers: inspect the right window, "
                          f"or use ui_read_screen and click by x,y.")
        raise UIError(f"element {n} is not in the latest ui_inspect (numbers expire when you inspect again). Call ui_inspect.")
    _check_pid(ELEMS["pid"], "")
    return info


def _wrap(info):
    from pywinauto.controls.uiawrapper import UIAWrapper
    return UIAWrapper(info)


@guard
def ui_click(element: int = None, x: int = None, y: int = None, button: str = "left", double: bool = False):
    """Click an element (number from ui_inspect) or a screen position."""
    _, pg = _deps()
    button = button if button in ("left", "right", "middle") else "left"
    _failsafe()
    if element is not None:
        info = _element(element)
        name = info.name or ""
        if is_risky(name) and not APPROVE(f"Jas wants to CLICK the button '{name}'."):
            raise UIError("the user declined that click. Do not retry; ask what they would prefer.")
        w = _wrap(info)
        if not w.is_enabled():
            raise UIError(f'element "{name}" is disabled')
        _focus_target()
        try:
            r = info.rectangle
            if _pid_at((r.left + r.right) // 2, (r.top + r.bottom) // 2) in _protected_pids():
                raise UIError("the Jas terminal is covering that control; move the app or the terminal so it is visible")
        except UIError:
            raise
        except Exception:
            pass
        w.click_input(button=button, double=bool(double))
        what = f'"{name}"' if name else f"element {element}"
    elif x is not None and y is not None:
        pid = _pid_at(int(x), int(y))
        _check_pid(pid, "")
        if pid and not LAST["pid"] and pid not in _protected_pids():
            _remember(None, _proc_name(pid), pid)             # remember what was clicked, so ui_click(x,y) twice needs one OK
        pg.click(int(x), int(y), button=button, clicks=2 if double else 1)
        what = f"({x},{y})"
    else:
        raise UIError("give an element number, or both x and y")
    time.sleep(0.5)
    CLICKED["last"] = time.time()
    return f"Clicked {what}. Call ui_inspect again to see what changed."


@guard
def ui_type(text: str, element: int = None, clear: bool = False, submit: bool = False):
    """Type text into an element (or the focused field), optionally clearing it first and pressing Enter after."""
    pw, _ = _deps()
    from pywinauto import keyboard
    _failsafe()
    if element is not None:
        info = _element(element)
        name = info.name or ""
        if is_sensitive(name) or is_sensitive(getattr(info, "automation_id", "") or ""):
            raise UIError("that looks like a password/PIN/card field. Jas never types those; ask the user to type it.")
        try:
            if info.element.CurrentIsPassword:
                raise UIError("that is a password field. Jas never types into those; ask the user to type it.")
        except UIError:
            raise
        except Exception:
            pass
        w = _wrap(info)
        _focus_target()
        try:
            w.set_focus()
        except Exception:
            w.click_input()
        time.sleep(0.15)
    else:
        _target_ready()
    if clear:
        keyboard.send_keys("^a{BACKSPACE}", pause=0.02)
    keyboard.send_keys(escape_keys(text), with_spaces=True, with_tabs=True, with_newlines=True, pause=0.01)
    if submit:
        keyboard.send_keys("{ENTER}")
    time.sleep(0.4)
    return f"Typed {len(text)} characters" + (" and pressed Enter" if submit else "") + ". Call ui_inspect to verify."


@guard
def ui_keys(keys: str):
    """Press keys or shortcuts in the focused window, e.g. 'ctrl+s', 'enter', 'ctrl+a delete'."""
    _, pg = _deps()
    combos = parse_keys(keys)
    _failsafe()
    _target_ready()
    if any("win" in c for c in combos) and not APPROVE(f"Jas wants to press the Windows key combination '{keys}'."):
        raise UIError("the user declined that key combination.")
    for c in combos:
        pg.hotkey(*c) if len(c) > 1 else pg.press(c[0])
        time.sleep(0.15)
    time.sleep(0.4)
    return f"Pressed {keys}. Call ui_inspect to see the result."


@guard
def ui_scroll(direction: str = "down", amount: int = 5):
    """Scroll the focused window up or down."""
    _, pg = _deps()
    _failsafe()
    _target_ready()
    l, t, r, b = _foreground_rect()
    pg.moveTo((l + r) // 2, (t + b) // 2)
    pg.scroll(abs(int(amount)) * (120 if direction == "up" else -120))
    time.sleep(0.3)
    return f"Scrolled {direction} by {amount}. Call ui_inspect (or ui_read_screen) to see the result."


# ---------------------------------------------------------------- screen reading (OCR)
_OCR = None


def _ocr_engine():
    global _OCR
    if _OCR is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            try:
                from rapidocr import RapidOCR
            except ImportError:
                raise UIError("screen reading needs OCR: pip install rapidocr-onnxruntime")
        _OCR = RapidOCR()
    return _OCR


def _mask_protected(img, origin):
    """Black out the VISIBLE parts of Jas's own terminal, so the model never reads its own words (or your
    approval prompts) back from the screen. Windows are painted bottom to top, so an app in front of the
    terminal stays readable."""
    try:
        from PIL import Image
        mine = _protected_pids()
        mask = Image.new("L", img.size, 255)
        for w, title, pid in reversed(_windows()):            # EnumWindows order is top to bottom
            protected = pid in mine or bool(TERMINAL_HINT.match(title or "")) or _proc_name(pid).lower() in TERMINALS
            r = w.rectangle()
            box = (r.left - origin[0], r.top - origin[1], r.right - origin[0], r.bottom - origin[1])
            mask.paste(0 if protected else 255, box)
        img.paste((0, 0, 0), mask=mask.point(lambda v: 255 - v))
    except Exception:
        pass
    return img


def ocr_lines(image):
    """-> [(text, center_x, center_y)] for a PIL image (coordinates relative to the image)."""
    import numpy as np
    res = _ocr_engine()(np.array(image.convert("RGB")))
    out = []
    if isinstance(res, tuple):                       # rapidocr-onnxruntime: (list of [box, text, score], timing)
        triples = [(b, t, s) for b, t, s in (res[0] or [])]
    else:                                            # newer rapidocr: object with boxes/txts/scores
        boxes = getattr(res, "boxes", None)
        triples = list(zip(boxes if boxes is not None else [], getattr(res, "txts", None) or [],
                           getattr(res, "scores", None) or []))
    for box, text, score in triples:
        if not str(text).strip():
            continue
        xs, ys = [p[0] for p in box], [p[1] for p in box]
        out.append((str(text).strip(), int(sum(xs) / len(xs)), int(sum(ys) / len(ys))))
    out.sort(key=lambda r: (r[2] // 12, r[1]))        # reading order: rows top to bottom, then left to right
    return out


@guard
def ui_read_screen(window: str = "", budget: int = 5000):
    """Read the visible text of a window with OCR; returns text with x,y screen positions.
    Default: the app Jas is working on. window='screen' reads the whole screen (Jas's own terminal is blacked out)."""
    _, pg = _deps()
    want = (window or "").strip()
    whole = want.lower() in ("screen", "all", "everything") or (not want and not LAST["pid"])
    if whole:
        if not SCREEN["granted"]:
            if not APPROVE("Jas wants to READ the text on your whole screen (its own terminal is blacked out)."):
                raise UIError("the user did not allow reading the whole screen. Read a specific window instead.")
            SCREEN["granted"] = True
        region, label = None, "the whole screen"
    else:
        if want:
            w, title, pid = _find_window(want)
            _check_pid(pid, title)
            _remember(w, title, pid)
        elif not LAST["pid"]:                         # nothing inspected yet: read the whole screen instead
            if not SCREEN["granted"]:
                if not APPROVE("Jas wants to READ the text on your whole screen (its own terminal is blacked out)."):
                    raise UIError("the user did not allow reading the whole screen. Inspect a specific window instead.")
                SCREEN["granted"] = True
            region, label = None, "the whole screen"
        else:                                         # default: the app Jas is already working on
            w, title, pid = LAST["w"], LAST["title"], LAST["pid"]
            _check_pid(pid, title)
        _focus_target()
        r = w.rectangle()
        region, label = (r.left, r.top, r.width(), r.height()), f'window "{title}"'
    _failsafe()
    img = pg.screenshot(region=region) if region else pg.screenshot()
    ox, oy = (region[0], region[1]) if region else (0, 0)
    img = _mask_protected(img, (ox, oy))
    try:                                                  # did the protected-window mask black out the whole capture?
        import numpy as np
        if float(np.asarray(img.convert("L")).mean()) < 3:
            return (f"ERROR: the whole capture of {label} was blacked out by the network-terminal mask, so nothing "
                    f"could be read. This happens when the window has no size or is fully covered. Nothing is known "
                    f"about its content: do not guess it.")
    except Exception:
        pass
    lines = ocr_lines(img)
    if not lines:
        return (f"No readable text found on {label} - so its content is unknown. Do not guess what it shows; "
                f"say you could not read it.\nSCREEN SUMMARY (empty): nothing was read, so you know nothing about "
                f"this window's content.")
    out, used = [f"Text on {label} (positions are screen coordinates for ui_click x,y):"], 0
    shown = 0
    for text, cx, cy in lines:
        row = f'"{text[:80]}" @({cx + ox},{cy + oy})'
        if used + len(row) > budget:
            out.append("...more text not shown; pass window= to narrow it down.")
            break
        out.append(row)
        used += len(row) + 1
        shown += 1
    words = sum(len(t.split()) for t, _, _ in lines)
    if shown == 0:
        kind = "empty"
    elif (shown <= 4 or words <= 12) and not CLICKED["last"]:
        kind = "unexplored page"       # a sparse screen nobody navigated to: the app's home page, not its data
    elif words <= 12:
        kind = "few words"
    else:
        kind = ""                      # a real, content-bearing screen: report it as read
    if kind:
        out.append(f"SCREEN SUMMARY ({kind}): this screen shows only {shown} short label(s). It is a landing/start "
                   f"page, NOT the data the user asked about. You have NOT explored this app. To find real content, "
                   f"ACT: click the section that holds it with ui_click x,y (positions above), then read again - "
                   f"or tell the user you could only see this page. Never present these labels as a list of items.")
    return "\n".join(out)