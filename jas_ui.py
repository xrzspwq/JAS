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
import os
import re
import sys
import time
from collections import deque

APPROVE = lambda msg: False          # jas.py replaces this with its own y/N prompt
GRANTED = set()                      # process ids the user has allowed Jas to control (this session)
ELEMS = {"pid": None, "items": {}}   # element number -> UIA element, from the latest ui_inspect
LAST = {"hwnd": None, "pid": None, "title": "", "w": None}   # the app window Jas is currently working on
SCREEN = {"granted": False}                                  # whole-screen reading allowed this session
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
    "pageup", "pagedown", "up", "down", "left", "right", "capslock", "printscreen", "apps",
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
    try:
        import psutil
        return psutil.Process(pid).name()
    except Exception:
        return "?"


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
    wins = _windows()
    for test in (lambda t, n: t.lower() == q, lambda t, n: t.lower().startswith(q),
                 lambda t, n: q in t.lower(), lambda t, n: q in n.lower()):
        hits = [(w, t, p) for w, t, p in wins if test(t, _proc_name(p))]
        if hits:
            return hits[0]
    names = ", ".join(f'"{t}"' for _, t, _ in wins[:12])
    raise UIError(f"no open window matches '{query}'. Open windows: {names or '(none)'}")


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
    lines = [f'- "{t[:80]}" ({_proc_name(p)})' + ("  [terminal: off limits]" if p in me else "") for _, t, p in wins]
    return "Open windows:\n" + "\n".join(lines) if lines else "No visible windows found."


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
        value = ""
        if ctype in ("Edit", "Document", "ComboBox"):
            try:
                from pywinauto.controls.uiawrapper import UIAWrapper
                value = " ".join(str(UIAWrapper(info).get_value() or "").split())
            except Exception:
                value = ""
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
    if not rows:
        return (header + "\n(no controls found" + (f" matching '{filter}'" if filter else "") +
                "). This app may not expose its interface; try ui_read_screen instead.")
    tail = f"\n...{truncated} more not shown; use filter='text' to find a specific control." if truncated else ""
    return header + "\n" + "\n".join(rows) + tail


def _element(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        raise UIError("element must be a number from the latest ui_inspect")
    info = ELEMS["items"].get(n)
    if info is None:
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
        pg.click(int(x), int(y), button=button, clicks=2 if double else 1)
        what = f"({x},{y})"
    else:
        raise UIError("give an element number, or both x and y")
    time.sleep(0.5)
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
    lines = ocr_lines(img)
    if not lines:
        return f"No readable text found on {label}."
    out, used = [f"Text on {label} (positions are screen coordinates for ui_click x,y):"], 0
    for text, cx, cy in lines:
        row = f'"{text[:80]}" @({cx + ox},{cy + oy})'
        if used + len(row) > budget:
            out.append("...more text not shown; pass window= to narrow it down.")
            break
        out.append(row)
        used += len(row) + 1
    return "\n".join(out)