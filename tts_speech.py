"""Text-to-speech helpers: gTTS bytes + a browser Web Speech API widget.

`text_to_speech` produces an MP3 byte-stream via gTTS (requires internet
access to Google Translate TTS endpoints) and returns None when offline or
on any failure so callers can degrade gracefully.
`speech_button_html` returns a self-contained HTML/JS widget that uses the
browser's built-in SpeechSynthesis API (works fully offline) with
Play / Pause / Resume / Stop / Mute / Restart and speed controls.
"""
from __future__ import annotations

import io
from typing import Optional


def text_to_speech(text: str, lang: str = "en",
                   filename: Optional[str] = None) -> Optional[bytes]:
    """Convert `text` to MP3 bytes (or write to `filename`).

    Returns bytes when successful and on-disk filename is not requested;
    returns None on any failure (offline, no credentials, gTTS error).
    """
    if not text or not str(text).strip():
        return None
    try:
        from gtts import gTTS
    except Exception:
        return None

    try:
        tts = gTTS(text=str(text), lang=lang, slow=False)
        if filename:
            tts.save(filename)
            return None
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Browser Web Speech API widget
# ---------------------------------------------------------------------------

_BTN_PRIMARY = "background:#2e8bff;color:#fff;border:none;border-radius:8px;padding:8px 12px;font-size:13px;cursor:pointer;"
_BTN_SECONDARY = "background:#112240;color:#e6f1ff;border:1px solid #2e8bff66;border-radius:8px;padding:8px 12px;font-size:13px;cursor:pointer;"
_BTN_DANGER = "background:#ff6b6b;color:#fff;border:none;border-radius:8px;padding:8px 12px;font-size:13px;cursor:pointer;"


def speech_button_html(text: str, widget_id: str = "airobot") -> str:
    """Return HTML/JS for a self-contained SpeechSynthesis player.

    Controls: Play, Pause, Resume, Stop, Mute, Restart, Speed (0.5x-2x).
    Uses the browser's built-in voices so no API key or internet is needed.
    """
    safe_id = "".join(c if c.isalnum() or c == "_" else "_" for c in widget_id)
    esc = str(text).replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")

    html = """
<div id="%(sid)s-wrap" style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#112240;border:1px solid #2e8bff55;border-radius:14px;padding:14px 16px;margin:6px 0;">
  <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
    <button onclick="%(sid)s_speak()"   style="%(_btn_primary)s">&#9654; PLAY AUDIO</button>
    <button onclick="%(sid)s_pause()"   style="%(_btn_secondary)s">&#10074;&#10074; PAUSE</button>
    <button onclick="%(sid)s_resume()"  style="%(_btn_secondary)s">&#9654; RESUME</button>
    <button onclick="%(sid)s_stop()"    style="%(_btn_danger)s">&#9632; STOP</button>
    <button onclick="%(sid)s_mute()"    style="%(_btn_secondary)s">&#128263; MUTE</button>
    <button onclick="%(sid)s_restart()" style="%(_btn_secondary)s">&#8635; RESTART</button>
  </div>
  <div style="margin-top:8px;display:flex;gap:8px;align-items:center;">
    <span style="color:#8aa2c8;font-size:13px;">Speed</span>
    <select id="%(sid)s_rate" onchange="%(sid)s_rate_change()" style="background:#0a1628;color:#e6f1ff;border:1px solid #2e8bff66;border-radius:8px;padding:4px 8px;">
      <option value="0.5">0.5x</option>
      <option value="0.75">0.75x</option>
      <option value="1" selected>1x</option>
      <option value="1.25">1.25x</option>
      <option value="1.5">1.5x</option>
      <option value="2">2x</option>
    </select>
    <span id="%(sid)s_status" style="color:#34d399;font-size:12px;margin-left:6px;">ready</span>
  </div>
  <div style="margin-top:8px;height:4px;background:#0a1628;border-radius:2px;overflow:hidden;">
    <div id="%(sid)s_bar" style="height:100%%;width:0%%;background:#2e8bff;border-radius:2px;transition:width .3s;"></div>
  </div>
</div>
<script>
function %(sid)s_speak(){
  if(!('speechSynthesis' in window)){document.getElementById('%(sid)s_status').textContent='not supported';return;}
  var rate=parseFloat(document.getElementById('%(sid)s_rate').value||'1');
  speechSynthesis.cancel();
  window['%(sid)s_muted']=window['%(sid)s_muted']||false;
  var u=new SpeechSynthesisUtterance('%(esc)s');
  u.rate=rate; u.pitch=1; u.volume=window['%(sid)s_muted']?0:1;
  window['%(sid)s_utter']=u;
  u.onstart=function(){document.getElementById('%(sid)s_status').textContent=window['%(sid)s_muted']?'muted':'speaking';};
  u.onend=function(){document.getElementById('%(sid)s_status').textContent='done';document.getElementById('%(sid)s_bar').style.width='0%%';};
  u.onerror=function(){document.getElementById('%(sid)s_status').textContent='error';};
  speechSynthesis.speak(u);
  document.getElementById('%(sid)s_bar').style.width='100%%';
}
function %(sid)s_pause(){ if('speechSynthesis' in window){speechSynthesis.pause();document.getElementById('%(sid)s_status').textContent='paused';} }
function %(sid)s_resume(){ if('speechSynthesis' in window){speechSynthesis.resume();document.getElementById('%(sid)s_status').textContent=window['%(sid)s_muted']?'muted':'speaking';} }
function %(sid)s_stop(){ if('speechSynthesis' in window){speechSynthesis.cancel();document.getElementById('%(sid)s_status').textContent='stopped';document.getElementById('%(sid)s_bar').style.width='0%%';} }
function %(sid)s_mute(){
  window['%(sid)s_muted']=!window['%(sid)s_muted'];
  if('speechSynthesis' in window && window['%(sid)s_utter']){
    window['%(sid)s_utter'].volume=window['%(sid)s_muted']?0:1;
  }
  document.getElementById('%(sid)s_status').textContent=window['%(sid)s_muted']?'muted':'speaking';
}
function %(sid)s_restart(){ if('speechSynthesis' in window){speechSynthesis.cancel();%(sid)s_speak();} }
function %(sid)s_rate_change(){ if('speechSynthesis' in window && window['%(sid)s_utter']){speechSynthesis.cancel();%(sid)s_speak();} }
</script>
"""
    return html % {
        "sid": safe_id,
        "esc": esc,
        "_btn_primary": _BTN_PRIMARY,
        "_btn_secondary": _BTN_SECONDARY,
        "_btn_danger": _BTN_DANGER,
    }