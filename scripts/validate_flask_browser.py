"""Headless Chrome/Edge integration checks against the actual local Flask app.

Uses the installed browser and optional requirements-web-test.txt. All screenshots,
downloads and browser profile files stay in a new ignored outputs/ directory.
The real prediction uses retained validation sample 686, never a simulated mask.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen

import websocket

ROOT = Path(__file__).resolve().parents[1]


class CDP:
    def __init__(self, address, origin):
        self.socket = websocket.create_connection(address, origin=origin, timeout=30)
        self.serial = 0
        self.errors = []

    def call(self, method, **params):
        self.serial += 1
        serial = self.serial
        self.socket.send(json.dumps({"id": serial, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("method") == "Runtime.exceptionThrown":
                self.errors.append("Uncaught browser JavaScript exception")
            if message.get("method") == "Log.entryAdded" and message["params"]["entry"].get("level") == "error":
                self.errors.append(message["params"]["entry"]["text"])
            if message.get("id") == serial:
                if "error" in message:
                    raise RuntimeError(message["error"])
                return message.get("result", {})

    def evaluate(self, expression, gesture=False):
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True,
                           awaitPromise=True, userGesture=gesture)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"]["text"])
        return result.get("result", {}).get("value")

    def wait(self, expression, timeout=120):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.evaluate(expression):
                return
            time.sleep(0.25)
        raise TimeoutError("Browser condition did not complete: " + expression)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:5000")
    parser.add_argument("--browser", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    browsers = [args.browser] if args.browser else [Path("C:/Program Files/Google/Chrome/Application/chrome.exe"), Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")]
    browser = next((p for p in browsers if p and p.is_file()), None)
    if not browser:
        parser.error("Supply an installed Chromium browser with --browser.")
    output = (args.output_dir or ROOT / "outputs/flask_browser" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    if not output.is_relative_to(ROOT / "outputs") or output.exists():
        parser.error("Use a new directory below outputs/.")
    output.mkdir(parents=True)
    downloads = output / "downloads"
    downloads.mkdir()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    command = [str(browser), "--headless=new", f"--remote-debugging-port={port}",
               "--remote-debugging-address=127.0.0.1", f"--remote-allow-origins={origin}",
               f"--user-data-dir={output / 'browser_profile'}", "--no-first-run", "--no-default-browser-check",
               "--disable-background-networking", "--disable-component-update", "--disable-sync",
               "--disable-extensions", "about:blank"]
    report = {"status": "running", "real_browser": True, "real_prediction": False}
    with (output / "browser.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        cdp = None
        try:
            end = time.monotonic() + 90
            targets = []
            while time.monotonic() < end:
                try:
                    with urlopen(origin + "/json/list", timeout=3) as response:
                        targets = json.load(response)
                    if any(item.get("type") == "page" for item in targets):
                        break
                except OSError:
                    time.sleep(0.25)
                if process.poll() is not None:
                    raise RuntimeError("Browser exited before local DevTools startup; inspect browser.log for OS/sandbox errors.")
            target = next((item for item in targets if item.get("type") == "page"), None)
            if target is None:
                raise RuntimeError("Browser DevTools did not expose a page before the startup timeout; inspect browser.log.")
            cdp = CDP(target["webSocketDebuggerUrl"], origin)
            for domain in ("Page", "Runtime", "Log", "DOM"):
                cdp.call(domain + ".enable")
            cdp.call("Browser.setDownloadBehavior", behavior="allow", downloadPath=str(downloads))
            cdp.call("Emulation.setDeviceMetricsOverride", width=1440, height=1100, deviceScaleFactor=1, mobile=False)
            cdp.call("Page.navigate", url=args.url)
            cdp.wait("document.readyState === 'complete' && document.getElementById('model-label').textContent !== 'Checking model'")

            def screenshot(filename):
                result = cdp.call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
                (output / filename).write_bytes(base64.b64decode(result["data"]))

            def no_overflow():
                assert cdp.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Horizontal overflow"

            no_overflow()
            screenshot("desktop_empty.png")
            for width, height, label in ((1024, 900, "laptop"), (768, 1024, "tablet"), (390, 844, "mobile")):
                cdp.call("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=width < 500)
                no_overflow()
                screenshot(label + "_empty.png")
            report["responsive_empty_widths"] = [1440, 1024, 768, 390]
            cdp.call("Emulation.setDeviceMetricsOverride", width=1440, height=1100, deviceScaleFactor=1, mobile=False)
            document = cdp.call("DOM.getDocument")
            node = cdp.call("DOM.querySelector", nodeId=document["root"]["nodeId"], selector="#image-input")["nodeId"]
            # Confirm browser-side invalid-file validation before any inference.
            invalid = output / "invalid.png"
            invalid.write_bytes(b"not an image")
            cdp.call("DOM.setFileInputFiles", files=[str(invalid)], nodeId=node)
            cdp.wait("!document.getElementById('error-message').hidden")
            assert cdp.evaluate("document.getElementById('run-button').disabled")
            report["invalid_upload_rejected"] = True
            record = json.loads((ROOT / "experiments/phase2_pilot/data_manifest.json").read_text())["validation"][686]
            image = ROOT / record["image"].replace("\\", "/")
            cdp.call("DOM.setFileInputFiles", files=[str(image)], nodeId=node)
            cdp.wait("!document.getElementById('input-preview').hidden")
            cdp.evaluate("window.__validationUpload = document.getElementById('image-input').files[0]; document.getElementById('remove-image').click(); const transfer = new DataTransfer(); transfer.items.add(window.__validationUpload); document.getElementById('drop-zone').dispatchEvent(new DragEvent('drop', {dataTransfer:transfer,bubbles:true,cancelable:true})); delete window.__validationUpload;")
            cdp.wait("!document.getElementById('input-preview').hidden")
            report["remove_replace_and_drop_upload"] = True
            cdp.evaluate("document.querySelector('[data-prompt]').click()")
            assert cdp.evaluate("document.getElementById('prompt').value.startsWith('Bilateral pulmonary infection')")
            report["example_prompt_control"] = True
            cdp.evaluate("document.getElementById('prompt').value = " + json.dumps(record["prompt"]) + "; document.getElementById('prompt').dispatchEvent(new Event('input'))")
            cdp.wait("!document.getElementById('run-button').disabled", timeout=1200)
            cdp.evaluate("document.getElementById('run-button').click()", gesture=True)
            cdp.wait("!document.getElementById('result-content').hidden || (!document.getElementById('error-message').hidden && document.getElementById('run-spinner').hidden)", timeout=1200)
            assert cdp.evaluate("document.getElementById('error-message').hidden"), cdp.evaluate("document.getElementById('error-message').textContent")
            cdp.wait("['original-result','mask-result','overlay-result'].every(id => { const i=document.getElementById(id);return i.complete && i.naturalWidth > 0; })")
            report["real_prediction"] = True
            no_overflow()
            assert cdp.evaluate("document.getElementById('comparison-frame').clientWidth >= document.getElementById('viewer').clientWidth - 2"), "Comparison frame must fill its viewer"
            screenshot("desktop_prediction_PRIVATE.png")
            cdp.evaluate("document.getElementById('reveal').value = '73'; document.getElementById('reveal').dispatchEvent(new Event('input')); document.getElementById('opacity').value = '41'; document.getElementById('opacity').dispatchEvent(new Event('input'))")
            assert cdp.evaluate("document.getElementById('compare-overlay').style.opacity === '0.41' && document.getElementById('overlay-layer').style.clipPath.includes('27%')")
            report["comparison_and_opacity_controls"] = True
            cdp.evaluate("document.getElementById('fullscreen-button').click()", gesture=True)
            cdp.wait("document.fullscreenElement !== null")
            cdp.evaluate("document.exitFullscreen()")
            report["fullscreen"] = True
            for name in ("mask", "overlay"):
                cdp.evaluate(f"document.getElementById('download-{name}').click()", gesture=True)
            end = time.monotonic() + 30
            while time.monotonic() < end and not all((downloads / f"phase2_{name}.png").is_file() for name in ("mask", "overlay")):
                time.sleep(0.25)
            assert all((downloads / f"phase2_{name}.png").is_file() for name in ("mask", "overlay"))
            from PIL import Image, ImageChops
            for name, reference_name in (("mask", "sample_686_prediction_native.png"), ("overlay", "sample_686_overlay_native.png")):
                with Image.open(downloads / f"phase2_{name}.png") as actual, Image.open(ROOT / "outputs/run/phase2_live_inference" / reference_name) as expected:
                    assert ImageChops.difference(actual.convert("RGB"), expected.convert("RGB")).getbbox() is None
            report["downloads_match_reference_pixels"] = True
            for width in (1024, 768, 390):
                cdp.call("Emulation.setDeviceMetricsOverride", width=width, height=900, deviceScaleFactor=1, mobile=width < 500)
                no_overflow()
            screenshot("mobile_prediction_PRIVATE.png")
            report["responsive_prediction_widths"] = [1440, 1024, 768, 390]
            # Regression: a rejected replacement must invalidate the previous
            # actual prediction, preview, submit state and download links.
            cdp.call("DOM.setFileInputFiles", files=[str(invalid)], nodeId=node)
            cdp.wait("!document.getElementById('error-message').hidden")
            assert cdp.evaluate("document.getElementById('result-content').hidden && document.getElementById('input-preview').hidden && document.getElementById('run-button').disabled && !document.getElementById('download-mask').hasAttribute('href') && !document.getElementById('download-overlay').hasAttribute('href') && !document.getElementById('mask-result').hasAttribute('src')")
            report["invalid_replacement_clears_previous_prediction"] = True
            # Client-side size rejection uses the exact server image-byte limit.
            cdp.call("DOM.setFileInputFiles", files=[str(image)], nodeId=node)
            cdp.wait("!document.getElementById('input-preview').hidden")
            cdp.evaluate("const tooLarge = new File([new Uint8Array(Number(document.body.dataset.maxImageBytes) + 1)], 'oversize.png', {type:'image/png'}); const oversized = new DataTransfer(); oversized.items.add(tooLarge); document.getElementById('drop-zone').dispatchEvent(new DragEvent('drop', {dataTransfer:oversized,bubbles:true,cancelable:true}));")
            assert cdp.evaluate("document.getElementById('run-button').disabled && document.getElementById('input-preview').hidden && !document.getElementById('error-message').hidden")
            report["oversized_replacement_clears_selection"] = True
            # A pending browser decode must not resurrect a replaced image.
            cdp.evaluate("window.__decodeBeforeTest = Image.prototype.decode; window.__pendingDecodes = []; Image.prototype.decode = function(){ return new Promise(resolve => window.__pendingDecodes.push(resolve)); };")
            cdp.call("DOM.setFileInputFiles", files=[str(image)], nodeId=node)
            cdp.wait("window.__pendingDecodes.length === 1")
            cdp.evaluate("const bad = new DataTransfer(); bad.items.add(new File(['invalid'], 'wrong.txt', {type:'text/plain'})); document.getElementById('drop-zone').dispatchEvent(new DragEvent('drop', {dataTransfer:bad,bubbles:true,cancelable:true})); Image.prototype.decode = window.__decodeBeforeTest; window.__pendingDecodes.forEach(resolve => resolve());")
            cdp.wait("document.getElementById('input-preview').hidden && document.getElementById('run-button').disabled")
            cdp.evaluate("delete window.__decodeBeforeTest; delete window.__pendingDecodes;")
            report["stale_decode_cannot_restore_invalidated_selection"] = True
            cdp.evaluate("document.getElementById('reset-button').click()")
            assert cdp.evaluate("document.getElementById('result-content').hidden && document.getElementById('input-preview').hidden && document.getElementById('prompt').value === ''")
            report["reset_clears_results"] = True
            assert not cdp.errors, cdp.errors
            report.update(status="passed", browser_javascript_errors=cdp.errors)
            print(json.dumps(report, indent=2), flush=True)
        except Exception as error:
            report.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if cdp:
                try:
                    cdp.call("Browser.close")
                except Exception:
                    pass
                cdp.socket.close()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()  # Only this script's newly created browser process.


if __name__ == "__main__":
    main()
