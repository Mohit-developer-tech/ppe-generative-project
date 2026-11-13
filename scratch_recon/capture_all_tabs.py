import socket, base64, hashlib, os, json, time, subprocess, urllib.request

def ws_connect(ws_url):
    host_port, path = ws_url.replace('ws://', '').split('/', 1)
    host, port = host_port.split(':')
    s = socket.create_connection((host, int(port)), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode('utf-8')
    req = (f'GET /{path} HTTP/1.1\r\n'
           f'Host: {host}:{port}\r\n'
           f'Upgrade: websocket\r\n'
           f'Connection: Upgrade\r\n'
           f'Sec-WebSocket-Key: {key}\r\n'
           f'Sec-WebSocket-Version: 13\r\n\r\n')
    s.sendall(req.encode('utf-8'))
    resp = s.recv(4096)
    if b'101 ' not in resp:
        raise RuntimeError('WS handshake failed')
    return s

def ws_send(s, msg):
    data = msg.encode('utf-8')
    length = len(data)
    frame = bytearray([0x81])
    mask = os.urandom(4)
    if length < 126:
        frame.append(0x80 | length)
    elif length < 65536:
        frame.append(0x80 | 126)
        frame.extend(length.to_bytes(2, 'big'))
    else:
        frame.append(0x80 | 127)
        frame.extend(length.to_bytes(8, 'big'))
    frame.extend(mask)
    masked_data = bytearray(b ^ mask[i % 4] for i, b in enumerate(data))
    frame.extend(masked_data)
    s.sendall(frame)

def ws_recv(s):
    hdr = s.recv(2)
    if not hdr: return None
    b2 = hdr[1]
    length = b2 & 0x7F
    if length == 126:
        ext = s.recv(2)
        length = int.from_bytes(ext, 'big')
    elif length == 127:
        ext = s.recv(8)
        length = int.from_bytes(ext, 'big')
    data = bytearray()
    while len(data) < length:
        chunk = s.recv(min(length - len(data), 65536))
        if not chunk: break
        data.extend(chunk)
    return data.decode('utf-8', errors='ignore')

def call_cdp(s, msg_id, method, params=None):
    payload = {'id': msg_id, 'method': method}
    if params:
        payload['params'] = params
    ws_send(s, json.dumps(payload))
    while True:
        res = ws_recv(s)
        if not res: return None
        try:
            d = json.loads(res)
            if d.get('id') == msg_id:
                return d
        except Exception:
            pass

def main():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    proc = subprocess.Popen([
        chrome_path,
        "--headless=new",
        "--remote-debugging-port=9222",
        "--window-size=1400,1050",
        "--disable-gpu",
        "http://127.0.0.1:8000"
    ])
    print("[*] Chrome launched with PID", proc.pid)
    time.sleep(3)

    try:
        targets_raw = urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=5).read()
        targets = json.loads(targets_raw.decode('utf-8'))
        page = next(t for t in targets if t.get('type') == 'page')
        ws_url = page['webSocketDebuggerUrl']
        s = ws_connect(ws_url)
        print("[*] Connected to Chrome DevTools Protocol")

        # Wait for page readiness
        call_cdp(s, 1, "Page.enable")
        time.sleep(2)

        # 1. Capture Autoencoder Tab with sample click
        eval_sample = """
        (() => {
            const btns = document.querySelectorAll('#sample-gallery button');
            if (btns.length > 0) { btns[0].click(); return true; }
            return false;
        })()
        """
        call_cdp(s, 2, "Runtime.evaluate", {"expression": eval_sample})
        time.sleep(3)
        snap1 = call_cdp(s, 3, "Page.captureScreenshot", {"format": "png"})
        if snap1 and "result" in snap1:
            with open("screenshot_tab1_autoencoder.png", "wb") as f:
                f.write(base64.b64decode(snap1["result"]["data"]))
            print("[✓] Saved screenshot_tab1_autoencoder.png")

        # 2. Switch to CVAE Tab & click Generate
        eval_cvae = """
        (() => {
            const btn = document.querySelector('button[data-model-id="cvae"]');
            if (btn) { btn.click(); return true; }
            return false;
        })()
        """
        call_cdp(s, 4, "Runtime.evaluate", {"expression": eval_cvae})
        time.sleep(2)
        call_cdp(s, 5, "Runtime.evaluate", {"expression": "document.getElementById('cvae-generate-btn')?.click()"})
        time.sleep(3)
        snap2 = call_cdp(s, 6, "Page.captureScreenshot", {"format": "png"})
        if snap2 and "result" in snap2:
            with open("screenshot_tab2_cvae.png", "wb") as f:
                f.write(base64.b64decode(snap2["result"]["data"]))
            print("[✓] Saved screenshot_tab2_cvae.png")

        # 3. Switch to CGAN Tab & click Generate
        eval_cgan = """
        (() => {
            const btn = document.querySelector('button[data-model-id="cgan"]');
            if (btn) { btn.click(); return true; }
            return false;
        })()
        """
        call_cdp(s, 7, "Runtime.evaluate", {"expression": eval_cgan})
        time.sleep(2)
        call_cdp(s, 8, "Runtime.evaluate", {"expression": "document.getElementById('cgan-generate-btn')?.click()"})
        time.sleep(3)
        snap3 = call_cdp(s, 9, "Page.captureScreenshot", {"format": "png"})
        if snap3 and "result" in snap3:
            with open("screenshot_tab3_cgan.png", "wb") as f:
                f.write(base64.b64decode(snap3["result"]["data"]))
            print("[✓] Saved screenshot_tab3_cgan.png")

        # 4. Switch to Diffusion Tab & click Restore
        eval_diff = """
        (() => {
            const btn = document.querySelector('button[data-model-id="diffusion"]');
            if (btn) { btn.click(); return true; }
            return false;
        })()
        """
        call_cdp(s, 10, "Runtime.evaluate", {"expression": eval_diff})
        time.sleep(2)
        call_cdp(s, 11, "Runtime.evaluate", {"expression": "document.getElementById('diff-restore-btn')?.click()"})
        time.sleep(6) # Diffusion sampling takes ~4s
        snap4 = call_cdp(s, 12, "Page.captureScreenshot", {"format": "png"})
        if snap4 and "result" in snap4:
            with open("screenshot_tab4_diffusion.png", "wb") as f:
                f.write(base64.b64decode(snap4["result"]["data"]))
            print("[✓] Saved screenshot_tab4_diffusion.png")

        # 5. Switch to Policy Tab & evaluate
        eval_pol = """
        (() => {
            const btn = document.querySelector('button[data-model-id="policy"]');
            if (btn) { btn.click(); return true; }
            return false;
        })()
        """
        call_cdp(s, 13, "Runtime.evaluate", {"expression": eval_pol})
        time.sleep(3)
        snap5 = call_cdp(s, 14, "Page.captureScreenshot", {"format": "png"})
        if snap5 and "result" in snap5:
            with open("screenshot_tab5_policy.png", "wb") as f:
                f.write(base64.b64decode(snap5["result"]["data"]))
            print("[✓] Saved screenshot_tab5_policy.png")

        s.close()
    finally:
        proc.kill()
        print("[*] Chrome closed cleanly.")

if __name__ == "__main__":
    main()

