import socket, base64, os, json, time, subprocess, urllib.request

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
        if not res: break
        try:
            d = json.loads(res)
            if d.get('id') == msg_id:
                return d
        except:
            pass
    return None

def main():
    print('[*] Launching headless Chrome for upload test...')
    proc = subprocess.Popen([
        r'C:\Program Files\Google\Chrome\Application\chrome.exe',
        '--headless=new',
        '--remote-debugging-port=9222',
        '--window-size=1400,1600',
        '--user-data-dir=C:\\Users\\mrpat\\AppData\\Local\\Temp\\chrome_cdp_upload',
        'http://127.0.0.1:8000/'
    ])
    time.sleep(3)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:9222/json')
        tabs = json.loads(resp.read().decode('utf-8'))
        page_tab = next(t for t in tabs if 'http://127.0.0.1:8000/' in t.get('url', ''))
        s = ws_connect(page_tab['webSocketDebuggerUrl'])

        # Find file input and set file
        test_file = r'C:\Desktop\ppe-generative-project\crops_64\test\helmet\ds1_hard_hat_workers0_0.png'
        
        # Enable DOM
        call_cdp(s, 1, 'DOM.enable')
        doc = call_cdp(s, 2, 'DOM.getDocument')
        file_input_node = call_cdp(s, 3, 'DOM.querySelector', {
            'nodeId': doc['result']['root']['nodeId'],
            'selector': '#file-input'
        })
        node_id = file_input_node['result']['nodeId']
        print('[*] Setting file on input #file-input:', test_file)
        call_cdp(s, 4, 'DOM.setFileInputFiles', {
            'files': [test_file],
            'nodeId': node_id
        })

        # Wait 2 seconds for cropper to initialize
        time.sleep(2)

        # Click Analyse button
        print('[*] Clicking analyse button...')
        click_js = "document.querySelector('#btn-analyse').click();"
        call_cdp(s, 5, 'Runtime.evaluate', {'expression': click_js})

        # Wait 3 seconds for analysis
        time.sleep(3)

        # Capture screenshot
        snap = call_cdp(s, 6, 'Page.captureScreenshot', {'format': 'png'})
        img_bytes = base64.b64decode(snap['result']['data'])
        out_path = 'scratch_recon/live_dashboard_upload_test.png'
        with open(out_path, 'wb') as f:
            f.write(img_bytes)
        print('[+] Saved upload test screenshot to:', out_path)

    finally:
        proc.terminate()

if __name__ == '__main__':
    main()

