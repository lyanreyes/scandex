"""
Serve the ScanDex web app.

  python serve.py            http://localhost:8000  (PC webcam works on localhost)
  python serve.py --https    https://<your-LAN-IP>:8443 for phones on the same Wi-Fi

Browsers only allow camera access on HTTPS pages or on localhost, so phones
need the --https mode. A self-signed certificate is generated on first run;
the phone will show a security warning once - choose "Advanced -> Proceed".
"""
import datetime
import http.server
import ipaddress
import socket
import ssl
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).parent
WEB = ROOT / "web"
CERT, KEY = ROOT / ".cert" / "cert.pem", ROOT / ".cert" / "key.pem"


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def make_cert(ip):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "ScanDex local")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now).not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                x509.IPAddress(ipaddress.ip_address(ip))]), critical=False)
            .sign(key, hashes.SHA256()))
    CERT.parent.mkdir(exist_ok=True)
    KEY.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                      serialization.PrivateFormat.TraditionalOpenSSL,
                                      serialization.NoEncryption()))
    CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".webmanifest": "application/manifest+json",
                      ".json": "application/json", ".svg": "image/svg+xml"}

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    https = "--https" in sys.argv
    ip = lan_ip()
    port = 8443 if https else 8000
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), partial(Handler, directory=str(WEB)))
    if https:
        if not CERT.exists():
            make_cert(ip)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(CERT, KEY)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print(f"ScanDex on https://{ip}:{port}  (open this on your phone, same Wi-Fi)")
        print(f"          https://localhost:{port}")
    else:
        print(f"ScanDex on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
