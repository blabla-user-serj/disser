import http.server, os

BASEDIR = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/download':
            filepath = os.path.join(BASEDIR, 'index.html')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Disposition',
                             'attachment; filename="uralskaya-stal-hr-forecasting.html"')
            with open(filepath, 'rb') as f:
                data = f.read()
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
        elif path == '/download-pptx':
            filepath = os.path.join(BASEDIR, 'uralskaya-stal-hr-forecasting.pptx')
            self.send_response(200)
            self.send_header('Content-Type',
                             'application/vnd.openxmlformats-officedocument.presentationml.presentation')
            self.send_header('Content-Disposition',
                             'attachment; filename="uralskaya-stal-hr-forecasting.pptx"')
            with open(filepath, 'rb') as f:
                data = f.read()
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    os.chdir(BASEDIR)
    srv = http.server.HTTPServer(('0.0.0.0', 8765), Handler)
    print('Serving on :8765', flush=True)
    srv.serve_forever()
