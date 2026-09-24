import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'.presentation-build/research-deps'),str(root)]
from bakuml import config
config.ARTIFACTS_DIR=root/'.presentation-build/research-h3'
from app.flask_app import app
from flask import send_from_directory
@app.route('/presentation/<path:name>')
def presentation_file(name):
    return (root/'.presentation-build'/name).read_text(encoding='utf-8')
@app.after_request
def demo_label(response):
    if response.is_json:
        data=response.get_json()
        if isinstance(data,dict) and 'colors' in data:
            data['colors']=['#'+''.join(f'{round(v*255):02x}' for v in color[:3]) if isinstance(color,(list,tuple)) else color for color in data['colors']]
            response.set_data(app.json.dumps(data))
    elif response.content_type.startswith('text/html'):
        text=response.get_data(as_text=True)
        text=text.replace('<body>','<body><div style="background:#d9f1ed;padding:8px 24px;text-align:center;font:600 14px sans-serif;color:#124e48">SYNTHETIC RESEARCH DEMO · H3 grid · Ridge forecast fallback</div>')
        response.set_data(text)
    return response
app.run(host='127.0.0.1',port=5051,debug=False)
