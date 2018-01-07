"""Used to output data (tables, charts, and text) to a web browser."""

import bokeh.embed
import html
import io
import numpy as np
import os
import pandas as pd
import threading
import time
import traceback
import uuid

# for image encoding
from PIL import Image
import io
import base64

from bokeh import plotting
from http.server import HTTPServer, BaseHTTPRequestHandler

class Block:
    """A tree of html elements which can be 'dirty' if any subtree has
    been editted."""

    def __init__(self, parent):
        """Creates a new block. parent=None means no parent."""
        self._parent = parent
        self._dirty = False

    def get_html_list(self):
        """Returns a list of html elements to display."""
        raise NotImplementedError('Block is an abstract base class.')

    def is_dirty(self):
        """Indicates if any elements have been added to subtree since last
        get_html_list()."""
        return self._dirty

    def set_dirty(self, dirty):
        """Sets dirty flag. True propogates up the tree."""
        self._dirty = dirty
        if (dirty and self._parent):
            print('Propogating dirty flag up the tree.') # debug!
            self._parent.set_dirty()

class HTMLBLock(Block):
    """A single HTML element."""

    def __init__(self, parent, html):
        super().__init__(parent)
        self._html = html

    def get_html_list(self):
        return [ self._html ]

class VerticalBlock(Block):
    """A set of vertically stacked elements."""

    def __init__(self, parent):
        super().__init__(parent)
        self.clear()

    def get_html_list(self):
        """Returns a list of html elements to display."""
        html_list = []
        for block in self._blocks:
            html_list.extend(block.get_html_list())
        self._dirty = False
        return html_list

    def clear(self):
        """Clears the current list."""
        self._blocks = []
        self.set_dirty(True)

    def add_html(self, html):
        """Adds a html to this block."""
        self._blocks.append(HTMLBLock(self, html))
        self.set_dirty(True)

    def dom_elt(self, tag, args, spaces_become='&nbsp;', classes=[]):
        """Adds a DOM element (escaping the text)."""

        escaped_text = html.escape(' '.join(str(arg) for arg in args)) \
            .replace(' ', spaces_become).replace('\n', '<br/>')
        tag_class = ''
        if classes:
            tag_class = ' class="%s"' % ' '.join(classes)
        self.add_html(
            f'<{tag}{tag_class}>{escaped_text}</{tag}>')

    def text(self, text):
        """Adds some text to the notebook."""
        self.dom_elt('samp', (text,))

    def data_frame(self, df):
        """Render a Pandas dataframe as html."""
        if type(df) != pd.DataFrame:
            df = pd.DataFrame(df)
        id = f'dataframe-{uuid.uuid4()}'
        pandas_table = '<table border="1" class="dataframe">'
        notebook_table = f'<table id="{id}">'
        table_html = df.to_html(bold_rows=False, sparsify=False) \
            .replace(pandas_table, notebook_table)
        table_script = f'<script>notebook.styleDataFrame("{id}");</script>'
        self.add_html(table_html + table_script)

    def plot(self, p):
        """Adds a Bokeh plot to the notebook."""
        plot_script, plot_html = bokeh.embed.components(p)
        self.add_html(plot_html + plot_script)

    def image(self, img):
        img = Image.fromarray(255 - (img * 255).astype(np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='png')
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        style = 'style="border: 1px solid black;"'
        img_html = f'<img {style} src="data:image/png;base64,{img_base64}">'
        self.add_html(img_html)

    def progress(self, percent):
        percent = max(0, min(100, int(percent * 100)))
        html = f"""
            <div class="progress">
                <div class="progress-bar" role="progressbar"
                    style="width: {percent}%"
                    aria-valuenow="{percent}"
                    aria-valuemin="0"
                    aria-valuemax="100">
                </div>
            </div>
        """
        self.add_html(html)

    def __call__(self, *args, fmt='auto'):
        """Writes it's arguments to notebook pages.

        with Notebook() as print:
            print('Hello world.')
            print('This is an alert', fmt='alert')
            print('This is a dataframe', pd.DataFrame([1, 2, 3]))

        Supported types are:

            - Bokeh figures.
            - Pandas-DataFrame-like objects: DataFame, Series, and numpy.Array
            - String-like objects: By default, objects are cast to strings.

        The optional `fmt` argument can take on several values:

            - "auto"     : figures out the
            - "alert"    : formats the string as an alert
            - "header"   : formats the string as a header
            - "info"     : prints out df.info() on a DataFrame-like object
            - "img"      : prints an image out
            - "progress" : prints out a progress bar (for a 0<num<1)
        """
        # These types are output specially.
        dataframe_like_types = (pd.DataFrame, pd.Series, pd.Index, np.ndarray)
        figure_like_types = (plotting.Figure,)

        # Dispatch based on the format argument.
        if fmt == 'auto':
            string_buffer = []
            def flush_buffer():
                if string_buffer:
                    self.text(' '.join(string_buffer))
                string_buffer[:] = []
            for arg in args:
                if isinstance(arg, dataframe_like_types):
                    flush_buffer()
                    self.data_frame(arg)
                elif isinstance(arg, figure_like_types):
                    flush_buffer()
                    self.plot(arg)
                else:
                    string_buffer.append(str(arg))
            flush_buffer()
        elif fmt == 'alert':
            self.dom_elt('div', args, classes=['alert', 'alert-danger'],
                spaces_become=' ')
        elif fmt == 'header':
            self.dom_elt('h4', args, classes=['mt-3'])
        elif fmt == 'info':
            if len(args) != 1:
                raise RuntimeError('fmt="info" only operates on one argument.')
            if not isinstance(args[0], dataframe_like_types):
                raise RuntimeError('fmt="info" only operates on DataFrames.')
            stream = io.StringIO()
            pd.DataFrame(args[0]).info(buf=stream)
            self.text(stream.getvalue())
        elif fmt == 'img':
            self.image(args[0])
        elif fmt == 'progress':
            self.progress(args[0])
        else:
            raise RuntimeError(f'fmt="{fmt}" not valid.')

class Notebook(VerticalBlock):
    _IP = '127.0.0.1'
    _PORT = 8314
    _STATIC_PATHS = {
        '/': 'html/notebook.html',
        '/notebook.js': 'html/notebook.js'
    }
    _DYNAMIC_PATH = '/dynamic.html'
    _OPEN_WEBPAGE_SECS = 0.2
    _FINAL_SHUTDOWN_SECS = 3.0

    def __init__(self):
        super().__init__(None)

        # load the template html and javasript
        self._static_resources = {}
        for http_path, resource_path in Notebook._STATIC_PATHS.items():
            with open(resource_path) as data:
                self._static_resources[http_path] = bytes(data.read(), 'utf-8')

        # Create the webserver.
        self._received_GET = False
        class NotebookHandler(BaseHTTPRequestHandler):
            def do_GET(s):
                resource = self._get_resource(s.path)
                if resource == None:
                    s.send_error(404)
                else:
                    s.send_response(200)
                    s.end_headers()
                    s.wfile.write(resource)
                self._received_GET = True

            def log_message(self, format, *args):
                return

        self._httpd = \
            HTTPServer((Notebook._IP, Notebook._PORT), NotebookHandler)

    def __enter__(self):
        # start the webserver
        threading.Thread(target=self._run_server, daemon=True).start()

        # if no gets received after a timeout, then launch the browser
        threading.Thread(target=self._open_webpage_as_needed,
            daemon=True).start()

        # all done
        print(f'Started server at http://{Notebook._IP}:{Notebook._PORT}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Shut down the server."""
        # Display the stack trace if necessary.
        if exc_type != None:
            tb_list = traceback.format_list(traceback.extract_tb(exc_tb))
            tb_list.append(f'{exc_type.__name__}: {exc_val}')
            self('\n'.join(tb_list), fmt='alert')

        # A small delay to flush anything left.
        time.sleep(Notebook._OPEN_WEBPAGE_SECS)

        # Delay server shutdown if we haven't transmitted everything yet
        print(f'Sutting down server with dirty={self.is_dirty()}.') # debug
        if self.is_dirty() or not self._received_GET:
            print(f'Sleeping for {Notebook._FINAL_SHUTDOWN_SECS} '
                'seconds to flush all elements.')
            time.sleep(Notebook._FINAL_SHUTDOWN_SECS)
        self._keep_running = False
        self._httpd.server_close()
        print('Closed down server.')

    def _run_server(self):
        self._keep_running = True
        while self._keep_running:
            self._httpd.handle_request()

    def _open_webpage_as_needed(self):
        """If we've received no requests after a time, open a webpage."""
        time.sleep(Notebook._OPEN_WEBPAGE_SECS)
        if not self._received_GET:
            os.system(f'open http://{Notebook._IP}:{Notebook._PORT}')

    def _get_resource(self, path):
        """Returns a static / dynamic resource, or none if path is invalid."""
        if path in Notebook._STATIC_PATHS:
            return self._static_resources[path]
        elif path == Notebook._DYNAMIC_PATH:
            elts = '<div class="w-100"></div>'.join(
                f'<div class="col mb-2">{elt}</div>'
                    for elt in self.get_html_list())
            return bytes(elts, 'utf-8')
        else:
            return None
