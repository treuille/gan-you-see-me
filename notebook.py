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

class Notebook:
    _IP = '' #'ec2-34-239-163-48.compute-1.amazonaws.com'
    _PORT = 8314
    _STATIC_PATHS = {
        '/': 'html/notebook.html',
        '/notebook.js': 'html/notebook.js'
    }
    _DYNAMIC_PATH = '/dynamic.html'
    _OPEN_WEBPAGE_SECS = 0.2
    _FINAL_SHUTDOWN_SECS = 3.0

    def __init__(self):
        # load the template html and javasript
        self._static_resources = {}
        for http_path, resource_path in Notebook._STATIC_PATHS.items():
            with open(resource_path) as data:
                self._static_resources[http_path] = bytes(data.read(), 'utf-8')

        # This is the list of dynamic html elements.
        self._dynamic_elts = []

        # Number of elements transmitted via GET. (None means no GET received.)
        self._n_transmitted_elts = None

        # Create the webserver.
        class NotebookHandler(BaseHTTPRequestHandler):
            def do_GET(s):
                resource = self._get_resource(s.path)
                if resource == None:
                    s.send_error(404)
                else:
                    s.send_response(200)
                    s.end_headers()
                    s.wfile.write(resource)

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
        if self._n_transmitted_elts != len(self._dynamic_elts):
            print(f'Sleeping for {Notebook._FINAL_SHUTDOWN_SECS} '
                'seconds to flush all elements.')
            time.sleep(Notebook._FINAL_SHUTDOWN_SECS)
        self._keep_running = False
        self._httpd.server_close()
        print('Closed down server.')

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

            - "auto"   : figures out the
            - "alert"  : formats the string as an alert
            - "header" : formats the string as a header
            - "info"   : prints out df.info() on a DataFrame-like object
        """
        # These types are output specially.
        dataframe_like_types = (pd.DataFrame, pd.Series, pd.Index, np.ndarray)
        figure_like_types = (plotting.Figure,)

        # Dispatch based on the format argument.
        if fmt == 'auto':
            string_buffer = []
            def flush_buffer():
                if string_buffer:
                    self._write_text(' '.join(string_buffer))
                string_buffer[:] = []
            for arg in args:
                if isinstance(arg, dataframe_like_types):
                    flush_buffer()
                    self._write_data(arg)
                elif isinstance(arg, figure_like_types):
                    flush_buffer()
                    self._write_plot(arg)
                else:
                    string_buffer.append(str(arg))
            flush_buffer()
        elif fmt == 'alert':
            self._write_dom('div', args, classes=['alert', 'alert-danger'],
                spaces_become=' ')
        elif fmt == 'header':
            self._write_dom('h4', args, classes=['mt-3'])
        elif fmt == 'info':
            if len(args) != 1:
                raise RuntimeError('fmt="info" only operates on one argument.')
            if not isinstance(args[0], dataframe_like_types):
                raise RuntimeError('fmt="info" only operates on DataFrames.')
            stream = io.StringIO()
            pd.DataFrame(args[0]).info(buf=stream)
            self._write_text(stream.getvalue())
        elif fmt == 'img':
            self._write_image(args[0])
        else:
            raise RuntimeError(f'fmt="{fmt}" not valid.')


    def _run_server(self):
        self._keep_running = True
        while self._keep_running:
            self._httpd.handle_request()

    def _open_webpage_as_needed(self):
        """If we've received no requests after a time, open a webpage."""
#        time.sleep(Notebook._OPEN_WEBPAGE_SECS)
#        if self._n_transmitted_elts == None:
#            os.system(f'open http://{Notebook._IP}:{Notebook._PORT}')
        pass

    def _get_resource(self, path):
        """Returns a static / dynamic resource, or none if path is invalid."""
        if path in Notebook._STATIC_PATHS:
            return self._static_resources[path]
        elif path == Notebook._DYNAMIC_PATH:
            elts = '<div class="w-100"></div>'.join(
                f'<div class="col mb-2">{elt}</div>'
                    for elt in self._dynamic_elts)
            self._n_transmitted_elts = len(self._dynamic_elts)
            return bytes(elts, 'utf-8')
        else:
            return None

    def _write_dom(self, tag, args, spaces_become='&nbsp;', classes=[]):
        """Esacapes and wraps the text in an html tag."""
        escaped_text = html.escape(' '.join(str(arg) for arg in args)) \
            .replace(' ', spaces_become).replace('\n', '<br/>')
        tag_class = ''
        if classes:
            tag_class = ' class="%s"' % ' '.join(classes)
        self._dynamic_elts.append(
            f'<{tag}{tag_class}>{escaped_text}</{tag}>')

    def _write_text(self, text):
        """Writes some text to the notebook."""
        self._write_dom('samp', (text,))

    def _write_plot(self, p):
        """Adds a Bokeh plot to the notebook."""
        plot_script, plot_html = bokeh.embed.components(p)
        self._dynamic_elts.append(plot_html + plot_script)

    def _write_image(self, img): 
        img = Image.fromarray(255 - (img * 255).astype(np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='png')
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        style = 'style="border: 1px solid black;"'
        img_html = f'<img {style} src="data:image/png;base64,{img_base64}">'
        self._dynamic_elts.append(img_html)

    def _write_data(self, df):
        """Render a Pandas dataframe as html."""
        if type(df) != pd.DataFrame:
            df = pd.DataFrame(df)
        id = f'dataframe-{uuid.uuid4()}'
        pandas_table = '<table border="1" class="dataframe">'
        notebook_table = f'<table id="{id}">'
        table_html = df.to_html(bold_rows=False, sparsify=False) \
            .replace(pandas_table, notebook_table)
        table_script = f'<script>notebook.styleDataFrame("{id}");</script>'
        self._dynamic_elts.append(table_html + table_script)
