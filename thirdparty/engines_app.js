// Third-party sink harness. Local only, port 5020.
//
// Both of our Flask apps were written by us, which means we chose which endpoints are
// safe and which are not. A reviewer can fairly say the sinks were designed to fit the
// argument. This harness removes that objection for the escaping behaviour itself:
// every sink below is a documented output mode of a widely used template engine, and
// the decision about what gets escaped belongs to the library, not to us.
//
//   ejs         <%= %> escaped      <%- %> raw
//   handlebars  {{ }}  escaped      {{{ }}} raw
//   mustache    {{ }}  escaped      {{{ }}} raw
//   pug         #{}    escaped      !{}     raw
//
// We still wrote the surrounding page, so this is a partial answer to external validity,
// not a full one. A genuinely third-party application (Juice Shop, DVWA) remains the
// stronger test and is run separately.

const http = require('http');
const url = require('url');
const ejs = require('ejs');
const handlebars = require('handlebars');
const mustache = require('mustache');
const pug = require('pug');

const PORT = process.env.TP_PORT ? parseInt(process.env.TP_PORT, 10) : 5020;

// name -> [render(value) -> body html, sink label]
const SINKS = {
  'ejs-escaped':        [v => ejs.render('<p>You searched: <%= q %></p>', { q: v }),      'ejs_escaped'],
  'ejs-raw':            [v => ejs.render('<p>You searched: <%- q %></p>', { q: v }),      'ejs_raw'],
  'ejs-attr':           [v => ejs.render('<input value="<%- q %>">', { q: v }),           'ejs_raw_attribute'],
  'handlebars-escaped': [v => handlebars.compile('<p>You searched: {{q}}</p>')({ q: v }), 'handlebars_escaped'],
  'handlebars-raw':     [v => handlebars.compile('<p>You searched: {{{q}}}</p>')({ q: v }), 'handlebars_raw'],
  'mustache-escaped':   [v => mustache.render('<p>You searched: {{q}}</p>', { q: v }),    'mustache_escaped'],
  'mustache-raw':       [v => mustache.render('<p>You searched: {{{q}}}</p>', { q: v }),  'mustache_raw'],
  'pug-escaped':        [v => pug.render('p You searched: #{q}', { q: v }),               'pug_escaped'],
  'pug-raw':            [v => pug.render('p You searched: !{q}', { q: v }),               'pug_raw'],
};

const page = body =>
  `<!doctype html><html><body>${body}</body></html>`;

const server = http.createServer((req, res) => {
  const parsed = url.parse(req.url, true);
  const name = parsed.pathname.replace(/^\/+/, '');
  const q = parsed.query.q === undefined ? '' : String(parsed.query.q);

  if (name === 'sinks') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    const map = {};
    for (const k of Object.keys(SINKS)) map['/' + k] = SINKS[k][1];
    return res.end(JSON.stringify(map));
  }
  if (name === '' ) {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    return res.end(page('<p>third-party template engine sinks</p><ul>' +
      Object.keys(SINKS).map(k => `<li>/${k}</li>`).join('') + '</ul>'));
  }
  if (!SINKS[name]) {
    res.writeHead(404, { 'Content-Type': 'text/html' });
    return res.end(page('<p>not found</p>'));
  }
  let body;
  try {
    body = SINKS[name][0](q);
  } catch (e) {
    body = '<p>render error: ' + String(e.message).slice(0, 120) + '</p>';
  }
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end(page(body));
});

server.listen(PORT, '127.0.0.1', () => {
  console.log('third-party sinks on http://127.0.0.1:' + PORT);
});
