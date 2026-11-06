// Inlines the renderer's ES modules into one self-contained viewer page, which
// the Python side serves into Streamlit with the system to show filled in.
// Lifted from attractor-screensaver's tools/build-single.mjs.
//
//   node web/build.mjs     ->  src/vizdyn/frontend/webgl/viewer.html
//
// `bundle(entry)` is also importable, for embedding the renderer in some other
// page: it returns the module registry plus a `__req(entry)` loader.

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.join(ROOT, 'src');

const EXPORT_RE = /^export\s+(?:(async\s+function|function|class|const|let|var)\s+)([A-Za-z_$][\w$]*)/gm;

function transform(file) {
  let code = fs.readFileSync(path.join(SRC, file), 'utf8');
  const deps = [];

  code = code.replace(
    /^import\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s+'\.\/([^']+)';?$/gm,
    (_, name, dep) => { deps.push(dep); return `const ${name} = __req('${dep}');`; });

  code = code.replace(
    /^import\s+\{([^}]+)\}\s+from\s+'\.\/([^']+)';?$/gm,
    (_, names, dep) => { deps.push(dep); return `const {${names.replace(/\s+as\s+/g, ': ')}} = __req('${dep}');`; });

  const exported = [];
  for (const m of code.matchAll(EXPORT_RE)) exported.push(m[2]);
  code = code.replace(/^export\s+/gm, '');

  const tail = exported.length
    ? `\nObject.assign(__exports, { ${exported.join(', ')} });\n`
    : '';

  return { code: code + tail, deps: [...new Set(deps)] };
}

export function bundle(entry = 'main.js') {
  const modules = new Map();
  (function collect(file) {
    if (modules.has(file)) return;
    const m = transform(file);
    modules.set(file, m);
    m.deps.forEach(collect);
  })(entry);

  const registry = [...modules.entries()].map(([name, m]) =>
    `__def(${JSON.stringify(name)}, function (__exports, __req) {\n${m.code}\n});`
  ).join('\n\n');

  return { runtime: RUNTIME, registry, count: modules.size };
}

const RUNTIME = `
const __mods = {};
const __cache = {};
function __def (name, fn) { __mods[name] = fn; }
function __req (name) {
  if (!__cache[name]) {
    const exports = __cache[name] = {};
    __mods[name](exports, __req);
  }
  return __cache[name];
}
`;

if (process.argv[1] !== fileURLToPath(import.meta.url)) {
  // Imported for its bundle() export; nothing else to do.
} else {

const { runtime, registry, count } = bundle('embed.js');
const html = fs.readFileSync(path.join(ROOT, 'viewer.html'), 'utf8');
const bundled = html.replace(
  "import { mount } from './src/embed.js';",
  `const { mount } = (function () {
${runtime}
${registry}
return __req('embed.js');
})();`);
if (bundled === html) throw new Error('viewer.html: embed import not found');

const outDir = path.join(ROOT, '..', 'src', 'vizdyn', 'frontend', 'webgl');
fs.mkdirSync(outDir, { recursive: true });
const out = path.join(outDir, 'viewer.html');
fs.writeFileSync(out, bundled);
console.log(`${path.relative(path.join(ROOT, '..'), out)}  (${(bundled.length / 1024).toFixed(0)} kB, ${count} modules)`);

}
