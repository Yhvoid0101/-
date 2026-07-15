const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const packageRoot = 'C:/Program Files/WindowsApps';
const packageName = fs.readdirSync(packageRoot)
  .filter((name) => name.startsWith('OpenAI.Codex_'))
  .sort().reverse()[0];
if (!packageName) throw new Error('OpenAI Codex package was not found.');
const asar = path.join(packageRoot, packageName, 'app', 'resources', 'app.asar');
const projectRoot = path.resolve(__dirname, '..');
const backupRoot = path.join(projectRoot, 'work', 'vendor-backups');
const manifest = path.join(backupRoot, 'codex-host-wham.json');
const needle = Buffer.from('enabled:!0,placeholderData:i,queryFn:async()=>{try{return(await Ae.safeGet(`/wham/tasks/list`,{parameters:{query:{limit:20,task_filter:`current`}}})).items');
const replacement = Buffer.from(needle.toString().replace('enabled:!0,', 'enabled:!1,'));

function hash(file) {
  const h = crypto.createHash('sha256');
  const fd = fs.openSync(file, 'r');
  const buf = Buffer.allocUnsafe(1024 * 1024);
  try { let n; while ((n = fs.readSync(fd, buf, 0, buf.length, null)) > 0) h.update(buf.subarray(0, n)); }
  finally { fs.closeSync(fd); }
  return h.digest('hex');
}

function find(file, pattern) {
  const fd = fs.openSync(file, 'r');
  const chunkSize = 1024 * 1024;
  let carry = Buffer.alloc(0);
  let base = 0;
  try {
    while (true) {
      const chunk = Buffer.allocUnsafe(chunkSize);
      const n = fs.readSync(fd, chunk, 0, chunk.length, null);
      if (!n) return -1;
      const data = Buffer.concat([carry, chunk.subarray(0, n)]);
      const at = data.indexOf(pattern);
      if (at >= 0) return base - carry.length + at;
      carry = data.subarray(Math.max(0, data.length - pattern.length + 1));
      base += n;
    }
  } finally { fs.closeSync(fd); }
}

function output(value) { process.stdout.write(`${JSON.stringify(value)}\n`); }
const mode = process.argv[2] || 'patch';
if (mode === 'rollback') {
  if (!fs.existsSync(manifest)) throw new Error(`Rollback manifest not found: ${manifest}`);
  const record = JSON.parse(fs.readFileSync(manifest, 'utf8'));
  if (record.asar !== asar || !fs.existsSync(record.backup)) throw new Error('Rollback manifest does not match the installed package.');
  fs.copyFileSync(record.backup, asar);
  output({ status: 'ROLLED_BACK', asar, backup: record.backup });
  process.exit(0);
}

const originalAt = find(asar, needle);
const patchedAt = find(asar, replacement);
if (mode === 'verify') {
  output({ status: originalAt >= 0 ? 'UNPATCHED' : 'PATCHED', package: packageName, asar,
    sha256: hash(asar), originalFingerprint: originalAt >= 0, patchedFingerprint: patchedAt >= 0 });
  process.exit(0);
}
if (originalAt < 0 && patchedAt >= 0) { output({ status: 'ALREADY_PATCHED', asar, sha256: hash(asar) }); process.exit(0); }
if (originalAt < 0 || patchedAt >= 0) throw new Error('Unexpected mixed patch state; refusing to modify the archive.');

fs.mkdirSync(backupRoot, { recursive: true });
const beforeSha256 = hash(asar);
const backup = path.join(backupRoot, `app.asar.${new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 15)}.${beforeSha256}.bak`);
fs.copyFileSync(asar, backup);
const fd = fs.openSync(asar, 'r+');
try { fs.writeSync(fd, replacement, 0, replacement.length, originalAt); } finally { fs.closeSync(fd); }
const afterSha256 = hash(asar);
if (find(asar, needle) >= 0 || find(asar, replacement) < 0) { fs.copyFileSync(backup, asar); throw new Error('Post-write verification failed; original archive restored.'); }
fs.writeFileSync(manifest, JSON.stringify({ package: packageName, asar, backup, beforeSha256, afterSha256, behavior: 'Disable only host current-tasks background query.' }, null, 2));
output({ status: 'PATCHED', package: packageName, asar, backup, beforeSha256, afterSha256, changedBytes: replacement.length });
