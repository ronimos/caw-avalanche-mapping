<?php
// Save endpoint for the model-location editor (focus_map_editor.js).
// Copied next to focus_stations_map.html by scripts/focus_map.py.
//
//   GET   → the current model_locations.json (404 when nothing saved yet)
//   POST  → validate, and if the selections changed: copy the previous file to
//           backups/model_locations_<UTC timestamp>.json, then write the new one.
//
// Files live in ./model_locations/, which must be writable by the web server
// user (setup: see the focus_map.py docstring).
declare(strict_types=1);
header('Content-Type: application/json');
header('Cache-Control: no-store');

const SCHEMA  = 'caw-model-locations/1';
const ASPECTS = ['Flat', 'N', 'E', 'S', 'W'];
const MAX_BODY = 2 * 1024 * 1024;

$dir    = __DIR__ . '/model_locations';
$file   = $dir . '/model_locations.json';
$backup = $dir . '/backups';

function fail(int $code, string $msg): void {
    http_response_code($code);
    echo json_encode(['status' => 'error', 'error' => $msg]);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] === 'GET') {
    if (!is_file($file)) fail(404, 'nothing saved yet');
    readfile($file);
    exit;
}
if ($_SERVER['REQUEST_METHOD'] !== 'POST') fail(405, 'GET or POST only');

$body = file_get_contents('php://input', false, null, 0, MAX_BODY + 1);
if ($body === false || strlen($body) > MAX_BODY) fail(413, 'request too large');
$doc = json_decode($body, true);
if (!is_array($doc) || ($doc['schema'] ?? null) !== SCHEMA || !is_array($doc['selections'] ?? null)) {
    fail(400, 'not a ' . SCHEMA . ' document');
}

// Keep only known fields, with checked types — the file feeds sync_pro_files.py.
$clean = [];
foreach ($doc['selections'] as $i => $r) {
    if (!is_array($r)) fail(400, "selection $i is not an object");
    $sid = (string)($r['station'] ?? '');
    if (!preg_match('/^\d{1,9}$/', $sid)) fail(400, "selection $i: bad station id");
    $action = $r['action'] ?? '';
    if ($action !== 'add' && $action !== 'remove') fail(400, "station $sid: action must be add or remove");
    $aspects = $r['aspects'] ?? null;
    if (!is_array($aspects) || !$aspects || array_diff($aspects, ASPECTS)) fail(400, "station $sid: bad aspects");
    $clean[$sid] = [
        'station'   => $sid,
        'action'    => $action,
        'aspects'   => array_values(array_intersect(ASPECTS, $aspects)),
        'note'      => mb_substr((string)($r['note'] ?? ''), 0, 500),
        'source'    => mb_substr((string)($r['source'] ?? ''), 0, 200),
        'lat'       => (float)($r['lat'] ?? 0),
        'lon'       => (float)($r['lon'] ?? 0),
        'elevation' => (float)($r['elevation'] ?? 0),
        'updated'   => mb_substr((string)($r['updated'] ?? ''), 0, 40),
    ];
}
ksort($clean, SORT_NATURAL);
$selections = array_values($clean);

if (!is_dir($dir) || !is_writable($dir)) fail(500, 'model_locations/ is missing or not writable by the web server');

$lock = fopen($dir . '/.lock', 'c');
if (!$lock || !flock($lock, LOCK_EX)) fail(500, 'could not lock');

$prev = is_file($file) ? json_decode((string)file_get_contents($file), true) : null;
if (is_array($prev) && ($prev['selections'] ?? null) == $selections) {
    echo json_encode(['status' => 'unchanged', 'saved' => $prev['saved'] ?? null]);
    exit;
}

$now = gmdate('Y-m-d\TH:i:s\Z');
$backupName = null;
if (is_array($prev)) {
    if (!is_dir($backup) && !mkdir($backup, 0775)) fail(500, 'could not create backups/');
    $backupName = 'model_locations_' . gmdate('Ymd-His') . '.json';
    if (!copy($file, "$backup/$backupName")) fail(500, 'backup failed — not saved');
}

$out = [
    'schema'     => SCHEMA,
    'season'     => (string)($doc['season'] ?? ''),
    'run_dir'    => (string)($doc['run_dir'] ?? ''),
    'baseline'   => (string)($doc['baseline'] ?? ''),
    'aspects'    => ['Flat' => '', 'N' => '1', 'E' => '2', 'S' => '3', 'W' => '4'],
    'saved'      => $now,
    'saved_from' => $_SERVER['REMOTE_ADDR'] ?? '',
    'selections' => $selections,
];
$tmp = "$file.tmp";
if (file_put_contents($tmp, json_encode($out, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE) . "\n") === false
    || !rename($tmp, $file)) {
    fail(500, 'write failed');
}
echo json_encode(['status' => 'saved', 'saved' => $now, 'backup' => $backupName, 'count' => count($selections)]);
