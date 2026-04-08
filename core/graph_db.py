# core/graph_db.py
import kuzu
import os
import gc
import csv
import shutil
import tempfile

# Known Godot built-in types for INHERITS edges
_GODOT_BUILTINS = {
    "Node", "Node2D", "Node3D", "Control", "CanvasItem",
    "CharacterBody2D", "CharacterBody3D", "RigidBody2D", "RigidBody3D",
    "StaticBody2D", "StaticBody3D", "Area2D", "Area3D",
    "Sprite2D", "Sprite3D", "AnimatedSprite2D", "AnimatedSprite3D",
    "Camera2D", "Camera3D", "Light2D", "DirectionalLight3D",
    "AudioStreamPlayer", "AudioStreamPlayer2D", "AudioStreamPlayer3D",
    "Timer", "HTTPRequest", "SubViewport",
    "Resource", "RefCounted", "Object",
    "TileMap", "TileMapLayer", "NavigationAgent2D", "NavigationAgent3D",
    "RayCast2D", "RayCast3D", "CollisionShape2D", "CollisionShape3D",
    "AnimationPlayer", "AnimationTree", "GPUParticles2D", "GPUParticles3D",
    "Label", "Button", "TextureRect", "Panel", "MarginContainer",
    "HBoxContainer", "VBoxContainer", "GridContainer",
    "LineEdit", "RichTextLabel", "ProgressBar",
    "Path2D", "PathFollow2D", "Path3D", "PathFollow3D",
}


class ScoutGraph:
    def __init__(self, project_id, scout_dir, graph_name="graph_v1", read_only=False):
        safe_name = "".join([c for c in graph_name if c.isalnum() or c in ('_', '-')])
        self.db_path = os.path.join(scout_dir, f"graph_{safe_name}")

        os.makedirs(scout_dir, exist_ok=True)

        if not read_only and os.path.exists(self.db_path) and os.path.isdir(self.db_path):
            if not os.listdir(self.db_path): os.rmdir(self.db_path)

        # Kuzu cannot open a non-existent DB in read_only mode.
        # Fall back to writable mode to create the empty DB + schema.
        db_exists = (os.path.exists(self.db_path) and os.path.isdir(self.db_path)
                     and len(os.listdir(self.db_path)) > 0)
        effective_read_only = read_only and db_exists

        self.db = kuzu.Database(self.db_path, read_only=effective_read_only)
        self.conn = kuzu.Connection(self.db)

        if not effective_read_only:
            self._init_schema()

    def close(self):
        """Forcefully annihilate the C++ connection pointers to release the .wal lock"""
        if hasattr(self, 'conn'): del self.conn
        if hasattr(self, 'db'): del self.db
        gc.collect()

    def _init_schema(self):
        statements = [
            # Original Code Nodes
            "CREATE NODE TABLE Script(path STRING, PRIMARY KEY (path))",
            "CREATE NODE TABLE Topic(name STRING, PRIMARY KEY (name))",
            "CREATE NODE TABLE Function(id STRING, name STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE Signal(name STRING, PRIMARY KEY (name))",

            # Data & Architecture Nodes
            "CREATE NODE TABLE Document(path STRING, PRIMARY KEY (path))",
            "CREATE NODE TABLE Scene(path STRING, PRIMARY KEY (path))",
            "CREATE NODE TABLE DataChunk(id STRING, name STRING, PRIMARY KEY (id))",

            # Phase 1: Variable tracking + Typed inheritance
            "CREATE NODE TABLE Variable(id STRING, name STRING, kind STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE Class(name STRING, is_builtin BOOL, PRIMARY KEY (name))",

            # Original Relationships
            "CREATE REL TABLE IMPLEMENTS(FROM Script TO Topic)",
            "CREATE REL TABLE OWNS(FROM Script TO Function)",
            "CREATE REL TABLE DEFINES(FROM Script TO Signal)",
            "CREATE REL TABLE FIRES(FROM Function TO Signal)",
            "CREATE REL TABLE CONTAINS_DOC(FROM Document TO DataChunk)",
            "CREATE REL TABLE CONTAINS_SCENE(FROM Scene TO DataChunk)",
            "CREATE REL TABLE EXTENDS(FROM Script TO Topic)",
            "CREATE REL TABLE CALLS(FROM Script TO Script)",

            # Phase 1: Variable read/write tracking
            "CREATE REL TABLE DECLARES(FROM Script TO Variable)",
            "CREATE REL TABLE READS(FROM Function TO Variable)",
            "CREATE REL TABLE WRITES(FROM Function TO Variable)",

            # Phase 1: Scene tree hierarchy + script bindings
            "CREATE REL TABLE PARENT_OF(FROM DataChunk TO DataChunk)",
            "CREATE REL TABLE ATTACHED_TO(FROM Script TO DataChunk)",

            # Phase 1: Typed inheritance
            "CREATE REL TABLE INHERITS(FROM Script TO Class)",
        ]
        for stmt in statements:
            try: self.conn.execute(stmt)
            except Exception as e:
                if "already exists" not in str(e).lower(): print(e)

    # ------------------------------------------------------------------
    #  BULK SYNC — collect everything, deduplicate, COPY FROM CSV
    # ------------------------------------------------------------------
    def bulk_sync(self, parsed_map):
        """Index an entire project at once via CSV bulk load.

        Args:
            parsed_map: dict of {rel_path: parsed_data} from the parser.
        """
        # Collect all unique nodes and relationships
        scripts = set()
        topics = set()
        functions = {}       # id -> name
        signals = set()
        documents = set()
        scenes = set()
        datachunks = {}      # id -> name

        # Phase 1: new node collections
        variables = {}           # id -> {"name": str, "kind": str}
        classes = {}             # name -> {"is_builtin": bool}

        rel_implements = set()   # (script_path, topic_name)
        rel_owns = set()         # (script_path, function_id)
        rel_defines = set()      # (script_path, signal_name)
        rel_fires = set()        # (function_id, signal_name)
        rel_contains_doc = set() # (doc_path, chunk_id)
        rel_contains_scene = set() # (scene_path, chunk_id)
        rel_extends = set()      # (script_path, parent_topic_name)
        rel_calls = set()        # (from_script_path, to_script_path)

        # Phase 1: new relationship collections
        rel_declares = set()     # (script_path, var_id)
        rel_reads = set()        # (function_id, var_id)
        rel_writes = set()       # (function_id, var_id)
        rel_parent_of = set()    # (parent_chunk_id, child_chunk_id)
        rel_attached_to = set()  # (script_path, chunk_id)
        rel_inherits = set()     # (script_path, class_name)

        # Store cross_refs for second-pass CALLS resolution (needs full scripts set)
        script_cross_refs = {}   # rel_path -> list of raw ref strings

        for rel_path, parsed_data in parsed_map.items():
            ext = os.path.splitext(rel_path)[1].lower()
            chunks = parsed_data.get('chunks', parsed_data.get('units', []))

            if ext in ['.gd', '.py', '.cs']:
                topic_name = parsed_data.get('topic', 'Unnamed Script')
                scripts.add(rel_path)
                topics.add(topic_name)
                rel_implements.add((rel_path, topic_name))

                # EXTENDS: inheritance edge (Script → parent Topic)
                extends_parent = parsed_data.get('extends', '')
                if extends_parent:
                    topics.add(extends_parent)
                    rel_extends.add((rel_path, extends_parent))

                # Stash cross_refs for second pass
                cross_refs = parsed_data.get('cross_refs', [])
                if cross_refs:
                    script_cross_refs[rel_path] = cross_refs

                for sig in parsed_data.get('signals', []):
                    signals.add(sig)
                    rel_defines.add((rel_path, sig))

                # Phase 1: Variable nodes + DECLARES
                for var in parsed_data.get('variables', []):
                    var_id = f"{rel_path}::{var['name']}"
                    variables[var_id] = {"name": var["name"], "kind": var["kind"]}
                    rel_declares.add((rel_path, var_id))

                # Phase 1: Class node + INHERITS (typed inheritance)
                if extends_parent:
                    is_builtin = extends_parent in _GODOT_BUILTINS
                    classes[extends_parent] = {"is_builtin": is_builtin}
                    rel_inherits.add((rel_path, extends_parent))
                user_class = parsed_data.get('class_name', '')
                if user_class:
                    classes[user_class] = {"is_builtin": False}

                for fn in chunks:
                    fn_id = f"{rel_path}::{fn['name']}"
                    functions[fn_id] = fn['name']
                    rel_owns.add((rel_path, fn_id))
                    for emitted in fn.get('emits', []):
                        signals.add(emitted)
                        rel_fires.add((fn_id, emitted))
                    # Phase 1: READS/WRITES edges
                    for vname in fn.get('reads_vars', []):
                        var_id = f"{rel_path}::{vname}"
                        if var_id in variables:
                            rel_reads.add((fn_id, var_id))
                    for vname in fn.get('writes_vars', []):
                        var_id = f"{rel_path}::{vname}"
                        if var_id in variables:
                            rel_writes.add((fn_id, var_id))

            elif ext in ['.md', '.json']:
                documents.add(rel_path)
                for unit in chunks:
                    cid = f"{rel_path}::{unit['name']}"
                    datachunks[cid] = unit['name']
                    rel_contains_doc.add((rel_path, cid))

            elif ext == '.tscn':
                scenes.add(rel_path)
                for unit in chunks:
                    cid = f"{rel_path}::{unit['name']}"
                    datachunks[cid] = unit['name']
                    rel_contains_scene.add((rel_path, cid))

                # Phase 1: Scene tree hierarchy + script bindings
                scene_nodes_data = parsed_data.get('scene_nodes', [])
                path_to_cid = {}
                for sn in scene_nodes_data:
                    cid = f"{rel_path}::node_{sn['node_path']}"
                    datachunks[cid] = sn['name']
                    rel_contains_scene.add((rel_path, cid))
                    path_to_cid[sn['node_path']] = cid

                for sn in scene_nodes_data:
                    child_cid = path_to_cid.get(sn['node_path'])
                    if not child_cid or sn['parent_path'] == "":
                        continue
                    if sn['parent_path'] == ".":
                        parent_cid = path_to_cid.get(".")
                    else:
                        parent_cid = path_to_cid.get(sn['parent_path'])
                    if parent_cid and parent_cid != child_cid:
                        rel_parent_of.add((parent_cid, child_cid))

                for sn in scene_nodes_data:
                    if sn['script_path']:
                        script_rel = sn['script_path'].replace("\\", "/")
                        chunk_id = path_to_cid.get(sn['node_path'])
                        if chunk_id:
                            scripts.add(script_rel)
                            rel_attached_to.add((script_rel, chunk_id))

        # Second pass: resolve CALLS by matching cross_refs to known script paths.
        # Reliable matches: preload paths (path suffix), stem name (dollar/static refs).
        for from_path, cross_refs in script_cross_refs.items():
            for ref in cross_refs:
                ref_lower = ref.lower().replace("\\", "/")
                for to_path in scripts:
                    if to_path == from_path:
                        continue
                    to_norm = to_path.replace("\\", "/")
                    to_base = os.path.basename(to_norm)
                    to_stem = os.path.splitext(to_base)[0].lower()
                    # Match: exact path suffix (preload), filename, or stem (class/node ref)
                    if (to_norm.endswith(ref_lower) or
                            to_base.lower() == ref_lower or
                            to_stem == ref_lower):
                        rel_calls.add((from_path, to_path))
                        break  # one match per ref is sufficient

        # Write CSVs and COPY FROM
        tmp = tempfile.mkdtemp(prefix="scout_csv_")
        try:
            self._bulk_load(tmp, scripts, topics, functions, signals,
                            documents, scenes, datachunks,
                            rel_implements, rel_owns, rel_defines, rel_fires,
                            rel_contains_doc, rel_contains_scene,
                            rel_extends, rel_calls,
                            variables=variables, classes=classes,
                            rel_declares=rel_declares, rel_reads=rel_reads,
                            rel_writes=rel_writes, rel_parent_of=rel_parent_of,
                            rel_attached_to=rel_attached_to, rel_inherits=rel_inherits)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @staticmethod
    def _sanitize_csv_value(val):
        """Strip characters that break Kuzu's CSV parser."""
        if not isinstance(val, str):
            return val
        # Replace problematic characters: raw quotes, newlines, carriage returns
        return val.replace('"', "'").replace('\n', ' ').replace('\r', '')

    def _write_csv(self, tmp_dir, name, rows):
        """Write rows to a CSV file. Returns the forward-slash path for Kuzu."""
        path = os.path.join(tmp_dir, f"{name}.csv")
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            for row in rows:
                writer.writerow([self._sanitize_csv_value(v) for v in row])
        return path.replace("\\", "/")

    def _copy_if(self, table, csv_path, header=False):
        """Run COPY FROM if the CSV is non-empty."""
        if os.path.getsize(csv_path.replace("/", os.sep)) > 0:
            self.conn.execute(f'COPY {table} FROM "{csv_path}" (HEADER=false)')

    def _bulk_load(self, tmp, scripts, topics, functions, signals,
                   documents, scenes, datachunks,
                   rel_implements, rel_owns, rel_defines, rel_fires,
                   rel_contains_doc, rel_contains_scene,
                   rel_extends=None, rel_calls=None,
                   variables=None, classes=None,
                   rel_declares=None, rel_reads=None, rel_writes=None,
                   rel_parent_of=None, rel_attached_to=None, rel_inherits=None):
        """Write CSVs and execute COPY FROM for each table."""
        exe = self.conn.execute

        # --- Node CSVs ---
        if scripts:
            p = self._write_csv(tmp, 'scripts', [[s] for s in scripts])
            self._copy_if('Script', p)
        if topics:
            p = self._write_csv(tmp, 'topics', [[t] for t in topics])
            self._copy_if('Topic', p)
        if functions:
            p = self._write_csv(tmp, 'functions', [[fid, fname] for fid, fname in functions.items()])
            self._copy_if('Function', p)
        if signals:
            p = self._write_csv(tmp, 'signals', [[s] for s in signals])
            self._copy_if('Signal', p)
        if documents:
            p = self._write_csv(tmp, 'documents', [[d] for d in documents])
            self._copy_if('Document', p)
        if scenes:
            p = self._write_csv(tmp, 'scenes', [[s] for s in scenes])
            self._copy_if('Scene', p)
        if datachunks:
            p = self._write_csv(tmp, 'datachunks', [[cid, cname] for cid, cname in datachunks.items()])
            self._copy_if('DataChunk', p)
        # Phase 1: Variable + Class nodes
        if variables:
            p = self._write_csv(tmp, 'variables', [[vid, vd["name"], vd["kind"]] for vid, vd in variables.items()])
            self._copy_if('Variable', p)
        if classes:
            p = self._write_csv(tmp, 'classes', [[name, str(data["is_builtin"])] for name, data in classes.items()])
            self._copy_if('Class', p)

        # --- Relationship CSVs ---
        if rel_implements:
            p = self._write_csv(tmp, 'implements', list(rel_implements))
            self._copy_if('IMPLEMENTS', p)
        if rel_owns:
            p = self._write_csv(tmp, 'owns', list(rel_owns))
            self._copy_if('OWNS', p)
        if rel_defines:
            p = self._write_csv(tmp, 'defines', list(rel_defines))
            self._copy_if('DEFINES', p)
        if rel_fires:
            p = self._write_csv(tmp, 'fires', list(rel_fires))
            self._copy_if('FIRES', p)
        if rel_contains_doc:
            p = self._write_csv(tmp, 'contains_doc', list(rel_contains_doc))
            self._copy_if('CONTAINS_DOC', p)
        if rel_contains_scene:
            p = self._write_csv(tmp, 'contains_scene', list(rel_contains_scene))
            self._copy_if('CONTAINS_SCENE', p)
        if rel_extends:
            p = self._write_csv(tmp, 'extends', list(rel_extends))
            self._copy_if('EXTENDS', p)
        if rel_calls:
            p = self._write_csv(tmp, 'calls', list(rel_calls))
            self._copy_if('CALLS', p)
        # Phase 1: Variable tracking
        if rel_declares:
            p = self._write_csv(tmp, 'declares', list(rel_declares))
            self._copy_if('DECLARES', p)
        if rel_reads:
            p = self._write_csv(tmp, 'reads', list(rel_reads))
            self._copy_if('READS', p)
        if rel_writes:
            p = self._write_csv(tmp, 'writes', list(rel_writes))
            self._copy_if('WRITES', p)
        # Phase 1: Scene hierarchy
        if rel_parent_of:
            p = self._write_csv(tmp, 'parent_of', list(rel_parent_of))
            self._copy_if('PARENT_OF', p)
        if rel_attached_to:
            p = self._write_csv(tmp, 'attached_to', list(rel_attached_to))
            self._copy_if('ATTACHED_TO', p)
        # Phase 1: Typed inheritance
        if rel_inherits:
            p = self._write_csv(tmp, 'inherits', list(rel_inherits))
            self._copy_if('INHERITS', p)

    # ------------------------------------------------------------------
    #  Legacy per-file sync (kept for compatibility / read-modify use)
    # ------------------------------------------------------------------
    def sync_script(self, rel_path, parsed_data):
        """Routes parsed data into Kuzu one file at a time (legacy, slower)."""
        ext = os.path.splitext(rel_path)[1].lower()
        chunks = parsed_data.get('chunks', parsed_data.get('units', []))
        exe = self.conn.execute

        if ext in ['.gd', '.py', '.cs']:
            topic_name = parsed_data.get('topic', 'Unnamed Script')
            exe("MERGE (s:Script {path: $path})", {"path": rel_path})
            exe("MERGE (t:Topic {name: $name})", {"name": topic_name})
            exe("MATCH (s:Script {path: $p}), (t:Topic {name: $n}) MERGE (s)-[:IMPLEMENTS]->(t)",
                {"p": rel_path, "n": topic_name})

            for sig in parsed_data.get('signals', []):
                exe("MERGE (sig:Signal {name: $name})", {"name": sig})
                exe("MATCH (s:Script {path: $p}), (sig:Signal {name: $n}) MERGE (s)-[:DEFINES]->(sig)",
                    {"p": rel_path, "n": sig})

            for fn in chunks:
                fn_id = f"{rel_path}::{fn['name']}"
                exe("MERGE (f:Function {id: $id, name: $name})", {"id": fn_id, "name": fn['name']})
                exe("MATCH (s:Script {path: $p}), (f:Function {id: $id}) MERGE (s)-[:OWNS]->(f)",
                    {"p": rel_path, "id": fn_id})
                for emitted in fn.get('emits', []):
                    exe("MERGE (sig:Signal {name: $name})", {"name": emitted})
                    exe("MATCH (f:Function {id: $id}), (sig:Signal {name: $n}) MERGE (f)-[:FIRES]->(sig)",
                        {"id": fn_id, "n": emitted})

        elif ext in ['.md', '.json']:
            exe("MERGE (d:Document {path: $path})", {"path": rel_path})
            for unit in chunks:
                cid = f"{rel_path}::{unit['name']}"
                exe("MERGE (c:DataChunk {id: $id, name: $name})", {"id": cid, "name": unit['name']})
                exe("MATCH (d:Document {path: $p}), (c:DataChunk {id: $id}) MERGE (d)-[:CONTAINS_DOC]->(c)",
                    {"p": rel_path, "id": cid})

        elif ext == '.tscn':
            exe("MERGE (sc:Scene {path: $path})", {"path": rel_path})
            for unit in chunks:
                cid = f"{rel_path}::{unit['name']}"
                exe("MERGE (c:DataChunk {id: $id, name: $name})", {"id": cid, "name": unit['name']})
                exe("MATCH (sc:Scene {path: $p}), (c:DataChunk {id: $id}) MERGE (sc)-[:CONTAINS_SCENE]->(c)",
                    {"p": rel_path, "id": cid})

    def propagate_systems(self, parsed_map):
        """Propagate system tags through signal chains in the graph.

        If script A fires signal S and script B defines signal S, both scripts
        share their system tags. This gives graph-derived system membership
        instead of relying purely on keyword heuristics.

        Args:
            parsed_map: dict of {rel_path: parsed_data} — same as bulk_sync input.

        Returns:
            dict of {rel_path: set of system_names} — enriched system assignments.
        """
        from collections import defaultdict

        # Collect initial keyword-based systems per script
        script_systems = defaultdict(set)
        for rel_path, parsed in parsed_map.items():
            for chunk in parsed.get('chunks', []):
                for sys_name in chunk.get('systems', []):
                    script_systems[rel_path].add(sys_name)

        # Build signal connectivity from graph: who defines and who fires each signal
        signal_scripts = defaultdict(set)  # signal_name → set of script_paths
        try:
            res = self.conn.execute(
                "MATCH (s:Script)-[:DEFINES]->(sig:Signal) RETURN s.path, sig.name"
            )
            while res.has_next():
                sp, sn = res.get_next()
                signal_scripts[sn].add(sp)
        except Exception:
            pass
        try:
            res = self.conn.execute(
                "MATCH (s:Script)-[:OWNS]->(f:Function)-[:FIRES]->(sig:Signal) "
                "RETURN s.path, sig.name"
            )
            while res.has_next():
                sp, sn = res.get_next()
                signal_scripts[sn].add(sp)
        except Exception:
            pass

        # Propagate: scripts connected via the same signal share system tags
        for sig_name, connected_scripts in signal_scripts.items():
            # Union all systems from all scripts connected via this signal
            shared_systems = set()
            for sp in connected_scripts:
                shared_systems.update(script_systems.get(sp, set()))
            # Propagate back
            for sp in connected_scripts:
                script_systems[sp].update(shared_systems)

        return dict(script_systems)

    @staticmethod
    def wipe_database(scout_dir, graph_name="graph_v1"):
        import time
        safe_name = "".join([c for c in graph_name if c.isalnum() or c in ('_', '-')])
        db_path = os.path.join(scout_dir, f"graph_{safe_name}")

        gc.collect()
        time.sleep(0.5) # Give the OS a heartbeat to drop the lock

        if os.path.exists(db_path):
            # Try 3 times to kill it (Windows is stubborn with locks)
            for _ in range(3):
                try:
                    shutil.rmtree(db_path, ignore_errors=True)
                    if not os.path.exists(db_path): break
                    time.sleep(0.5)
                except: pass
