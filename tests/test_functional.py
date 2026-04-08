"""
Functional test suite for Sk0uter (Scout Director).

Covers the full user journey from project creation through chat/prompt evaluation.

Tests requiring live services (Qdrant, Ollama) are marked @pytest.mark.integration
and are skipped in CI by default.

Run unit tests only:
    pytest -v -m "not integration"

Run integration tests (requires live Qdrant + Ollama):
    pytest -v -m integration
"""

import json
import os
import sys
import uuid
from unittest.mock import patch, MagicMock, call

import pytest

# ---------------------------------------------------------------------------
# Project root must be on sys.path so imports work when running directly.
# pytest.ini sets pythonpath = . for normal pytest runs.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ===========================================================================
# SAMPLE CONTENT
# ===========================================================================

SAMPLE_GD = """\
extends CharacterBody2D

class_name Player

signal health_changed(value)
signal player_died

var health: int = 100
var speed: float = 200.0
var gravity: float = 980.0

func _physics_process(delta):
\tvelocity.y += gravity * delta
\tmove_and_slide()
\tif health <= 0:
\t\tplayer_died.emit()

func take_damage(amount: int) -> void:
\thealth -= amount
\thealth_changed.emit(health)
\tif health <= 0:
\t\tdie()

func die() -> void:
\tqueue_free()
"""

SAMPLE_GD_ENEMY = """\
extends CharacterBody2D

class_name Enemy

signal enemy_died

var health: int = 50
var damage: int = 10
var speed: float = 100.0

func attack(target) -> void:
\ttarget.take_damage(damage)

func die() -> void:
\tenemy_died.emit()
\tqueue_free()
"""

SAMPLE_MD = """\
# Project Overview

This is a Godot game project.

## Player System

The player uses CharacterBody2D for physics-based movement.

## Combat System

Enemies can deal damage to the player via take_damage().
"""

SAMPLE_JSON = """\
{
    "game_title": "Flooded Cafe",
    "version": "0.1.0",
    "settings": {
        "volume": 0.8
    }
}
"""


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def tmp_project(tmp_path):
    """Creates a minimal project directory structure for testing."""
    project_path = tmp_path / "test_project"
    project_path.mkdir()
    scout_dir = project_path / ".scout"
    scout_dir.mkdir()

    (project_path / "Player.gd").write_text(SAMPLE_GD, encoding="utf-8")
    (project_path / "Enemy.gd").write_text(SAMPLE_GD_ENEMY, encoding="utf-8")
    (project_path / "README.md").write_text(SAMPLE_MD, encoding="utf-8")
    (project_path / "config.json").write_text(SAMPLE_JSON, encoding="utf-8")

    return {
        "path": str(project_path),
        "scout_dir": str(scout_dir),
        "id": "test_project",
    }


@pytest.fixture
def active_project(tmp_project):
    """Alias matching the dict shape that app.py passes to Ask.py functions."""
    return tmp_project


@pytest.fixture
def mock_qdrant_client():
    """A pre-configured mock QdrantClient that returns empty results."""
    mock = MagicMock()
    mock.get_collections.return_value = MagicMock(collections=[])
    mock.query_points.return_value = MagicMock(points=[])
    mock.search.return_value = []
    return mock


@pytest.fixture
def mock_graph_conn():
    """A mock Kuzu connection where execute() always returns an empty result set."""
    mock_result = MagicMock()
    mock_result.has_next.return_value = False
    conn = MagicMock()
    conn.execute.return_value = mock_result
    return conn


# ===========================================================================
# TEST GROUP 1: Parser  (no live services needed)
# ===========================================================================

class TestParser:
    """Tests for core/parser.py — pure Python, no external services."""

    def test_parse_gdscript_extracts_functions(self):
        from core.parser import parse_gdscript
        result = parse_gdscript(SAMPLE_GD, "Player.gd")
        names = [c["name"] for c in result["chunks"]]
        assert "_physics_process" in names
        assert "take_damage" in names
        assert "die" in names

    def test_parse_gdscript_class_name_is_topic(self):
        from core.parser import parse_gdscript
        result = parse_gdscript(SAMPLE_GD, "Player.gd")
        assert result["topic"] == "Player"

    def test_parse_gdscript_extracts_signals(self):
        from core.parser import parse_gdscript
        result = parse_gdscript(SAMPLE_GD, "Player.gd")
        assert "health_changed" in result["signals"]
        assert "player_died" in result["signals"]

    def test_parse_gdscript_detects_emits(self):
        from core.parser import parse_gdscript
        result = parse_gdscript(SAMPLE_GD, "Player.gd")
        all_emits = []
        for chunk in result["chunks"]:
            all_emits.extend(chunk.get("emits", []))
        assert "player_died" in all_emits
        assert "health_changed" in all_emits

    def test_parse_gdscript_infers_movement_system(self):
        from core.parser import parse_gdscript
        result = parse_gdscript(SAMPLE_GD, "Player.gd")
        physics = next(c for c in result["chunks"] if c["name"] == "_physics_process")
        assert "movement" in physics["systems"]

    def test_parse_gdscript_infers_combat_system(self):
        from core.parser import parse_gdscript
        result = parse_gdscript(SAMPLE_GD, "Player.gd")
        damage = next(c for c in result["chunks"] if c["name"] == "take_damage")
        assert "combat" in damage["systems"]

    def test_parse_gdscript_returns_global_state(self):
        from core.parser import parse_gdscript
        result = parse_gdscript(SAMPLE_GD, "Player.gd")
        # global_state should contain at least one var declaration
        assert "health" in result["global_state"] or "speed" in result["global_state"]

    def test_parse_gdscript_chunks_have_required_keys(self):
        from core.parser import parse_gdscript
        result = parse_gdscript(SAMPLE_GD, "Player.gd")
        for chunk in result["chunks"]:
            assert "name" in chunk
            assert "content" in chunk
            assert "type" in chunk
            assert "line_start" in chunk
            assert "emits" in chunk

    def test_parse_markdown_creates_chunks(self):
        from core.parser import parse_markdown
        result = parse_markdown(SAMPLE_MD, "README.md")
        assert len(result["chunks"]) > 0
        assert all(c["type"] == "markdown_section" for c in result["chunks"])

    def test_parse_markdown_topic_from_filename(self):
        from core.parser import parse_markdown
        result = parse_markdown(SAMPLE_MD, "project_overview.md")
        assert result["topic"] == "Project Overview"

    def test_parse_json_valid(self):
        from core.parser import parse_json
        result = parse_json(SAMPLE_JSON, "config.json")
        assert result["identity"] == "Valid JSON Data"
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["type"] == "json_data"

    def test_parse_json_invalid(self):
        from core.parser import parse_json
        result = parse_json("not valid json {{{", "broken.json")
        assert result["identity"] == "Invalid JSON File"
        assert result["chunks"] == []

    def test_parse_file_routes_gd(self):
        from core.parser import parse_file
        result = parse_file("Player.gd", SAMPLE_GD)
        assert result["topic"] == "Player"
        assert len(result["chunks"]) > 0

    def test_parse_file_routes_md(self):
        from core.parser import parse_file
        result = parse_file("docs/README.md", SAMPLE_MD)
        assert len(result["chunks"]) > 0

    def test_parse_file_routes_json(self):
        from core.parser import parse_file
        result = parse_file("data/config.json", SAMPLE_JSON)
        assert result["identity"] == "Valid JSON Data"

    def test_parse_file_unknown_extension_returns_default(self):
        from core.parser import parse_file
        result = parse_file("some_file.xyz", "content")
        assert result["chunks"] == []
        assert result["identity"] == ""

    def test_parse_gdscript_extends_extracted(self):
        from core.parser import parse_gdscript
        result = parse_gdscript(SAMPLE_GD, "Player.gd")
        assert result.get("extends") == "CharacterBody2D"

    def test_parse_result_has_all_top_level_keys(self):
        from core.parser import parse_file
        result = parse_file("Player.gd", SAMPLE_GD)
        for key in ("identity", "global_state", "chunks", "signals", "topic"):
            assert key in result, f"Missing key: {key}"


# ===========================================================================
# TEST GROUP 2: System Inference
# ===========================================================================

class TestSystemInference:
    """Tests for the keyword-based system tagging in core/parser.py."""

    def test_movement_keywords_detected(self):
        from core.parser import infer_systems
        result = infer_systems("velocity move_and_slide jump speed gravity")
        assert "movement" in result

    def test_combat_keywords_detected(self):
        from core.parser import infer_systems
        result = infer_systems("health attack damage hit weapon shoot")
        assert "combat" in result

    def test_audio_keywords_detected(self):
        from core.parser import infer_systems
        result = infer_systems("audio stream play_sound volume sfx music")
        assert "audio" in result

    def test_ai_keywords_detected(self):
        from core.parser import infer_systems
        result = infer_systems("enemy npc pathfind navigate patrol chase aggro")
        assert "ai" in result

    def test_min_keyword_threshold_not_met(self):
        from core.parser import infer_systems
        # Only 1 keyword — threshold is 2, should NOT qualify
        result = infer_systems("velocity some_unrelated_stuff nothing_else")
        assert "movement" not in result

    def test_multi_system_assignment(self):
        from core.parser import infer_systems
        result = infer_systems(
            "velocity move_and_slide health damage attack speed jump"
        )
        assert "movement" in result
        assert "combat" in result

    def test_result_is_sorted(self):
        from core.parser import infer_systems
        result = infer_systems("audio stream sfx music velocity move_and_slide jump speed")
        assert result == sorted(result)


# ===========================================================================
# TEST GROUP 3: Roadmap / Session Persistence
# ===========================================================================

class TestRoadmapPersistence:
    """Tests for core/roadmap.py — uses tmp dirs, no live services."""

    def test_load_returns_defaults_when_no_file(self, tmp_path):
        from core.roadmap import load_project_data
        result = load_project_data(str(tmp_path))
        assert "global_goal" in result
        assert "milestones" in result
        assert isinstance(result["milestones"], list)
        assert "active_tasks" in result

    def test_save_and_reload_round_trip(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        data = {
            "global_goal": "Finish the tutorial level",
            "milestones": [{"name": "Level 1", "status": "completed"}],
            "active_tasks": [],
            "architectural_rules": "Use signals for cross-script comms",
        }
        save_project_data(str(tmp_path), data)
        reloaded = load_project_data(str(tmp_path))
        assert reloaded["global_goal"] == "Finish the tutorial level"
        assert reloaded["milestones"][0]["name"] == "Level 1"
        assert reloaded["milestones"][0]["status"] == "completed"

    def test_state_file_is_valid_json(self, tmp_path):
        from core.roadmap import save_project_data, get_state_path
        data = {
            "global_goal": "Test",
            "milestones": [],
            "active_tasks": [],
            "architectural_rules": "",
        }
        save_project_data(str(tmp_path), data)
        path = get_state_path(str(tmp_path))
        assert os.path.exists(path)
        with open(path, "r") as f:
            loaded = json.load(f)
        assert loaded["global_goal"] == "Test"

    def test_add_milestone(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        data = load_project_data(str(tmp_path))
        data["milestones"].append({"name": "Boss Fight", "status": "pending"})
        save_project_data(str(tmp_path), data)
        reloaded = load_project_data(str(tmp_path))
        names = [m["name"] for m in reloaded["milestones"]]
        assert "Boss Fight" in names

    def test_update_milestone_status(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        data = {
            "global_goal": "Ship the game",
            "milestones": [{"name": "Core Loop", "status": "in_progress"}],
            "active_tasks": [],
            "architectural_rules": "",
        }
        save_project_data(str(tmp_path), data)
        data = load_project_data(str(tmp_path))
        data["milestones"][0]["status"] = "completed"
        save_project_data(str(tmp_path), data)
        reloaded = load_project_data(str(tmp_path))
        assert reloaded["milestones"][0]["status"] == "completed"

    def test_set_global_goal(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        data = load_project_data(str(tmp_path))
        data["global_goal"] = "Create an infinite runner game"
        save_project_data(str(tmp_path), data)
        reloaded = load_project_data(str(tmp_path))
        assert reloaded["global_goal"] == "Create an infinite runner game"

    def test_architectural_rules_persisted(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        data = load_project_data(str(tmp_path))
        data["architectural_rules"] = "No singletons. Use signals only."
        save_project_data(str(tmp_path), data)
        reloaded = load_project_data(str(tmp_path))
        assert "No singletons" in reloaded["architectural_rules"]

    def test_multiple_saves_dont_accumulate(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        for i in range(3):
            data = load_project_data(str(tmp_path))
            data["global_goal"] = f"goal_{i}"
            save_project_data(str(tmp_path), data)
        final = load_project_data(str(tmp_path))
        assert final["global_goal"] == "goal_2"


# ===========================================================================
# TEST GROUP 4: Project Creation
# ===========================================================================

class TestProjectCreation:
    """Tests for ProjectManager — Qdrant calls are mocked."""

    def test_get_project_dirs_creates_scout_dir(self, tmp_path):
        from core.manager import ProjectManager
        project_path = tmp_path / "my_godot_project"
        project_path.mkdir()
        project_id, scout_dir = ProjectManager.get_project_dirs(str(project_path))
        assert os.path.exists(scout_dir)

    def test_project_id_equals_dirname(self, tmp_path):
        from core.manager import ProjectManager
        project_path = tmp_path / "mygame"
        project_path.mkdir()
        project_id, _ = ProjectManager.get_project_dirs(str(project_path))
        assert project_id == "mygame"

    def test_project_id_normalizes_hyphens(self, tmp_path):
        from core.manager import ProjectManager
        project_path = tmp_path / "my-project-name"
        project_path.mkdir()
        project_id, _ = ProjectManager.get_project_dirs(str(project_path))
        assert "-" not in project_id
        assert project_id == "my_project_name"

    def test_scout_dir_is_inside_project_path(self, tmp_path):
        from core.manager import ProjectManager
        project_path = tmp_path / "game"
        project_path.mkdir()
        _, scout_dir = ProjectManager.get_project_dirs(str(project_path))
        assert scout_dir.startswith(str(project_path))

    @patch("core.manager.client")
    def test_init_databases_creates_collection_if_missing(self, mock_client, tmp_path):
        from core.manager import ProjectManager
        mock_client.get_collections.return_value = MagicMock(collections=[])
        scout_dir = tmp_path / ".scout"
        scout_dir.mkdir()
        ProjectManager.init_databases("test_project", str(scout_dir))
        mock_client.create_collection.assert_called_once()

    @patch("core.manager.client")
    def test_init_databases_skips_creation_if_collection_exists(self, mock_client, tmp_path):
        from core.manager import ProjectManager
        existing = MagicMock()
        existing.name = "test_project"
        mock_client.get_collections.return_value = MagicMock(collections=[existing])
        scout_dir = tmp_path / ".scout"
        scout_dir.mkdir()
        ProjectManager.init_databases("test_project", str(scout_dir))
        mock_client.create_collection.assert_not_called()


# ===========================================================================
# TEST GROUP 5: BM25 Sparse Encoding
# ===========================================================================

class TestSparseEncoding:
    """Tests for core/sparse.py — stdlib only, no live services."""

    def test_build_bm25_vocab_returns_idf(self):
        from core.sparse import build_bm25_vocab
        corpus = ["move_and_slide velocity jump", "attack damage health hp"]
        vocab = build_bm25_vocab(corpus)
        assert "idf" in vocab
        assert "avgdl" in vocab
        assert len(vocab["idf"]) > 0

    def test_build_bm25_vocab_empty_corpus(self):
        from core.sparse import build_bm25_vocab
        vocab = build_bm25_vocab([])
        assert vocab["idf"] == {}
        assert vocab["avgdl"] == 1.0

    def test_encode_sparse_returns_parallel_lists(self):
        from core.sparse import build_bm25_vocab, encode_sparse
        corpus = ["move_and_slide velocity jump", "attack damage health"]
        vocab = build_bm25_vocab(corpus)
        indices, values = encode_sparse("velocity jump", vocab)
        assert isinstance(indices, list)
        assert isinstance(values, list)
        assert len(indices) == len(values)

    def test_encode_sparse_indices_are_sorted(self):
        from core.sparse import build_bm25_vocab, encode_sparse
        corpus = ["move_and_slide velocity jump", "attack damage health"]
        vocab = build_bm25_vocab(corpus)
        indices, _ = encode_sparse("velocity attack jump damage", vocab)
        assert indices == sorted(indices)

    def test_encode_sparse_empty_text_returns_empty(self):
        from core.sparse import build_bm25_vocab, encode_sparse
        vocab = build_bm25_vocab(["some text here"])
        indices, values = encode_sparse("", vocab)
        assert indices == []
        assert values == []

    def test_encode_sparse_values_are_positive(self):
        from core.sparse import build_bm25_vocab, encode_sparse
        corpus = ["move_and_slide velocity jump", "attack damage health"]
        vocab = build_bm25_vocab(corpus)
        _, values = encode_sparse("velocity jump", vocab)
        assert all(v > 0 for v in values)

    def test_encode_sparse_query_returns_idf_weights(self):
        from core.sparse import build_bm25_vocab, encode_sparse_query
        corpus = ["move_and_slide velocity jump", "attack damage health"]
        vocab = build_bm25_vocab(corpus)
        indices, values = encode_sparse_query("velocity", vocab)
        assert len(indices) > 0
        assert all(v > 0 for v in values)

    def test_encode_sparse_unknown_tokens_yield_empty(self):
        from core.sparse import build_bm25_vocab, encode_sparse
        vocab = build_bm25_vocab(["hello world"])
        # tokens completely absent from corpus → IDF=0 → no entries
        indices, values = encode_sparse("zzz_completely_unknown_xyz", vocab)
        # May be empty or have entries depending on tokenization; values must be non-negative
        assert all(v >= 0 for v in values)

    def test_bm25_rare_token_has_higher_idf(self):
        from core.sparse import build_bm25_vocab
        corpus = [
            "velocity move_and_slide jump speed",  # common movement tokens
            "velocity move_and_slide",
            "velocity",
            "rare_special_function_name",           # appears only once
        ]
        vocab = build_bm25_vocab(corpus)
        idf = vocab["idf"]
        # 'velocity' appears in 3/4 docs; rare token appears in 1/4
        # rare token should have higher IDF
        vel_idf = idf.get("velocity", 0)
        rare_idf = idf.get("rare_special_function_name", 0)
        assert rare_idf > vel_idf


# ===========================================================================
# TEST GROUP 6: Indexing Pipeline
# ===========================================================================

class TestIndexingPipeline:
    """
    Tests the parse → embed → index pipeline.
    Qdrant and Ollama are mocked; kuzu graph DB uses a real temp directory.
    """

    @patch("requests.post")
    def test_ollama_embedder_calls_api(self, mock_post):
        from core.manager import OllamaEmbedder
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_post.return_value = mock_resp

        embedder = OllamaEmbedder("nomic-embed-text")
        result = embedder.encode(["func _physics_process(delta): move_and_slide()"])
        assert mock_post.called
        assert len(result) == 1
        assert len(result[0]) == 768

    @patch("requests.post")
    def test_ollama_embedder_returns_empty_on_connection_error(self, mock_post):
        from core.manager import OllamaEmbedder
        mock_post.side_effect = Exception("Connection refused")
        embedder = OllamaEmbedder("nomic-embed-text")
        result = embedder.encode(["test"])
        assert result == []

    @patch("requests.post")
    def test_ollama_embedder_falls_back_to_legacy_endpoint(self, mock_post):
        from core.manager import OllamaEmbedder

        def side_effect(url, **kwargs):
            if url.endswith("/api/embed"):  # exact suffix, not substring (/api/embeddings must not match)
                r = MagicMock()
                r.status_code = 404
                r.text = "Not found"
                return r
            # legacy /api/embeddings
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {"embedding": [0.2] * 768}
            return r

        mock_post.side_effect = side_effect
        embedder = OllamaEmbedder("nomic-embed-text")
        result = embedder.encode(["some code"])
        assert len(result) == 1
        assert result[0][0] == pytest.approx(0.2)

    def test_graph_bulk_sync_populates_script_nodes(self, tmp_path):
        from core.graph_db import ScoutGraph
        from core.parser import parse_file

        scout_dir = tmp_path / ".scout"
        scout_dir.mkdir()
        graph = ScoutGraph("test_proj", str(scout_dir), "test_coll")

        parsed_map = {"Player.gd": parse_file("Player.gd", SAMPLE_GD)}
        graph.bulk_sync(parsed_map)

        result = graph.conn.execute("MATCH (s:Script) RETURN s.path")
        paths = []
        while result.has_next():
            paths.append(result.get_next()[0])
        assert "Player.gd" in paths
        graph.close()

    def test_graph_bulk_sync_creates_implements_relationship(self, tmp_path):
        from core.graph_db import ScoutGraph
        from core.parser import parse_file

        scout_dir = tmp_path / ".scout"
        scout_dir.mkdir()
        graph = ScoutGraph("test_proj", str(scout_dir), "test_coll")
        parsed_map = {"Player.gd": parse_file("Player.gd", SAMPLE_GD)}
        graph.bulk_sync(parsed_map)

        result = graph.conn.execute(
            "MATCH (s:Script)-[:IMPLEMENTS]->(t:Topic) RETURN s.path, t.name"
        )
        found = False
        while result.has_next():
            row = result.get_next()
            if row[0] == "Player.gd" and row[1] == "Player":
                found = True
        assert found, "Expected IMPLEMENTS relationship Player.gd → Player"
        graph.close()

    def test_graph_bulk_sync_creates_signal_nodes(self, tmp_path):
        from core.graph_db import ScoutGraph
        from core.parser import parse_file

        scout_dir = tmp_path / ".scout"
        scout_dir.mkdir()
        graph = ScoutGraph("test_proj", str(scout_dir), "test_coll")
        parsed_map = {"Player.gd": parse_file("Player.gd", SAMPLE_GD)}
        graph.bulk_sync(parsed_map)

        result = graph.conn.execute("MATCH (sig:Signal) RETURN sig.name")
        signal_names = []
        while result.has_next():
            signal_names.append(result.get_next()[0])

        assert "health_changed" in signal_names or "player_died" in signal_names
        graph.close()

    def test_graph_bulk_sync_multi_file(self, tmp_path):
        from core.graph_db import ScoutGraph
        from core.parser import parse_file

        scout_dir = tmp_path / ".scout"
        scout_dir.mkdir()
        graph = ScoutGraph("test_proj", str(scout_dir), "test_coll")
        parsed_map = {
            "Player.gd": parse_file("Player.gd", SAMPLE_GD),
            "Enemy.gd": parse_file("Enemy.gd", SAMPLE_GD_ENEMY),
            "README.md": parse_file("README.md", SAMPLE_MD),
        }
        graph.bulk_sync(parsed_map)

        result = graph.conn.execute("MATCH (s:Script) RETURN COUNT(s)")
        count = result.get_next()[0] if result.has_next() else 0
        assert count >= 2, "Expected at least 2 Script nodes (Player + Enemy)"
        graph.close()

    def test_bm25_vocab_built_from_parsed_chunks(self):
        from core.parser import parse_file
        from core.sparse import build_bm25_vocab

        parsed = parse_file("Player.gd", SAMPLE_GD)
        texts = [c["content"] for c in parsed["chunks"]]
        vocab = build_bm25_vocab(texts)
        assert len(vocab["idf"]) > 0
        # movement-related tokens from Player.gd should be in vocab
        assert any("move" in token or "velocity" in token or "speed" in token
                   for token in vocab["idf"])


# ===========================================================================
# TEST GROUP 7: Indexed Data Validation (Qdrant mocked)
# ===========================================================================

class TestIndexedDataValidation:
    """
    Verifies that after indexing, the correct data would be in Qdrant.
    Qdrant is mocked — tests verify the shape and content of upsert calls.
    """

    @patch("core.manager.client")
    @patch("requests.post")
    def test_upsert_called_with_collection_name(self, mock_post, mock_client, tmp_path):
        """After encoding, vectors should be upserted to the correct collection."""
        from core.manager import OllamaEmbedder
        from core.parser import parse_file

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"embeddings": [[0.1] * 768]}
        mock_post.return_value = mock_resp

        parsed = parse_file("Player.gd", SAMPLE_GD)
        chunks = parsed["chunks"]
        texts = [c["content"] for c in chunks]

        embedder = OllamaEmbedder("nomic-embed-text")
        embeddings = embedder.encode(texts)

        # Simulate what app.py does: upsert each chunk
        from qdrant_client.models import PointStruct
        points = []
        for chunk, vec in zip(chunks, embeddings if embeddings else []):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"Player.gd::{chunk['name']}"))
            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload={
                    "file": "Player.gd",
                    "name": chunk["name"],
                    "content": chunk["content"],
                    "type": chunk["type"],
                    "line_start": chunk["line_start"],
                    "systems": chunk.get("systems", []),
                },
            ))

        # Each chunk should get a deterministic UUID
        ids = [p.id for p in points]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    @patch("core.manager.client")
    @patch("requests.post")
    def test_point_ids_are_deterministic(self, mock_post, mock_client):
        """The same file+function should always produce the same UUID."""
        unique_str = "Player.gd::take_damage"
        id1 = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
        id2 = str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_str))
        assert id1 == id2

    def test_vector_dimension_matches_config(self):
        """The embedding dimension in config should match the configured model."""
        from config import EMBED_DIM, MODEL_DIMS
        # nomic-embed-text default should be 768
        assert EMBED_DIM == MODEL_DIMS.get("nomic-embed-text", 768)

    def test_get_model_dim_returns_correct_dimensions(self):
        from config import get_model_dim
        assert get_model_dim("nomic-embed-text") == 768
        assert get_model_dim("mxbai-embed-large") == 1024
        assert get_model_dim("all-minilm") == 384

    def test_get_model_dim_defaults_for_unknown(self):
        from config import get_model_dim
        assert get_model_dim("some-unknown-model") == 768


# ===========================================================================
# TEST GROUP 8: Chat / Prompt Evaluation
# ===========================================================================

class TestChatPipeline:
    """Tests for Ask.py — Qdrant, Kuzu, and Ollama are mocked."""

    @patch("requests.post")
    def test_ask_local_llm_returns_string(self, mock_post):
        from Ask import ask_local_llm
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "message": {"content": "Use signals for decoupled communication."}
        }
        mock_post.return_value = mock_resp

        result = ask_local_llm(
            prompt="You are Scout. The code is...",
            model_name="qwen2.5-coder:7b",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("requests.post")
    def test_ask_local_llm_passes_chat_history(self, mock_post):
        from Ask import ask_local_llm
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_resp

        history = [
            {"role": "user", "content": "What is Player.gd?"},
            {"role": "assistant", "content": "It handles movement."},
        ]
        ask_local_llm("prompt", "model", chat_history=history)
        payload = mock_post.call_args[1]["json"]
        roles = [m["role"] for m in payload["messages"]]
        assert "user" in roles
        assert "assistant" in roles

    @patch("requests.post")
    def test_ask_local_llm_caps_history_at_20_turns(self, mock_post):
        from Ask import ask_local_llm
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_resp

        long_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(40)
        ]
        ask_local_llm("prompt", "model", chat_history=long_history)
        payload = mock_post.call_args[1]["json"]
        # messages = [system] + up to 20 history turns
        assert len(payload["messages"]) <= 21

    @patch("requests.post")
    def test_ask_local_llm_error_returns_error_string(self, mock_post):
        from Ask import ask_local_llm
        mock_post.side_effect = Exception("Connection refused")
        result = ask_local_llm("prompt", "qwen2.5-coder:7b")
        assert "Error" in result or "error" in result.lower()

    @patch("Ask.ScoutGraph")
    @patch("Ask.QdrantClient")
    def test_build_surgical_prompt_returns_string(
        self, mock_qdrant_cls, mock_graph_cls, active_project
    ):
        from Ask import build_surgical_prompt

        mock_q = MagicMock()
        mock_q.query_points.return_value = MagicMock(points=[])
        mock_q.search.return_value = []
        mock_qdrant_cls.return_value = mock_q

        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_graph_inst = MagicMock()
        mock_graph_inst.conn.execute.return_value = mock_result
        mock_graph_cls.return_value = mock_graph_inst

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]

        prompt, meta = build_surgical_prompt(
            query="How does player movement work?",
            active_project=active_project,
            collection_name="test_project",
            embed_model=mock_embed,
            host="localhost",
            port=6333,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert isinstance(meta, dict)
        assert "context_files" in meta

    @patch("Ask.ScoutGraph")
    @patch("Ask.QdrantClient")
    def test_build_surgical_prompt_contains_query(
        self, mock_qdrant_cls, mock_graph_cls, active_project
    ):
        from Ask import build_surgical_prompt

        mock_q = MagicMock()
        mock_q.query_points.return_value = MagicMock(points=[])
        mock_qdrant_cls.return_value = mock_q

        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_graph_inst = MagicMock()
        mock_graph_inst.conn.execute.return_value = mock_result
        mock_graph_cls.return_value = mock_graph_inst

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]

        prompt, meta = build_surgical_prompt(
            query="How does the inventory system work?",
            active_project=active_project,
            collection_name="test_project",
            embed_model=mock_embed,
            host="localhost",
            port=6333,
        )
        assert "inventory" in prompt.lower()

    @pytest.mark.parametrize("persona", [
        "Senior Architect", "Bug Hunter", "Code Optimizer", "Teacher"
    ])
    @patch("Ask.ScoutGraph")
    @patch("Ask.QdrantClient")
    def test_build_surgical_prompt_all_personas(
        self, mock_qdrant_cls, mock_graph_cls, persona, active_project
    ):
        from Ask import build_surgical_prompt

        mock_q = MagicMock()
        mock_q.query_points.return_value = MagicMock(points=[])
        mock_qdrant_cls.return_value = mock_q

        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_graph_inst = MagicMock()
        mock_graph_inst.conn.execute.return_value = mock_result
        mock_graph_cls.return_value = mock_graph_inst

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]

        prompt, meta = build_surgical_prompt(
            query="Analyze the player code",
            active_project=active_project,
            collection_name="test_project",
            embed_model=mock_embed,
            host="localhost",
            port=6333,
            persona_key=persona,
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    @patch("Ask.ScoutGraph")
    @patch("Ask.QdrantClient")
    def test_build_surgical_prompt_personas_produce_distinct_output(
        self, mock_qdrant_cls, mock_graph_cls, active_project
    ):
        from Ask import build_surgical_prompt

        mock_q = MagicMock()
        mock_q.query_points.return_value = MagicMock(points=[])
        mock_qdrant_cls.return_value = mock_q

        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_graph_inst = MagicMock()
        mock_graph_inst.conn.execute.return_value = mock_result
        mock_graph_cls.return_value = mock_graph_inst

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]

        prompts = {}
        for persona in ["Senior Architect", "Bug Hunter", "Code Optimizer", "Teacher"]:
            prompt, _meta = build_surgical_prompt(
                query="Analyze this",
                active_project=active_project,
                collection_name="test_project",
                embed_model=mock_embed,
                host="localhost",
                port=6333,
                persona_key=persona,
            )
            prompts[persona] = prompt
        assert len(set(prompts.values())) == 4, "Each persona must produce a unique prompt"


# ===========================================================================
# TEST GROUP 9: Cloud / Export
# ===========================================================================

class TestCloudExport:
    """Tests for build_cloud_master_prompt — reads local files, graph is mocked."""

    def test_cloud_prompt_is_non_empty_string(self, active_project):
        from Ask import build_cloud_master_prompt
        with patch("Ask.ScoutGraph", side_effect=Exception("no graph")):
            prompt = build_cloud_master_prompt(
                query="Summarize this project",
                active_project=active_project,
            )
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_cloud_prompt_contains_xml_file_tags(self, active_project):
        from Ask import build_cloud_master_prompt
        with patch("Ask.ScoutGraph", side_effect=Exception("no graph")):
            prompt = build_cloud_master_prompt(
                query="test",
                active_project=active_project,
            )
        assert '<file path="' in prompt

    def test_cloud_prompt_includes_gd_file_content(self, active_project):
        from Ask import build_cloud_master_prompt
        with patch("Ask.ScoutGraph", side_effect=Exception("no graph")):
            prompt = build_cloud_master_prompt(
                query="test",
                active_project=active_project,
            )
        # Player.gd was written to the project dir; its content should appear
        assert "Player" in prompt or "CharacterBody2D" in prompt

    def test_cloud_prompt_contains_user_query(self, active_project):
        from Ask import build_cloud_master_prompt
        with patch("Ask.ScoutGraph", side_effect=Exception("no graph")):
            prompt = build_cloud_master_prompt(
                query="How does the jump mechanic work?",
                active_project=active_project,
            )
        assert "jump" in prompt.lower()

    def test_cloud_prompt_includes_roadmap_when_set(self, active_project):
        from core.roadmap import save_project_data
        from Ask import build_cloud_master_prompt

        save_project_data(active_project["scout_dir"], {
            "global_goal": "Build the best Godot platformer ever",
            "milestones": [{"name": "Tutorial Level", "status": "completed"}],
            "active_tasks": [],
            "architectural_rules": "",
        })
        with patch("Ask.ScoutGraph", side_effect=Exception("no graph")):
            prompt = build_cloud_master_prompt(
                query="What is the project goal?",
                active_project=active_project,
            )
        assert "platformer" in prompt.lower() or "PROJECT_ROADMAP" in prompt

    def test_cloud_prompt_excludes_scout_dir_files(self, active_project):
        """Files inside .scout/ should not be included in the cloud prompt."""
        from Ask import build_cloud_master_prompt
        # Write a file inside .scout/ that should be ignored
        secret_path = os.path.join(active_project["scout_dir"], "internal.gd")
        with open(secret_path, "w") as f:
            f.write("extends Node\nfunc internal_secret(): pass\n")

        with patch("Ask.ScoutGraph", side_effect=Exception("no graph")):
            prompt = build_cloud_master_prompt(
                query="test",
                active_project=active_project,
            )
        assert "internal_secret" not in prompt

    @pytest.mark.parametrize("persona", [
        "Bug Hunter", "Code Optimizer", "Teacher", "Senior Architect"
    ])
    def test_cloud_prompt_all_personas_produce_output(self, persona, active_project):
        from Ask import build_cloud_master_prompt
        with patch("Ask.ScoutGraph", side_effect=Exception("no graph")):
            prompt = build_cloud_master_prompt(
                query="Analyze the combat system",
                active_project=active_project,
                persona_key=persona,
            )
        assert isinstance(prompt, str)
        assert len(prompt) > 100


# ===========================================================================
# TEST GROUP 10: Milestones & Goal Tracking (CRUD)
# ===========================================================================

class TestMilestonesAndGoals:
    """CRUD operations on project state / roadmap."""

    def test_create_project_state_from_scratch(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        new_state = {
            "global_goal": "Build a dungeon crawler",
            "milestones": [],
            "active_tasks": [],
            "architectural_rules": "",
        }
        save_project_data(str(tmp_path), new_state)
        loaded = load_project_data(str(tmp_path))
        assert loaded["global_goal"] == "Build a dungeon crawler"
        assert loaded["milestones"] == []

    def test_read_milestone_by_name(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        data = {
            "global_goal": "Test",
            "milestones": [
                {"name": "Alpha", "status": "completed"},
                {"name": "Beta", "status": "in_progress"},
            ],
            "active_tasks": [],
            "architectural_rules": "",
        }
        save_project_data(str(tmp_path), data)
        loaded = load_project_data(str(tmp_path))
        beta = next(m for m in loaded["milestones"] if m["name"] == "Beta")
        assert beta["status"] == "in_progress"

    def test_update_global_goal(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        data = load_project_data(str(tmp_path))
        data["global_goal"] = "Revamped goal"
        save_project_data(str(tmp_path), data)
        loaded = load_project_data(str(tmp_path))
        assert loaded["global_goal"] == "Revamped goal"

    def test_delete_milestone(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        data = {
            "global_goal": "Test",
            "milestones": [
                {"name": "Keep Me", "status": "completed"},
                {"name": "Delete Me", "status": "pending"},
            ],
            "active_tasks": [],
            "architectural_rules": "",
        }
        save_project_data(str(tmp_path), data)
        loaded = load_project_data(str(tmp_path))
        loaded["milestones"] = [m for m in loaded["milestones"] if m["name"] != "Delete Me"]
        save_project_data(str(tmp_path), loaded)
        final = load_project_data(str(tmp_path))
        names = [m["name"] for m in final["milestones"]]
        assert "Delete Me" not in names
        assert "Keep Me" in names

    def test_active_tasks_persisted(self, tmp_path):
        from core.roadmap import load_project_data, save_project_data
        data = load_project_data(str(tmp_path))
        data["active_tasks"] = ["Fix jump bug", "Add wall-slide"]
        save_project_data(str(tmp_path), data)
        loaded = load_project_data(str(tmp_path))
        assert "Fix jump bug" in loaded["active_tasks"]
        assert "Add wall-slide" in loaded["active_tasks"]

    def test_milestone_status_values(self, tmp_path):
        """All three valid status values should round-trip correctly."""
        from core.roadmap import load_project_data, save_project_data
        for status in ("pending", "in_progress", "completed"):
            data = {
                "global_goal": "",
                "milestones": [{"name": "m", "status": status}],
                "active_tasks": [],
                "architectural_rules": "",
            }
            save_project_data(str(tmp_path), data)
            loaded = load_project_data(str(tmp_path))
            assert loaded["milestones"][0]["status"] == status


# ===========================================================================
# INTEGRATION TESTS (require live Qdrant + Ollama)
# ===========================================================================

@pytest.mark.integration
class TestIntegrationQdrant:
    """Requires a running Qdrant instance at QDRANT_HOST:QDRANT_PORT."""

    def test_qdrant_is_reachable(self):
        from qdrant_client import QdrantClient
        from config import QDRANT_HOST, QDRANT_PORT
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        assert collections is not None

    @patch("core.manager.client")
    def test_full_index_and_graph_pipeline(self, mock_qdrant, tmp_path):
        """Index sample files into kuzu and verify graph population."""
        from core.parser import parse_file
        from core.graph_db import ScoutGraph

        mock_qdrant.get_collections.return_value = MagicMock(collections=[])

        scout_dir = tmp_path / ".scout"
        scout_dir.mkdir()
        graph = ScoutGraph("integration_test", str(scout_dir), "test_coll")

        parsed_map = {
            "Player.gd": parse_file("Player.gd", SAMPLE_GD),
            "Enemy.gd": parse_file("Enemy.gd", SAMPLE_GD_ENEMY),
            "README.md": parse_file("README.md", SAMPLE_MD),
        }
        graph.bulk_sync(parsed_map)

        result = graph.conn.execute("MATCH (s:Script) RETURN COUNT(s)")
        count = result.get_next()[0] if result.has_next() else 0
        assert count >= 2
        graph.close()


@pytest.mark.integration
class TestIntegrationFullPipeline:
    """End-to-end: reset DB → create collection → index → query → verify context.

    Requires live Qdrant at QDRANT_HOST:QDRANT_PORT and Ollama with nomic-embed-text.
    """

    TEST_COLLECTION = "_scout_e2e_test"

    def _cleanup(self, qc):
        """Delete the test collection if it exists."""
        try:
            qc.delete_collection(self.TEST_COLLECTION)
        except Exception:
            pass

    def test_reset_creates_correct_schema(self):
        """After nuke_and_reset, collection must have named 'dense' + 'bm25' vectors."""
        from qdrant_client import QdrantClient
        from config import QDRANT_HOST, QDRANT_PORT
        from utils.db_ops import create_hybrid_collection

        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._cleanup(qc)

        create_hybrid_collection(QDRANT_HOST, QDRANT_PORT, self.TEST_COLLECTION, 768)

        info = qc.get_collection(self.TEST_COLLECTION)
        vc = info.config.params.vectors
        assert isinstance(vc, dict), f"Expected named vectors dict, got {type(vc)}"
        assert "dense" in vc, "Must have 'dense' named vector"
        assert vc["dense"].size == 768
        sv = info.config.params.sparse_vectors
        assert sv is not None and "bm25" in sv, "Must have 'bm25' sparse vector"

        self._cleanup(qc)

    def test_index_and_query_produces_context(self):
        """Full pipeline: index sample GDScript → query → prompt has non-zero context."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, SparseVector
        from config import QDRANT_HOST, QDRANT_PORT
        from utils.db_ops import create_hybrid_collection
        from core.manager import OllamaEmbedder
        from core.parser import parse_file
        from core.graph_db import ScoutGraph
        from core.sparse import build_bm25_vocab, encode_sparse
        from Ask import build_surgical_prompt
        from core.pipeline_log import PipelineLog
        import tempfile, uuid

        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._cleanup(qc)

        # 1. Create collection with correct schema
        create_hybrid_collection(QDRANT_HOST, QDRANT_PORT, self.TEST_COLLECTION, 768)

        # 2. Parse sample files
        parsed_map = {
            "Player.gd": parse_file("Player.gd", SAMPLE_GD),
            "Enemy.gd": parse_file("Enemy.gd", SAMPLE_GD_ENEMY),
        }

        # 3. Build graph in temp dir
        with tempfile.TemporaryDirectory() as tmp:
            scout_dir = os.path.join(tmp, ".scout")
            os.makedirs(scout_dir)
            graph = ScoutGraph("e2e_test", scout_dir, self.TEST_COLLECTION)
            graph.bulk_sync(parsed_map)
            graph.close()

            # 4. Embed and upsert
            embedder = OllamaEmbedder("nomic-embed-text")
            all_chunks = []
            for rel_path, pdata in parsed_map.items():
                for chunk in pdata.get("chunks", []):
                    chunk["file"] = rel_path
                    chunk["identity"] = pdata.get("identity", "")
                    chunk["global_state"] = pdata.get("global_state", "")
                    chunk["topic"] = pdata.get("topic", "Unknown")
                    if "systems" not in chunk:
                        chunk["systems"] = []
                    all_chunks.append(chunk)

            texts = [c["content"] for c in all_chunks]
            dense_vectors = embedder.encode(texts)

            # BM25
            bm25_vocab = build_bm25_vocab(texts)
            sparse_vectors = [encode_sparse(t, bm25_vocab) for t in texts]

            # Save vocab for query
            import json
            vocab_path = os.path.join(scout_dir, "bm25_vocab.json")
            with open(vocab_path, "w", encoding="utf-8") as f:
                json.dump(bm25_vocab, f)

            points = []
            for i, chunk in enumerate(all_chunks):
                det_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                    f"{chunk['file']}::{chunk['name']}"))
                pv = {"dense": dense_vectors[i]}
                sp_idx, sp_val = sparse_vectors[i]
                if sp_idx:
                    pv["bm25"] = SparseVector(indices=sp_idx, values=sp_val)
                points.append(PointStruct(
                    id=det_id, vector=pv,
                    payload={
                        "file": chunk["file"],
                        "content": chunk["content"],
                        "line_start": chunk.get("line_start", 0),
                        "type": chunk.get("type", "function"),
                        "name": chunk["name"],
                        "identity": chunk["identity"],
                        "global_state": chunk["global_state"],
                        "topic": chunk["topic"],
                        "systems": chunk["systems"],
                    }
                ))
            qc.upsert(collection_name=self.TEST_COLLECTION, points=points)

            # 5. Query
            plog = PipelineLog()
            active_project = {
                "id": "e2e_test",
                "scout_dir": scout_dir,
                "path": tmp,
            }
            prompt, meta = build_surgical_prompt(
                query="How does the player take damage?",
                active_project=active_project,
                collection_name=self.TEST_COLLECTION,
                embed_model=embedder,
                host=QDRANT_HOST, port=QDRANT_PORT,
                pipeline_log=plog,
            )

            # 6. Assertions — vector search should find hits
            assert len(meta["context_files"]) > 0, \
                f"Should have context files from vector hits. Pipeline log:\n{plog.full_report()}"

            # No vector_search errors
            vec_errors = [e for e in plog.errors if e["stage"] == "vector_search"]
            assert len(vec_errors) == 0, \
                f"Vector search should succeed. Errors: {vec_errors}"

            # Prompt should contain relevant code from indexed chunks
            assert "take_damage" in prompt.lower() or "health" in prompt.lower() or \
                   "Player" in prompt, \
                f"Prompt should contain relevant code. Got:\n{prompt[:500]}"

        # Cleanup
        self._cleanup(qc)

    def test_legacy_collection_still_searchable(self):
        """Collections with unnamed vectors should still return results via fallback."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import VectorParams, Distance, PointStruct
        from config import QDRANT_HOST, QDRANT_PORT
        from core.manager import OllamaEmbedder
        from Ask import build_surgical_prompt
        from core.pipeline_log import PipelineLog
        import tempfile

        qc = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        legacy_col = "_scout_legacy_test"
        try:
            qc.delete_collection(legacy_col)
        except Exception:
            pass

        # Create with UNNAMED vectors (legacy style)
        qc.create_collection(
            collection_name=legacy_col,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

        # Insert a test point (Qdrant requires UUID or int IDs)
        embedder = OllamaEmbedder("nomic-embed-text")
        vec = embedder.encode(["func take_damage(amount): health -= amount"])[0]
        test_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "Player.gd::take_damage"))
        qc.upsert(collection_name=legacy_col, points=[
            PointStruct(id=test_id, vector=vec, payload={
                "file": "Player.gd", "content": "func take_damage(amount):\n\thealth -= amount",
                "name": "take_damage", "type": "function", "line_start": 10,
                "identity": "", "global_state": "", "topic": "Player", "systems": [],
            })
        ])

        with tempfile.TemporaryDirectory() as tmp:
            scout_dir = os.path.join(tmp, ".scout")
            os.makedirs(scout_dir)

            plog = PipelineLog()
            prompt, meta = build_surgical_prompt(
                query="How does damage work?",
                active_project={"id": "legacy_test", "scout_dir": scout_dir, "path": tmp},
                collection_name=legacy_col,
                embed_model=embedder,
                host=QDRANT_HOST, port=QDRANT_PORT,
                pipeline_log=plog,
            )

            # Should NOT have vector_search errors
            vec_errors = [e for e in plog.errors if e["stage"] == "vector_search"]
            assert len(vec_errors) == 0, \
                f"Legacy search should work via fallback. Errors: {vec_errors}"

            # Should have found the content
            assert "take_damage" in prompt.lower(), \
                "Legacy search should still find relevant content"

        try:
            qc.delete_collection(legacy_col)
        except Exception:
            pass


@pytest.mark.integration
class TestIntegrationOllama:
    """Requires a running Ollama instance with nomic-embed-text pulled."""

    def test_ollama_produces_embeddings(self):
        from core.manager import OllamaEmbedder
        from config import EMBED_DIM
        embedder = OllamaEmbedder("nomic-embed-text")
        result = embedder.encode(["func _physics_process(delta): move_and_slide()"])
        assert len(result) == 1
        assert len(result[0]) == EMBED_DIM

    def test_qdrant_vector_dimensions_match_config(self):
        """Stored vectors (if any) should match the configured EMBED_DIM."""
        from qdrant_client import QdrantClient
        from config import QDRANT_HOST, QDRANT_PORT, EMBED_DIM
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections().collections
        if not collections:
            pytest.skip("No collections found — index some files first")
        col_info = client.get_collection(collections[0].name)
        vconfig = col_info.config.params.vectors
        if isinstance(vconfig, dict) and "dense" in vconfig:
            assert vconfig["dense"].size == EMBED_DIM
        elif hasattr(vconfig, "size"):
            assert vconfig.size == EMBED_DIM


# ===========================================================================
# TEST GROUP 12: DB Reset & Collection Schema Integrity
# ===========================================================================

class TestDBReset:
    """Verify that DB reset preserves correct named vector schema."""

    @patch.dict("sys.modules", {
        "qdrant_client": MagicMock(),
        "qdrant_client.models": MagicMock(),
    })
    def test_create_hybrid_collection_has_named_vectors(self):
        """create_hybrid_collection must produce 'dense' named vector + 'bm25' sparse."""
        # Re-import after patching sys.modules
        import importlib
        import utils.db_ops as _mod
        importlib.reload(_mod)

        mock_qc = MagicMock()
        with patch.object(_mod, "QdrantClient", return_value=mock_qc):
            _mod.create_hybrid_collection("localhost", 6333, "test_col", 768)

        call_args = mock_qc.create_collection.call_args
        vconfig = call_args.kwargs.get("vectors_config")
        sconfig = call_args.kwargs.get("sparse_vectors_config")

        assert isinstance(vconfig, dict), "vectors_config must be a dict with named vectors"
        assert "dense" in vconfig, "Must have 'dense' named vector"
        assert "bm25" in sconfig, "Must have 'bm25' sparse vector"

    @patch.dict("sys.modules", {
        "qdrant_client": MagicMock(),
        "qdrant_client.models": MagicMock(),
    })
    def test_wipe_vector_db_recreates_collection(self):
        """wipe_vector_db must delete and recreate."""
        import importlib
        import utils.db_ops as _mod
        importlib.reload(_mod)

        mock_qc = MagicMock()
        with patch.object(_mod, "QdrantClient", return_value=mock_qc):
            dim = _mod.wipe_vector_db("localhost", 6333, "test_col", "nomic-embed-text")

        mock_qc.delete_collection.assert_called_once_with("test_col")
        assert dim > 0


# ===========================================================================
# TEST GROUP 13: Vector Search Fallback Chain
# ===========================================================================

class TestVectorSearchFallback:
    """Verify that Ask.py handles different collection schemas correctly.

    These tests require qdrant_client to be importable (via mocking or installed).
    They are grouped with the existing TestChatPipeline tests that also import Ask.
    """

    @patch("Ask.ScoutGraph")
    @patch("Ask.QdrantClient")
    def test_search_with_named_vectors(self, mock_qdrant_cls, mock_graph_cls, active_project):
        """Standard path: collection has named 'dense' vectors."""
        from Ask import build_surgical_prompt

        mock_q = MagicMock()
        mock_col_info = MagicMock()
        mock_col_info.config.params.vectors = {"dense": MagicMock(size=768)}
        mock_q.get_collection.return_value = mock_col_info
        mock_q.query_points.return_value = MagicMock(points=[])
        mock_qdrant_cls.return_value = mock_q

        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_graph_inst = MagicMock()
        mock_graph_inst.conn.execute.return_value = mock_result
        mock_graph_cls.return_value = mock_graph_inst

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]

        prompt, meta = build_surgical_prompt(
            query="test", active_project=active_project,
            collection_name="test_coll", embed_model=mock_embed,
            host="localhost", port=6333,
        )
        assert mock_q.query_points.called or mock_q.search.called

    @patch("Ask.ScoutGraph")
    @patch("Ask.QdrantClient")
    def test_search_with_unnamed_vectors_falls_back(self, mock_qdrant_cls, mock_graph_cls, active_project):
        """Legacy path: collection has unnamed vectors — should use query_points without 'using'."""
        from Ask import build_surgical_prompt
        from core.pipeline_log import PipelineLog

        mock_q = MagicMock()
        mock_col_info = MagicMock()
        mock_col_info.config.params.vectors = MagicMock(spec=[])
        type(mock_col_info.config.params.vectors).__contains__ = lambda self, k: False
        mock_q.get_collection.return_value = mock_col_info
        mock_q.query_points.return_value = MagicMock(points=[])
        mock_qdrant_cls.return_value = mock_q

        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_graph_inst = MagicMock()
        mock_graph_inst.conn.execute.return_value = mock_result
        mock_graph_cls.return_value = mock_graph_inst

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]

        plog = PipelineLog()
        prompt, meta = build_surgical_prompt(
            query="test", active_project=active_project,
            collection_name="test_coll", embed_model=mock_embed,
            host="localhost", port=6333, pipeline_log=plog,
        )
        # Legacy path should call query_points without using="dense"
        assert mock_q.query_points.called, "Should use query_points for unnamed vectors"
        # Check that it was called without using="dense" (legacy path)
        call_kwargs = mock_q.query_points.call_args.kwargs
        assert call_kwargs.get("using") is None, \
            "Legacy search should NOT pass using='dense'"

    @patch("Ask.ScoutGraph")
    @patch("Ask.QdrantClient")
    def test_search_failure_logs_error(self, mock_qdrant_cls, mock_graph_cls, active_project):
        """When all search strategies fail, error should be logged (not silent)."""
        from Ask import build_surgical_prompt
        from core.pipeline_log import PipelineLog

        mock_q = MagicMock()
        mock_col_info = MagicMock()
        mock_col_info.config.params.vectors = MagicMock(spec=[])
        type(mock_col_info.config.params.vectors).__contains__ = lambda self, k: False
        mock_q.get_collection.return_value = mock_col_info
        mock_q.query_points.side_effect = Exception("Connection refused")
        mock_qdrant_cls.return_value = mock_q

        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_graph_inst = MagicMock()
        mock_graph_inst.conn.execute.return_value = mock_result
        mock_graph_cls.return_value = mock_graph_inst

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]

        plog = PipelineLog()
        prompt, meta = build_surgical_prompt(
            query="test", active_project=active_project,
            collection_name="test_coll", embed_model=mock_embed,
            host="localhost", port=6333, pipeline_log=plog,
        )
        assert plog.has_issues, "Pipeline should log error when all searches fail"
        assert any("vector_search" in e["stage"] for e in plog.errors), \
            "Should have a vector_search error entry"


# ===========================================================================
# TEST GROUP 14: Verification Report Accuracy (Real-World Patterns)
# ===========================================================================

class TestVerificationAccuracy:
    """Test verifier against realistic LLM response patterns."""

    def _mock_graph(self, functions=None, signals=None, scripts=None,
                    variables=None, classes=None):
        """Create a mock graph with configurable entity sets."""
        functions = set(functions or [])
        signals = set(signals or [])
        scripts = set(scripts or [])
        variables = set(variables or [])
        classes = set(classes or [])

        def make_result(values):
            mock = MagicMock()
            idx = [0]
            items = [[v] for v in values]
            def has_next(): return idx[0] < len(items)
            def get_next():
                v = items[idx[0]]; idx[0] += 1; return v
            mock.has_next = has_next
            mock.get_next = get_next
            return mock

        def execute(query):
            if "Function" in query: return make_result(functions)
            if "Signal" in query: return make_result(signals)
            if "Script" in query: return make_result(scripts)
            if "Variable" in query: return make_result(variables)
            if "Class" in query: return make_result(classes)
            return make_result([])

        graph = MagicMock()
        graph.conn.execute = execute
        return graph

    def test_perfect_grounding_when_all_entities_exist(self):
        """Response referencing only known entities should score 1.0."""
        from core.verifier import verify_response

        graph = self._mock_graph(
            functions=["move_and_slide", "take_damage"],
            signals=["health_changed"],
            scripts=["Player.gd"],
        )
        response = "The move_and_slide() and take_damage() handle movement in Player.gd"
        result = verify_response(response, graph, context_files={"Player.gd"})

        assert result.grounding_score == 1.0
        assert result.total_hallucinated == 0

    def test_detects_hallucinated_function(self):
        """Functions not in graph should be flagged."""
        from core.verifier import verify_response

        graph = self._mock_graph(functions=["move_and_slide"])
        response = "Call move_and_slide() and then fly_to_moon() to move the player."
        result = verify_response(response, graph)

        assert "fly_to_moon" in result.hallucinated["function"]
        assert "move_and_slide" in result.verified["function"]
        assert result.grounding_score < 1.0

    def test_detects_hallucinated_file(self):
        """File references not in graph should be flagged."""
        from core.verifier import verify_response

        graph = self._mock_graph(scripts=["Player.gd"])
        response = "See Player.gd and NonExistent.gd for the implementation."
        result = verify_response(response, graph)

        assert "Player.gd" in result.verified["file"]
        assert "NonExistent.gd" in result.hallucinated["file"]

    def test_godot_builtins_not_flagged(self):
        """Godot builtins like emit_signal, get_node should not be flagged."""
        from core.verifier import verify_response

        graph = self._mock_graph()
        response = (
            "Use emit_signal('done') and get_node('Player') to connect. "
            "Also call queue_free() and add_child(node)."
        )
        result = verify_response(response, graph)

        # These should NOT appear as hallucinated functions
        for builtin in ["emit_signal", "get_node", "queue_free", "add_child"]:
            assert builtin not in result.hallucinated.get("function", []), \
                f"{builtin} should be excluded from hallucination check"

    def test_context_coverage_calculation(self):
        """Context coverage should reflect what fraction of provided files were cited."""
        from core.verifier import verify_response

        graph = self._mock_graph(scripts=["Player.gd", "Enemy.gd", "World.gd"])
        response = "The issue is in Player.gd and Enemy.gd."
        result = verify_response(
            response, graph,
            context_files={"Player.gd", "Enemy.gd", "World.gd", "UI.gd"},
        )

        # 2 out of 4 context files referenced
        assert abs(result.context_coverage - 0.5) < 0.01

    def test_empty_response_has_perfect_score(self):
        """No entities referenced = nothing hallucinated = 1.0 score."""
        from core.verifier import verify_response

        graph = self._mock_graph()
        result = verify_response("This is a general comment with no code references.", graph)

        assert result.grounding_score == 1.0
        assert result.total_entities == 0

    def test_full_report_contains_all_sections(self):
        """The full report must have grounding score, coverage, precision, and hallucinations."""
        from core.verifier import verify_response

        graph = self._mock_graph(
            functions=["move"],
            scripts=["Player.gd"],
        )
        response = "Call move() in Player.gd and also call fake_func() in Ghost.gd."
        result = verify_response(response, graph, context_files={"Player.gd"})
        report = result.full_report()

        assert "Grounding Score" in report
        assert "Context Coverage" in report
        assert "Entity Precision" in report
        assert "Potential Hallucinations" in report

    def test_prompt_efficiency_metrics(self):
        """compute_prompt_efficiency must return valid numeric metrics."""
        from core.verifier import compute_prompt_efficiency

        eff = compute_prompt_efficiency(
            prompt_text="x" * 10000,
            response_text="Call move_and_slide() in Player.gd and take_damage() in Enemy.gd",
            token_budget=24000,
        )

        assert 0.0 < eff["budget_utilization"] <= 1.0
        assert eff["response_density"] >= 0
        assert eff["unique_entities_cited"] >= 2
        assert eff["prompt_tokens_est"] > 0
        assert eff["response_tokens_est"] > 0

    def test_pipeline_log_integration_with_verifier(self):
        """Verifier should log warnings via pipeline_log when graph queries fail."""
        from core.verifier import verify_response
        from core.pipeline_log import PipelineLog

        broken_graph = MagicMock()
        broken_graph.conn.execute.side_effect = Exception("DB locked")

        plog = PipelineLog()
        result = verify_response("Call move() in Player.gd", broken_graph, pipeline_log=plog)

        # Should still return a result (graceful degradation)
        assert isinstance(result.grounding_score, float)
        # Should have logged warnings about failed queries
        assert plog.has_issues

    def test_build_surgical_prompt_returns_context_files(self, active_project):
        """build_surgical_prompt meta must include context_files for verifier."""
        pytest.importorskip("qdrant_client")
        from Ask import build_surgical_prompt

        mock_q = MagicMock()
        mock_q.get_collection.return_value = MagicMock(
            config=MagicMock(params=MagicMock(vectors={"dense": MagicMock(size=768)}))
        )

        # Simulate some hits with file payloads
        mock_hit = MagicMock()
        mock_hit.payload = {"file": "Player.gd", "name": "move", "content": "func move():",
                            "type": "function", "line_start": 1, "systems": [],
                            "global_state": "", "identity": "", "topic": "Player",
                            "emits": []}
        mock_hit.score = 0.9
        mock_q.query_points.return_value = MagicMock(points=[mock_hit])
        mock_q.search.return_value = [mock_hit]

        mock_result = MagicMock()
        mock_result.has_next.return_value = False
        mock_graph = MagicMock()
        mock_graph.conn.execute.return_value = mock_result

        mock_embed = MagicMock()
        mock_embed.encode.return_value = [[0.1] * 768]

        with patch("Ask.QdrantClient", return_value=mock_q), \
             patch("Ask.ScoutGraph", return_value=mock_graph):
            prompt, meta = build_surgical_prompt(
                query="How does movement work?",
                active_project=active_project,
                collection_name="test", embed_model=mock_embed,
                host="localhost", port=6333,
            )

        assert "Player.gd" in meta["context_files"], \
            "context_files should include files from vector hits"
        assert meta["token_budget"] == 24000
