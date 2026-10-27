import json
import zipfile

from scripts.install_chat_release import install
from scripts.package_chat_service import package


def test_source_release_installs_vector_brand_asset_without_private_deployment_files(tmp_path):
    root = tmp_path / "project"
    static = root / "memorydraft/chat/static"
    static.mkdir(parents=True)
    (static / "favicon.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    (root / "pyproject.toml").write_text('[project]\nname = "memorydraft"\n')
    (root / "README.md").write_text("Treeflow")
    private = root / "private"
    private.mkdir()
    (private / "gateway.env").write_text("placeholder-only")
    (static / "accidental.env").write_text("placeholder-only")
    archive = tmp_path / "release.zip"
    result = package(root, archive)
    with zipfile.ZipFile(archive) as bundle:
        assert "memorydraft/chat/static/favicon.svg" in bundle.namelist()
        assert not any(name.endswith(".env") for name in bundle.namelist())
        assert json.loads(bundle.read("SOURCE.json"))["sha256"] == result["source_sha256"]
    install(archive, tmp_path / "service", result["source_sha256"])
    asset = tmp_path / "service/releases" / result["source_sha256"][:12] / "memorydraft/chat/static/favicon.svg"
    assert asset.read_bytes() == (static / "favicon.svg").read_bytes()
