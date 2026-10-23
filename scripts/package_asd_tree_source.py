"""Export exact committed Git blobs, avoiding checkout EOL conversions."""
import hashlib
import argparse
import json
from pathlib import Path
import subprocess
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="artifacts/asd-budget-tree-source.zip")
    parser.add_argument("--manifest",default="ASD_TREE_SOURCE_MANIFEST.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
    raw = subprocess.check_output(["git","ls-tree","-r","-z",commit],cwd=root)
    entries = []
    for entry in raw.split(b"\0"):
        if entry:
            metadata,name = entry.split(b"\t",1)
            mode,kind,oid = metadata.split()
            if kind != b"blob" or mode not in (b"100644",b"100755"):
                raise ValueError("Unexpected Git source entry")
            entries.append((name.decode(),oid))
    data = subprocess.run(["git","cat-file","--batch"],cwd=root,
        input=b"".join(oid+b"\n" for _,oid in entries),stdout=subprocess.PIPE,check=True).stdout
    archive = root/args.output
    if not archive.resolve().is_relative_to(root) or Path(args.manifest).name != args.manifest:
        raise ValueError("Source package must stay within the workspace")
    files,offset = {},0
    with zipfile.ZipFile(archive,"x",zipfile.ZIP_DEFLATED) as z:
        for name,oid in entries:
            end = data.index(b"\n",offset)
            header = data[offset:end].split()
            if header[:2] != [oid,b"blob"]:
                raise ValueError("Git batch source mismatch")
            size = int(header[2])
            content = data[end+1:end+1+size]
            offset = end+size+2
            files[name] = hashlib.sha256(content).hexdigest()
            z.writestr(name,content)
        manifest = dict(commit=commit,files=files,
            source_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
            source_bytes="canonical Git blobs; no git archive/checkout conversion")
        encoded = (json.dumps(manifest,indent=2)+"\n").encode()
        z.writestr(args.manifest,encoded)
    (root/"artifacts"/args.manifest).write_bytes(encoded)
    print(json.dumps(dict(commit=commit,files=len(files),source_sha256=manifest["source_sha256"],
        zip_bytes=archive.stat().st_size,zip_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))


if __name__ == "__main__":main()
