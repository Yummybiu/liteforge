"""死导入扫描（make lint 用）：AST 解析 + 全文计数。"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "liteforge"


def check(path: Path) -> list:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [(a.asname or a.name).split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += [a.asname or a.name for a in node.names]
    body_uses = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attr_uses = {n.value.id for n in ast.walk(tree)
                 if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    return [i for i in imported if i not in body_uses and i not in attr_uses
            and src.count(i) <= 1 and not i.startswith("_")]


def main() -> int:
    bad = 0
    for path in sorted(ROOT.rglob("*.py")):
        unused = check(path)
        if unused:
            print(f"{path.relative_to(ROOT.parent)}: 未使用导入 {unused}")
            bad += 1
    print("lint:", "发现问题" if bad else "通过")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
