# ruff: noqa: E501
"""Markdown rendering is shared by the live and saved viewers."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest


def test_markdown_builds_safe_dom_for_common_markup() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is not installed")
    path = Path(__file__).resolve().parents[2] / "scripts/coursework_markdown.py"
    spec = importlib.util.spec_from_file_location("coursework_markdown_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setup = r"""
class Node {
 constructor(tag, text=''){this.tag=tag;this.textContent=text;this.children=[];this.dataset={};}
 append(...nodes){this.children.push(...nodes)}
 replaceChildren(...nodes){this.children=[...nodes]}
}
const document={createElement: tag=>new Node(tag),createTextNode: text=>new Node('#text',text)};
"""
    checks = r"""
const root=new Node('div');
renderMarkdown(root, '# Заголовок\n\n**жирный** и *курсив* и `код`\n\n- один\n  - вложенный\n- два\n\n```js\n<script>alert(1)</script>\n```\n\n| Имя | Значение |\n| --- | --- |\n| a | b |\n\n[опасно](javascript:alert) [ссылка](https://example.org)');
const text=n=>n.tag==='#text'?n.textContent:n.textContent+n.children.map(text).join('');
const all=n=>[n,...n.children.flatMap(all)];
const nodes=all(root);
if(!['h1','strong','em','code','ul','li','pre','table','th','td','a'].every(t=>nodes.some(n=>n.tag===t)))throw new Error('missing Markdown element');
if(!nodes.some(n=>n.tag==='li'&&n.children.some(c=>c.tag==='ul')))throw new Error('nested list missing');
if(nodes.some(n=>n.tag==='script'))throw new Error('HTML injection');
if(!text(root).includes('<script>alert(1)</script>'))throw new Error('code text missing');
if(nodes.filter(n=>n.tag==='a').length!==1||!nodes.find(n=>n.tag==='a').href.startsWith('https:'))throw new Error('unsafe link');
renderMarkdown(root,'новый ответ');
if(text(root)!=='новый ответ')throw new Error('replacement failed');
"""
    subprocess.run(
        [node, "-e", setup + module.MARKDOWN_JS + checks],
        check=True,
        capture_output=True,
        text=True,
    )
